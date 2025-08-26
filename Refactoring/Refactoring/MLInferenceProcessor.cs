using System;
using System.Collections.Generic;
using System.Data.SqlClient;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Net.Mail;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;

namespace MLInferenceService
{
    public class MLInferenceProcessor
    {
        private static Dictionary<string, object> _modelCache = new Dictionary<string, object>();
        private static int _highFraudCount = 0;
        private static readonly HttpClient _httpClient = new HttpClient();
        
        public async Task<string> ProcessMLInferenceRequest(string requestData)
        {
            try
            {
                if (string.IsNullOrEmpty(requestData))
                {
                    return JsonSerializer.Serialize(new { error = "Invalid request" });
                }
                
                var request = JsonSerializer.Deserialize<Dictionary<string, object>>(requestData);
                
                if (!request.ContainsKey("user_id"))
                {
                    return JsonSerializer.Serialize(new { error = "User ID required" });
                }
                
                var userId = Convert.ToInt32(request["user_id"]);
                var modelName = request.ContainsKey("model") ? request["model"].ToString() : "default_classifier";
                var requestId = request.ContainsKey("request_id") ? request["request_id"].ToString() : DateTime.Now.Ticks.ToString();
                var features = request.ContainsKey("features") ? 
                    JsonSerializer.Deserialize<Dictionary<string, object>>(request["features"].ToString()) : 
                    new Dictionary<string, object>();
                
                // User validation and rate limiting
                var connectionString = "Server=prod-db.company.com;Database=MLService;User Id=sa;Password=Prod123!;";
                using var connection = new SqlConnection(connectionString);
                connection.Open();
                var query = "SELECT tier, requests_today, max_requests FROM users WHERE id = @userId";
                using (var command = new SqlCommand(query, connection))
                {
                    command.Parameters.AddWithValue("@userId", userId);
                    using (var reader = command.ExecuteReader())
                    {
                        if (!reader.Read())
                        {
                            return JsonSerializer.Serialize(new { error = "User not found" });
                        }

                        var tier = reader["tier"].ToString();
                        var requestsToday = Convert.ToInt32(reader["requests_today"]);
                        var maxRequests = Convert.ToInt32(reader["max_requests"]);

                        // Rate limiting logic
                        if (tier == "free" && requestsToday >= 100)
                        {
                            return JsonSerializer.Serialize(new { error = "Rate limit exceeded" });
                        }
                        else if (tier == "premium" && requestsToday >= 10000)
                        {
                            return JsonSerializer.Serialize(new { error = "Rate limit exceeded" });
                        }
                        else if (tier == "enterprise" && requestsToday >= 100000)
                        {
                            // Send alert email
                            var smtpClient = new SmtpClient("smtp.company.com", 587);
                            smtpClient.Credentials = new System.Net.NetworkCredential("admin", "prod_password_2024!");
                            smtpClient.EnableSsl = true;

                            var mailMessage = new MailMessage();
                            mailMessage.From = new MailAddress("alerts@company.com");
                            mailMessage.To.Add("admin@company.com");
                            mailMessage.Subject = "Rate Limit Alert";
                            mailMessage.Body = $"Enterprise user {userId} approaching limit";

                            smtpClient.Send(mailMessage);
                        }

                        // Model loading and processing
                        var modelKey = $"{modelName}_{tier}";
                        object prediction = null;

                        if (!_modelCache.ContainsKey(modelKey))
                        {
                            // Load model based on type
                            if (modelName == "fraud_detector")
                            {
                                var modelPath = $@"C:\Models\fraud\{tier}_model.json";
                                var scalerPath = $@"C:\Models\fraud\{tier}_scaler.json";

                                if (File.Exists(modelPath) && File.Exists(scalerPath))
                                {
                                    var modelJson = File.ReadAllText(modelPath);
                                    var scalerJson = File.ReadAllText(scalerPath);

                                    _modelCache[modelKey] = new
                                    {
                                        model = JsonSerializer.Deserialize<Dictionary<string, object>>(modelJson),
                                        scaler = JsonSerializer.Deserialize<Dictionary<string, object>>(scalerJson),
                                        type = "fraud"
                                    };
                                }
                                else
                                {
                                    return JsonSerializer.Serialize(new { error = "Model files not found" });
                                }
                            }
                            else if (modelName == "recommendation_engine")
                            {
                                var embeddingsPath = $@"C:\Models\recsys\{tier}_embeddings.json";
                                if (File.Exists(embeddingsPath))
                                {
                                    var embeddingsJson = File.ReadAllText(embeddingsPath);
                                    _modelCache[modelKey] = new
                                    {
                                        embeddings = JsonSerializer.Deserialize<double[][]>(embeddingsJson),
                                        type = "recsys"
                                    };
                                }
                            }
                            else if (modelName == "sentiment_analyzer")
                            {
                                var modelPath = $@"C:\Models\sentiment\{tier}\model.json";
                                if (File.Exists(modelPath))
                                {
                                    var modelJson = File.ReadAllText(modelPath);
                                    _modelCache[modelKey] = new
                                    {
                                        model = JsonSerializer.Deserialize<Dictionary<string, object>>(modelJson),
                                        type = "nlp"
                                    };
                                }
                            }
                            else
                            {
                                // Default classifier
                                var defaultPath = @"C:\Models\default\classifier.json";
                                if (File.Exists(defaultPath))
                                {
                                    var modelJson = File.ReadAllText(defaultPath);
                                    _modelCache[modelKey] = new
                                    {
                                        model = JsonSerializer.Deserialize<Dictionary<string, object>>(modelJson),
                                        type = "default"
                                    };
                                }
                            }
                        }

                        var cachedModel = _modelCache[modelKey];
                        var modelType = GetProperty(cachedModel, "type").ToString();

                        // Feature preprocessing and prediction
                        if (modelType == "fraud")
                        {
                            // Validate required features
                            var requiredFeatures = new[] { "amount", "merchant_category", "hour", "day_of_week", "user_history_score" };
                            if (!requiredFeatures.All(f => features.ContainsKey(f)))
                            {
                                return JsonSerializer.Serialize(new { error = "Missing required features for fraud detection" });
                            }

                            // Feature engineering
                            var amount = Convert.ToDouble(features["amount"]);
                            var hour = Convert.ToInt32(features["hour"]);
                            var dayOfWeek = Convert.ToInt32(features["day_of_week"]);
                            var merchantCategory = features["merchant_category"].ToString();
                            var userHistoryScore = Convert.ToDouble(features["user_history_score"]);

                            var amountLog = Math.Log(1 + amount);
                            var isWeekend = (dayOfWeek == 5 || dayOfWeek == 6) ? 1 : 0;
                            var isNight = (hour < 6 || hour > 22) ? 1 : 0;

                            // One-hot encoding for merchant category
                            var merchantCategories = new[] { "grocery", "gas", "restaurant", "online", "other" };
                            var merchantFeatures = new Dictionary<string, int>();
                            foreach (var cat in merchantCategories)
                            {
                                merchantFeatures[$"merchant_{cat}"] = (merchantCategory == cat) ? 1 : 0;
                            }

                            // Simple fraud probability calculation (mock ML model)
                            var fraudScore = amountLog * 0.1 + isNight * 0.3 + (1 - userHistoryScore) * 0.4;
                            if (merchantCategory == "online") fraudScore += 0.2;

                            var fraudProbability = 1.0 / (1.0 + Math.Exp(-fraudScore)); // Sigmoid
                            var isFraud = fraudProbability > 0.7;

                            prediction = new { fraud_probability = fraudProbability, is_fraud = isFraud };

                            // High-risk alert
                            if (fraudProbability > 0.9)
                            {
                                Task.Run(() => SendFraudAlert(userId, fraudProbability, requestId));
                            }

                            // Model drift detection
                            if (fraudProbability > 0.8)
                            {
                                Interlocked.Increment(ref _highFraudCount);
                                if (_highFraudCount > 1000)
                                {
                                    // Send drift alert
                                    var driftSmtp = new SmtpClient("smtp.company.com", 587);
                                    driftSmtp.Credentials = new System.Net.NetworkCredential("admin", "prod_password_2024!");
                                    driftSmtp.EnableSsl = true;

                                    var driftMail = new MailMessage();
                                    driftMail.From = new MailAddress("ml-ops@company.com");
                                    driftMail.To.Add("data-science@company.com");
                                    driftMail.Subject = "Model Drift Alert";
                                    driftMail.Body = "High fraud rate detected. Consider model retraining.";

                                    driftSmtp.Send(driftMail);
                                    Interlocked.Exchange(ref _highFraudCount, 0);
                                }
                            }
                        }
                        else if (modelType == "recsys")
                        {
                            if (!features.ContainsKey("user_profile") || !features.ContainsKey("item_interactions"))
                            {
                                return JsonSerializer.Serialize(new { error = "Missing interaction data" });
                            }

                            var userProfile = JsonSerializer.Deserialize<double[]>(features["user_profile"].ToString());
                            var itemEmbeddings = (double[][])GetProperty(cachedModel, "embeddings");

                            // Calculate similarities (simplified)
                            var similarities = new List<(int index, double similarity)>();
                            for (int i = 0; i < itemEmbeddings.Length; i++)
                            {
                                var similarity = DotProduct(userProfile, itemEmbeddings[i]);
                                similarities.Add((i, similarity));
                            }

                            var topItems = similarities.OrderByDescending(s => s.similarity).Take(10).ToList();
                            prediction = new
                            {
                                recommended_items = topItems.Select(t => t.index).ToArray(),
                                scores = topItems.Select(t => t.similarity).ToArray()
                            };
                        }
                        else if (modelType == "nlp")
                        {
                            if (!features.ContainsKey("text"))
                            {
                                return JsonSerializer.Serialize(new { error = "Text input required" });
                            }

                            var text = features["text"].ToString();
                            if (text.Length > 5000)
                            {
                                return JsonSerializer.Serialize(new { error = "Text too long" });
                            }

                            // Simple sentiment analysis (mock)
                            var positiveWords = new[] { "good", "great", "excellent", "amazing", "wonderful", "fantastic" };
                            var negativeWords = new[] { "bad", "terrible", "awful", "horrible", "disappointing", "poor" };

                            var words = text.ToLower().Split(' ');
                            var positiveCount = words.Count(w => positiveWords.Contains(w));
                            var negativeCount = words.Count(w => negativeWords.Contains(w));

                            var sentimentScore = (positiveCount - negativeCount + words.Length) / (double)(words.Length * 2);
                            prediction = new
                            {
                                sentiment_score = sentimentScore,
                                sentiment = sentimentScore > 0.5 ? "positive" : "negative"
                            };
                        }
                        else
                        {
                            // Default model
                            if (!features.ContainsKey("feature_vector"))
                            {
                                return JsonSerializer.Serialize(new { error = "Feature vector required" });
                            }

                            var featureVector = JsonSerializer.Deserialize<double[]>(features["feature_vector"].ToString());
                            var result = featureVector.Sum() > 0 ? 1 : 0; // Mock prediction
                            prediction = new { @class = result, confidence = 0.85 };
                        }

                        // Update usage statistics
                        var updateQuery = "UPDATE users SET requests_today = requests_today + 1 WHERE id = @userId";
                        using (var updateCommand = new SqlCommand(updateQuery, connection))
                        {
                            updateCommand.Parameters.AddWithValue("@userId", userId);
                            updateCommand.ExecuteNonQuery();
                        }

                        // Log inference request
                        var logQuery = @"INSERT INTO inference_logs 
                                           (user_id, model_name, request_id, timestamp, latency_ms, prediction) 
                                           VALUES (@userId, @modelName, @requestId, @timestamp, @latency, @prediction)";
                        using (var logCommand = new SqlCommand(logQuery, connection))
                        {
                            logCommand.Parameters.AddWithValue("@userId", userId);
                            logCommand.Parameters.AddWithValue("@modelName", modelName);
                            logCommand.Parameters.AddWithValue("@requestId", requestId);
                            logCommand.Parameters.AddWithValue("@timestamp", DateTime.Now);
                            logCommand.Parameters.AddWithValue("@latency", new Random().Next(10, 100));
                            logCommand.Parameters.AddWithValue("@prediction", JsonSerializer.Serialize(prediction));
                            logCommand.ExecuteNonQuery();
                        }

                        // A/B testing logic
                        if (userId % 10 == 0 && modelName == "fraud_detector")
                        {
                            var logPath = @"C:\Logs\ab_test_group_a.log";
                            var logEntry = $"{DateTime.Now},{userId},{requestId},{JsonSerializer.Serialize(prediction)}\n";
                            File.AppendAllText(logPath, logEntry);
                        }

                        // Memory management for model cache
                        if (_modelCache.Count > 50)
                        {
                            var oldestKey = _modelCache.Keys.First();
                            _modelCache.Remove(oldestKey);
                        }

                        return JsonSerializer.Serialize(new
                        {
                            request_id = requestId,
                            model = modelName,
                            prediction = prediction,
                            timestamp = DateTime.Now.ToString("O"),
                            user_tier = tier
                        });
                    }
                }
            }
            catch (Exception ex)
            {
                return JsonSerializer.Serialize(new { error = $"Processing failed: {ex.Message}" });
            }
        }
        
        private async Task SendFraudAlert(int userId, double probability, string requestId)
        {
            try
            {
                // Send to internal API
                var alertData = new { user_id = userId, probability = probability, request_id = requestId };
                var json = JsonSerializer.Serialize(alertData);
                var content = new StringContent(json, System.Text.Encoding.UTF8, "application/json");
                
                await _httpClient.PostAsync("http://fraud-alerts.internal.com/alert", content);
                
                // Send email alert
                var smtpClient = new SmtpClient("smtp.company.com", 587);
                smtpClient.Credentials = new System.Net.NetworkCredential("admin", "prod_password_2024!");
                smtpClient.EnableSsl = true;
                
                var mailMessage = new MailMessage();
                mailMessage.From = new MailAddress("fraud-alerts@company.com");
                mailMessage.To.Add("fraud-team@company.com");
                mailMessage.Subject = "URGENT: High Fraud Risk Detected";
                mailMessage.Body = $"High fraud risk: User {userId}, Probability: {probability:F3}";
                
                smtpClient.Send(mailMessage);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Alert sending failed: {ex.Message}");
            }
        }
        
        private double DotProduct(double[] a, double[] b)
        {
            if (a.Length != b.Length) return 0;
            return a.Zip(b, (x, y) => x * y).Sum();
        }
        
        private object GetProperty(object obj, string propertyName)
        {
            return obj.GetType().GetProperty(propertyName)?.GetValue(obj);
        }
    }
}

// Example usage:
// var processor = new MLInferenceProcessor();
// var request = JsonSerializer.Serialize(new
// {
//     user_id = 12345,
//     model = "fraud_detector",
//     features = new
//     {
//         amount = 1500.50,
//         merchant_category = "online",
//         hour = 23,
//         day_of_week = 5,
//         user_history_score = 0.85
//     },
//     request_id = "req_001"
// });
// var result = await processor.ProcessMLInferenceRequest(request);
