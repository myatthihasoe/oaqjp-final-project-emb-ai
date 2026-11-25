"""Return formatted scores from the Watson NLP emotion model."""
import requests

URL = (
    "https://sn-watson-emotion.labs.skills.network/v1/"
    "watson.runtime.nlp.v1/NlpService/EmotionPredict"
)
HEADERS = {
    "grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"
}


def emotion_detector(text_to_analyze):
    """Return five emotion scores and the highest-scoring emotion."""
    response = requests.post(
        URL,
        json={"raw_document": {"text": text_to_analyze}},
        headers=HEADERS,
        timeout=30,
    )
    response.raise_for_status()
    emotions = response.json()["emotionPredictions"][0]["emotion"]
    scores = {
        name: emotions[name]
        for name in ("anger", "disgust", "fear", "joy", "sadness")
    }
    scores["dominant_emotion"] = max(scores, key=scores.get)
    return scores
