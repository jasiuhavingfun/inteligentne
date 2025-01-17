import unittest
from app import app

class TestPlantDiseaseApp(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_homepage(self):
        response = self.app.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Upload a plant leaf image:", response.data)

    def test_prediction(self):
        with open(r"C:\Users\janzi\Desktop\inteligentne\static\uploads\8c87c6d58cb99c71.jpg", "rb") as img:
            response = self.app.post("/", data={"file": (img, "test_image.jpg")})
            self.assertEqual(response.status_code, 200)
            self.assertIn(b"Prediction", response.data)

if __name__ == "__main__":
    unittest.main()
