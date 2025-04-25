## Realtime Lie Detector using Gemini Flash

This is a python program that takes you screen input and inputs each frame into gemini flash in order to get real-time responses on whether the person in frame is lying or not based purely on visual cues.

`agent.py` was my attempt at teaching an agent to only have understanding around identifying people that are lying through visual cues. Though due to limited time this proved to be quite difficult due to requiring a good data pipeline.

Alternatively in `face_reader.py` I wanted to see how exactly I could capture the frames from the screen and see how well it would work when inputting it to gemini.

Below we see a screenshot of the model responding to what's on screen with it's response based on it's facial features.

![image](https://github.com/user-attachments/assets/cefc5675-d63f-41c4-ae9b-92c17343e564)

you can run this yourself by simply running `python3 face_reader.py` while also being able to see the exact face that is being analyzed by including the `--debug flag`

![image](https://github.com/user-attachments/assets/a539a599-e522-403e-98ea-b2384841f9e1)

### Using Gemini

Since this utilizes the Gemini Flash Model you'll need to have a GCP account with the gemini api enabled. You'll provide this in a `.env` file.
