import time
import edgeiq

def main():
    obj_detect = edgeiq.ObjectDetection("alwaysai/mobilenet_ssd")
    obj_detect.load(engine=edgeiq.Engine.DNN)

    fps = edgeiq.FPS()

    with edgeiq.WebcamVideoStream(cam=0) as video_stream, \
                edgeiq.Streamer() as streamer:
        time.sleep(2.0)
        fps.start()

        while True:
            frame = video_stream.read()
            results = obj_detect.detect_objects(frame, confidence_level=.5)
            frame = edgeiq.markup_image(
                    frame, results.predictions, colors=obj_detect.colors)

            text = ["Model: {}".format(obj_detect.model_id)]
            text.append(
                    "Inference time: {:1.3f} s".format(results.duration))
            text.append("Objects:")

            for prediction in results.predictions:
                text.append("{}: {:2.2f}%".format(
                    prediction.label, prediction.confidence * 100))

            streamer.send_data(frame, text)

            fps.update()

            if streamer.check_exit():
                break

        fps.stop()
        print("elapsed time: {:.2f}".format(fps.get_elapsed_seconds()))
        print("approx. FPS: {:.2f}".format(fps.compute_fps()))

if __name__ == "__main__":
    main()
