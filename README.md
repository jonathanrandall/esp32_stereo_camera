# esp32_stereo_camera
 stereo camera files for esp32

Youtube video at:
https://youtu.be/CAVYHlFGpaw 


The code implements the stereo camera for object detection, labelling and distance estimation with the esp32.

The notebooks are:

1. stereo_image_v6.ipynb: this is for calibrating the camera.
2. esp32_stereo_cam.ipynb: this connects to the cameras and does the inference.

The sketch in esp32_webcam can be used to run the camera (although not properly tested) if you can't get the "CameraWebServer" sketch to compile. Note, the CameraWebServer sketch might not compile in newer esp32 board managers (?=2.x.x).

If you want to use any stream handler from another sketch with python, you need to do the following:

Find the following code in the stream handler:

  ```
  if(res == ESP_OK){  
        size_t hlen = snprintf((char *)part_buf, 64, _STREAM_PART, _jpg_buf_len);      
        res = httpd_resp_send_chunk(req, (const char *)part_buf, hlen);      
    }
    if(res == ESP_OK){
        res = httpd_resp_send_chunk(req, (const char *)_jpg_buf, _jpg_buf_len);
    }
    if(res == ESP_OK){
        res = httpd_resp_send_chunk(req, _STREAM_BOUNDARY, strlen(_STREAM_BOUNDARY));
    }
    ```
    
And copy the bottom statement (_STREAM_BOUNDAEY) to the top:
  ```
  ```
   if(res == ESP_OK){
        res = httpd_resp_send_chunk(req, _STREAM_BOUNDARY, strlen(_STREAM_BOUNDARY));
    }
   if(res == ESP_OK){
        size_t hlen = snprintf((char *)part_buf, 64, _STREAM_PART, _jpg_buf_len);
        res = httpd_resp_send_chunk(req, (const char *)part_buf, hlen);
    }
    if(res == ESP_OK){
        res = httpd_resp_send_chunk(req, (const char *)_jpg_buf, _jpg_buf_len);
    }
  ```    
    

