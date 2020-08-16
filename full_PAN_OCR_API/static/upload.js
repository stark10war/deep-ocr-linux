



$(document).ready(function() {
   $('#retrieve').click(function(){
       var form_data = new FormData($('#upload-file')[0]);
       $.ajax({
        type: 'POST',
        url: '/image',
        beforeSend: function(){
                $('.ajax-loader').css("visibility", "visible");
                $('#results_img').css("visibility", "hidden");
                $('#results_ocr').css("visibility", "hidden");

        },
        data: form_data,
        contentType: false,
        cache: false,
        processData: false,
        success: function(response) {
        console.log(response)
        page_count = response.meta.page_count
        doc_count = response.meta.doc_count
        response_copy = JSON.parse(JSON.stringify(response))
        delete response_copy["out_images"];
        var jsonPretty = JSON.stringify(response_copy,null,2);
        document.getElementById("doc_info").innerHTML = "<pre><code>{" + jsonPretty +"}</code></pre>"
        if (response.out_images.img.length > 0){
           $("#myimg").attr("src", "data:image/png;base64," + response.out_images.img);
           }
        if (response.out_images.img.length > 0){
           $("#myimg1").attr("src", "data:image/png;base64," + response.out_images.img1);
           }
        if (response.out_images.img.length > 0){
           $("#myimg2").attr("src", "data:image/png;base64," + response.out_images.img2);
           }

        if (response.data.page_1.doc_1.doc_type == "pan_landmark"){
           var ocr_results = response.data.page_1.doc_1.ocr_results
           document.getElementById("ocrText").innerHTML = ""
           document.getElementById("ocrText").innerHTML =  "NAME: " + ocr_results.name.text + "<br/>"+
                                                            "DOB: " + ocr_results.dob.text +"<br/>"+
                                                            "FATHER'S NAME: " + ocr_results.father_name.text + "<br/>"+
                                                            "PAN: " + ocr_results.pan.text ;
        }

        if (response.data.page_1.doc_1.doc_type == "aadhar_landmark"){
           document.getElementById("ocrText").innerHTML = ""
           var ocr_results = response.data.page_1.doc_1.ocr_results
           document.getElementById("ocrText").innerHTML = ocr_results
        }


      },
      complete: function(){
          $('.ajax-loader').css("visibility", "hidden");
          $('#results_img').css("visibility", "visible");
          $('#results_ocr').css("visibility", "visible");

      },

      error: function() {
          $('.ajax-loader').css("visibility", "hidden");
          $('#results_img').css("visibility", "visible");
          $('#results_ocr').css("visibility", "visible");

        //Do Something to handle error
     }
     });
   });
});
