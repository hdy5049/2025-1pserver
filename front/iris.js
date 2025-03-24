google.charts.load('current', {'packages':['corechart']});

function drawChart(prob, pid){
  var data = google.visualization.arrayToDataTable([
    ['iris species', 'prob'],
    ['setosa',     prob[0][0]],
    ['versicolor',  prob[0][1]],
    ['virginica', prob[0][2]]
  ]);

  var options = {
    title: '붓꽃 품종 확률',
    pieHole: 0.4,
  };

  var chart = new google.visualization.PieChart(pid);
  chart.draw(data, options);
}

function Send(){

sl = document.getElementById("sl")
sw = document.getElementById("sw")
pl = document.getElementById("pl")
pw = document.getElementById("pw")

  var data = {
    'sepal_length': sl.value,
    'sepal_width': sw.value,
    'petal_length': pl.value,
    'petal_width': pw.value,
  }

  $.ajax({
    type: "POST",
    url: 'http://localhost:8000/predict',
    headers:{
        "Accept" : "application/json",
        "Content-Type": "application/json",
        },
    data: JSON.stringify(data),

  }).done(function(response) {

        txtOut.value = response.prediction + "  " + response.probability
        // Set a callback to run when the Google Visualization API is loaded.
        google.charts.setOnLoadCallback(drawChart(response.probability, document.getElementById("donutchart")));
        console.log(response)
  }).fail(function(error) {
    alert("!/js/user.js에서 에러발생: " + error.statusText);
    console.log(error)
  }).always(function(r){
    console.log("always" + r)
  });
}