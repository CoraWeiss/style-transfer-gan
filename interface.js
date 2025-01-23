<!DOCTYPE html>
<html>
<head>
  <title>GAN Image Viewer</title>
  <script src="https://unpkg.com/react@17/umd/react.development.js"></script>
  <script src="https://unpkg.com/react-dom@17/umd/react-dom.development.js"></script>
  <script src="https://unpkg.com/babel-standalone@6/babel.min.js"></script>
  <script src="interface.js" type="text/babel"></script>
  <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
</head>
<body>
  <div id="root"></div>
  <script type="text/babel">
    ReactDOM.render(<GanViewer />, document.getElementById('root'));
  </script>
</body>
</html>
