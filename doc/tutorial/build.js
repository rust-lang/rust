var fs = require("fs"), md = require("./lib/markdown");
CodeMirror = require("./lib/codemirror-node");
require("./lib/codemirror-rust");

md.Markdown.dialects.Maruku.block.code = function code(block, next) {
  if (block.match(/^    /)) {
    var text = block.replace(/(^|\n)    /g, "$1"), accum = [], curstr = "", curstyle = null;
    function add(str, style) {
      if (style != curstyle) {
        if (curstyle) accum.push(["span", {"class": "cm-" + curstyle}, curstr]);
        else if (curstr) accum.push(curstr);
        curstr = str; curstyle = style;
      } else curstr += str;
    }
    CodeMirror.runMode(text, "rust", add);
    add("", "bogus"); // Flush pending string.
    return [["pre", {"class": "cm-s-default"}].concat(accum)];
  }
};    

function markdown(str) { return md.toHTML(str, "Maruku"); }

function fileDates(file, c) {
  function takeTime(str) {
    return Number(str.match(/^(\S+)\s/)[1]) * 1000;
  }
  require("child_process").exec("git rev-list --timestamp HEAD -- " + file, function(err, stdout) {
    if (err != null) { console.log("Failed to run git rev-list"); return; }
    var history = stdout.split("\n");
    if (history.length && history[history.length-1] == "") history.pop();
    var created = history.length ? takeTime(history[0]) : Date.now();
    var modified = created;
    if (history.length > 1) modified = takeTime(history[history.length-1]);
    c(created, modified);
  });
}

function head(title) {
  return "<html><head><link rel='stylesheet' href='style.css' type='text/css'>" +
    "<link rel='stylesheet' href='default.css' type='text/css'>" +
    "<meta http-equiv='Content-Type' content='text/html; charset=utf-8'><title>" +
    title + "</title></head><body>\n";
}

function foot(created, modified) {
  var r = "<p class='head'>"
  var crStr = formatTime(created), modStr = formatTime(modified);
  if (created) r += "Created " + crStr;
  if (crStr != modStr)
    r += (created ? ", l" : "L") + "ast modified on " + modStr;
  return r + "</p>";
}

function formatTime(tm) {
  var d = new Date(tm);
  var months = ["", "January", "February", "March", "April", "May", "June", "July", "August",
                "September", "October", "November", "December"];
  return months[d.getMonth()] + " " + d.getDate() + ", " + d.getFullYear();
}

var files = fs.readFileSync("order", "utf8").split("\n").filter(function(x) { return x; });
var max_modified = 0;
var sections = [];

// Querying git for modified dates has to be done async in node it seems...
var queried = 0;
for (var i = 0; i < files.length; ++i)
  (function(i) { // Make lexical i stable
    fileDates(files[i], function(ctime, mtime) {
      sections[i] = {
        text: fs.readFileSync(files[i] + ".md", "utf8"),
        ctime: ctime, mtime: mtime,
        name: files[i],
      };
      max_modified = Math.max(mtime, max_modified);
      if (++queried == files.length) buildTutorial();
    });
  })(i);

function htmlName(i) { return sections[i].name + ".html"; }

function buildTutorial() {
  var index = head("Rust language tutorial") + "<div id='content'>" +
    markdown(fs.readFileSync("index.md", "utf8")) + "<ol>";
  for (var i = 0; i < sections.length; ++i) {
    var s = sections[i];
    var html = htmlName(i);
    var title = s.text.match(/^# (.*)\n/)[1];
    index += '<li><a href="' + html + '">' + title + "</a></li>";
    
    var nav = '<p class="head">Section ' + (i + 1) + ' of the Rust language tutorial.<br>';
    if (i > 0) nav += '<a href="' + htmlName(i-1) + '">« Section ' + i + "</a> | ";
    nav += '<a href="index.html">Index</a>';
    if (i + 1 < sections.length) nav += ' | <a href="' + htmlName(i+1) + '">Section ' + (i + 2) + " »</a>";
    nav += "</p>";
    fs.writeFileSync("web/" + html, head(title) + nav + '<div id="content">' + markdown(s.text) + "</div>" +
                     nav + foot(s.ctime, s.mtime) + "</body></html>");
  }
  index += "</ol></div>" + foot(null, max_modified) + "</body></html>";
  fs.writeFileSync("web/index.html", index);
}
