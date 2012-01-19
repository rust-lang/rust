#!/usr/local/bin/node

/***
 * Script for extracting compilable fragments from markdown
 * documentation. See prep.js for a description of the format
 * recognized by this tool. Expects a directory fragements/ to exist
 * under the current directory, and writes the fragments in there as
 * individual .rs files.
 */

var fs = require("fs");

if (!process.argv[2]) {
  console.log("Please provide an input file name.");
  process.exit(1);
}

var lines = fs.readFileSync(process.argv[2]).toString().split(/\n\r?/g);
var cur = 0, line, chapter, chapter_n;

while ((line = lines[cur++]) != null) {
  var chap = line.match(/^# (.*)/);
  if (chap) {
    chapter = chap[1].toLowerCase().replace(/\W/g, "_");
    chapter_n = 1;
  } else if (/^~~~/.test(line)) {
    var block = "", ignore = false;
    while ((line = lines[cur++]) != null) {
      if (/^\s*## (?:notrust|ignore)/.test(line)) ignore = true;
      else if (/^~~~/.test(line)) break;
      else block += line.replace(/^# /, "") + "\n";
    }
    if (!ignore) {
      if (!/\bfn main\b/.test(block)) {
        if (/(^|\n) *(native|use|mod|import|export)\b/.test(block))
          block += "\nfn main() {}\n";
        else block = "fn main() {\n" + block + "\n}\n";
      }
      if (!/\buse std\b/.test(block)) block = "use std;\n" + block;
      var filename = "fragments/" + chapter + "_" + (chapter_n++) + ".rs";
      fs.writeFileSync(filename, block);
    }
  }
}
