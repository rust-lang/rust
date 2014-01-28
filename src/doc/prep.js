#!/usr/local/bin/node

/***
 * Pandoc-style markdown preprocessor that drops extra directives
 * included for running doc code, and that optionally, when
 * --highlight is provided, replaces code blocks that are Rust code
 * with highlighted HTML blocks. The directives recognized are:
 *
 * '## ignore' tells the test extractor (extract-tests.js) to ignore
 *   the block completely.
 * '## notrust' makes the test extractor ignore the block, makes
 *   this script not highlight the block.
 * '# [any text]' is a line that is stripped out by this script, and
 *   converted to a normal line of code (without the leading #) by
 *   the test extractor.
 */

var fs = require("fs");
CodeMirror = require("./lib/codemirror-node");
require("./lib/codemirror-rust");

function help() {
  console.log("usage: " + process.argv[0] + " [--highlight] [-o outfile] [infile]");
  process.exit(1);
}

var highlight = false, infile, outfile;

for (var i = 2; i < process.argv.length; ++i) {
  var arg = process.argv[i];
  if (arg == "--highlight") highlight = true;
  else if (arg == "-o" && outfile == null && ++i < process.argv.length) outfile = process.argv[i];
  else if (arg[0] != "-") infile = arg;
  else help();
}

var lines = fs.readFileSync(infile || "/dev/stdin").toString().split(/\n\r?/g), cur = 0, line;
var out = outfile ? fs.createWriteStream(outfile) : process.stdout;

while ((line = lines[cur++]) != null) {
  if (/^~~~/.test(line)) {
    var block = "", bline;
    var notRust =
          /notrust/.test(line)
          // These are all used by the language ref to indicate things
          // that are not Rust source code
          || /ebnf/.test(line)
          || /abnf/.test(line)
          || /keyword/.test(line)
          || /field/.test(line)
          || /precedence/.test(line);
    var isRust = !notRust;
    while ((bline = lines[cur++]) != null) {
      if (/^~~~/.test(bline)) break;
      if (!/^\s*##? /.test(bline)) block += bline + "\n";
    }
    if (!highlight || !isRust)
      out.write(line + "\n" + block + bline + "\n");
    else {
      var html = '<pre class="cm-s-default">', curstr = "", curstyle = null;
      function add(str, style) {
        if (style != curstyle) {
          if (curstyle) html +=
            '<span class="cm-' + CodeMirror.htmlEscape(curstyle) + '">' +
            CodeMirror.htmlEscape(curstr) + "</span>";
          else if (curstr) html += CodeMirror.htmlEscape(curstr);
          curstr = str; curstyle = style;
        } else curstr += str;
      }
      CodeMirror.runMode(block, "rust", add);
      add("", "bogus"); // Flush pending string.
      out.write(html + "</pre>\n");
    }
  } else {
    out.write(line + "\n");
  }
}
