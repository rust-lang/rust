var fs = require("fs"), md = require("./lib/markdown");

// Runs markdown.js over the tutorial, to find the code blocks in it.
// Uses the #-markers in those code blocks, along with some vague
// heuristics, to turn them into compilable files. Outputs these files
// to fragments/.
//
// '##ignore' means don't test this block
// '##notrust' means the block isn't rust code
//     (used by build.js to not highlight it)
// '# code' means insert the given code to complete the fragment
//     (build.js strips out such lines)

var curFile, curFrag;
md.Markdown.dialects.Maruku.block.code = function code(block, next) {
  if (block.match(/^    /)) {
    var ignore, text = String(block);
    while (next.length && next[0].match(/^    /)) text += "\n" + String(next.shift());
    text = text.split("\n").map(function(line) {
      line = line.slice(4);
      if (line == "## ignore" || line == "## notrust") { ignore = true; line = ""; }
      if (/^# /.test(line)) line = line.slice(2);
      return line;
    }).join("\n");
    if (ignore) return;
    if (!/\bfn main\b/.test(text)) {
      if (/(^|\n) *(native|use|mod|import|export)\b/.test(text))
        text += "\nfn main() {}\n";
      else text = "fn main() {\n" + text + "\n}\n";
    }
    if (!/\buse std\b/.test(text)) text = "use std;\n" + text;
    fs.writeFileSync("fragments/" + curFile + "_" + (++curFrag) + ".rs", text);
  }
};

fs.readFileSync("order", "utf8").split("\n").filter(id).forEach(handle);

function id(x) { return x; }
function handle(file) {
  curFile = file; curFrag = 0;
  md.parse(fs.readFileSync(file + ".md", "utf8"), "Maruku");
}
