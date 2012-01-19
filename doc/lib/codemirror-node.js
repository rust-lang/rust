exports.htmlEscape = function(text) {
  var replacements = {"<": "&lt;", ">": "&gt;",
                      "&": "&amp;", "\"": "&quot;"};
  return text.replace(/[<>&"]/g, function(character) {
    return replacements[character];
  });
};

exports.splitLines = function(string){return string.split(/\r?\n/);};

// Counts the column offset in a string, taking tabs into account.
// Used mostly to find indentation.
function countColumn(string, end) {
  if (end == null) {
    end = string.search(/[^\s\u00a0]/);
    if (end == -1) end = string.length;
  }
  for (var i = 0, n = 0; i < end; ++i) {
    if (string.charAt(i) == "\t") n += tabSize - (n % tabSize);
    else ++n;
  }
  return n;
}

function StringStream(string) {
  this.pos = this.start = 0;
  this.string = string;
}
StringStream.prototype = {
  eol: function() {return this.pos >= this.string.length;},
  sol: function() {return this.pos == 0;},
  peek: function() {return this.string.charAt(this.pos);},
  next: function() {
    if (this.pos < this.string.length)
      return this.string.charAt(this.pos++);
  },
  eat: function(match) {
    var ch = this.string.charAt(this.pos);
    if (typeof match == "string") var ok = ch == match;
    else var ok = ch && (match.test ? match.test(ch) : match(ch));
    if (ok) {++this.pos; return ch;}
  },
  eatWhile: function(match) {
    var start = this.pos;
    while (this.eat(match)){}
    return this.pos > start;
  },
  eatSpace: function() {
    var start = this.pos;
    while (/[\s\u00a0]/.test(this.string.charAt(this.pos))) ++this.pos;
    return this.pos > start;
  },
  skipToEnd: function() {this.pos = this.string.length;},
  skipTo: function(ch) {
    var found = this.string.indexOf(ch, this.pos);
    if (found > -1) {this.pos = found; return true;}
  },
  backUp: function(n) {this.pos -= n;},
  column: function() {return countColumn(this.string, this.start);},
  indentation: function() {return countColumn(this.string);},
  match: function(pattern, consume, caseInsensitive) {
    if (typeof pattern == "string") {
      function cased(str) {return caseInsensitive ? str.toLowerCase() : str;}
      if (cased(this.string).indexOf(cased(pattern), this.pos) == this.pos) {
        if (consume !== false) this.pos += pattern.length;
        return true;
      }
    }
    else {
      var match = this.string.slice(this.pos).match(pattern);
      if (match && consume !== false) this.pos += match[0].length;
      return match;
    }
  },
  current: function(){return this.string.slice(this.start, this.pos);}
};
exports.StringStream = StringStream;

exports.startState = function(mode, a1, a2) {
  return mode.startState ? mode.startState(a1, a2) : true;
};

var modes = {}, mimeModes = {};
exports.defineMode = function(name, mode) { modes[name] = mode; };
exports.defineMIME = function(mime, spec) { mimeModes[mime] = spec; };
exports.getMode = function(options, spec) {
  if (typeof spec == "string" && mimeModes.hasOwnProperty(spec))
    spec = mimeModes[spec];
  if (typeof spec == "string")
    var mname = spec, config = {};
  else if (spec != null)
    var mname = spec.name, config = spec;
  var mfactory = modes[mname];
  if (!mfactory) throw new Error("Unknown mode: " + spec);
  return mfactory(options, config || {});
};

exports.runMode = function(string, modespec, callback) {
  var mode = exports.getMode({indentUnit: 2}, modespec);
  var isNode = callback.nodeType == 1;
  if (isNode) {
    var node = callback, accum = [];
    callback = function(string, style) {
      if (string == "\n")
        accum.push("<br>");
      else if (style)
        accum.push("<span class=\"cm-" + exports.htmlEscape(style) + "\">" + exports.htmlEscape(string) + "</span>");
      else
        accum.push(exports.htmlEscape(string));
    }
  }
  var lines = exports.splitLines(string), state = exports.startState(mode);
  for (var i = 0, e = lines.length; i < e; ++i) {
    if (i) callback("\n");
    var stream = new exports.StringStream(lines[i]);
    while (!stream.eol()) {
      var style = mode.token(stream, state);
      callback(stream.current(), style, i, stream.start);
      stream.start = stream.pos;
    }
  }
  if (isNode)
    node.innerHTML = accum.join("");
};
