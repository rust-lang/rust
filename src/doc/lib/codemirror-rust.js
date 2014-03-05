// Copyright (C) 2013 by Marijn Haverbeke <marijnh@gmail.com> and others
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

CodeMirror.defineMode("rust", function() {
  var indentUnit = 4, altIndentUnit = 2;
  var valKeywords = {
    "if": "if-style", "while": "if-style", "loop": "if-style", "else": "else-style",
    "do": "else-style", "return": "else-style",
    "break": "atom", "cont": "atom", "const": "let", "resource": "fn",
    "let": "let", "fn": "fn", "for": "for", "match": "match", "trait": "trait",
    "impl": "impl", "type": "type", "enum": "enum", "struct": "atom", "mod": "mod",
    "as": "op", "true": "atom", "false": "atom", "assert": "op", "check": "op",
    "claim": "op", "extern": "ignore", "unsafe": "ignore", "import": "else-style",
    "export": "else-style", "copy": "op", "log": "op",
    "use": "op", "self": "atom", "pub": "atom", "priv": "atom"
  };
  var typeKeywords = function() {
    var keywords = {"fn": "fn"};
    var atoms = "bool uint int i8 i16 i32 i64 u8 u16 u32 u64 float f32 f64 str char".split(" ");
    for (var i = 0, e = atoms.length; i < e; ++i) keywords[atoms[i]] = "atom";
    return keywords;
  }();
  var operatorChar = /[+\-*&%=<>!?|\.@]/;

  // Tokenizer

  // Used as scratch variable to communicate multiple values without
  // consing up tons of objects.
  var tcat, content;
  function r(tc, style) {
    tcat = tc;
    return style;
  }

  function tokenBase(stream, state) {
    var ch = stream.next();
    if (ch == '"') {
      state.tokenize = tokenString;
      return state.tokenize(stream, state);
    }
    if (ch == "'") {
      tcat = "atom";
      if (stream.eat("\\")) {
        if (stream.skipTo("'")) { stream.next(); return "string"; }
        else { return "error"; }
      } else {
        stream.next();
        return stream.eat("'") ? "string" : "error";
      }
    }
    if (ch == "/") {
      if (stream.eat("/")) { stream.skipToEnd(); return "comment"; }
      if (stream.eat("*")) {
        state.tokenize = tokenComment(1);
        return state.tokenize(stream, state);
      }
    }
    if (ch == "#") {
      if (stream.eat("[")) { tcat = "open-attr"; return null; }
      stream.eatWhile(/\w/);
      return r("macro", "meta");
    }
    if (ch == ":" && stream.match(":<")) {
      return r("op", null);
    }
    if (ch.match(/\d/) || (ch == "." && stream.eat(/\d/))) {
      var flp = false;
      if (!stream.match(/^x[\da-f]+/i) && !stream.match(/^b[01]+/)) {
        stream.eatWhile(/\d/);
        if (stream.eat(".")) { flp = true; stream.eatWhile(/\d/); }
        if (stream.match(/^e[+\-]?\d+/i)) { flp = true; }
      }
      if (flp) stream.match(/^f(?:32|64)/);
      else stream.match(/^[ui](?:8|16|32|64)/);
      return r("atom", "number");
    }
    if (ch.match(/[()\[\]{}:;,]/)) return r(ch, null);
    if (ch == "-" && stream.eat(">")) return r("->", null);
    if (ch.match(operatorChar)) {
      stream.eatWhile(operatorChar);
      return r("op", null);
    }
    stream.eatWhile(/\w/);
    content = stream.current();
    if (stream.match(/^::\w/)) {
      stream.backUp(1);
      return r("prefix", "variable-2");
    }
    if (state.keywords.propertyIsEnumerable(content))
      return r(state.keywords[content], content.match(/true|false/) ? "atom" : "keyword");
    return r("name", "variable");
  }

  function tokenString(stream, state) {
    var ch, escaped = false;
    while (ch = stream.next()) {
      if (ch == '"' && !escaped) {
        state.tokenize = tokenBase;
        return r("atom", "string");
      }
      escaped = !escaped && ch == "\\";
    }
    // Hack to not confuse the parser when a string is split in
    // pieces.
    return r("op", "string");
  }

  function tokenComment(depth) {
    return function(stream, state) {
      var lastCh = null, ch;
      while (ch = stream.next()) {
        if (ch == "/" && lastCh == "*") {
          if (depth == 1) {
            state.tokenize = tokenBase;
            break;
          } else {
            state.tokenize = tokenComment(depth - 1);
            return state.tokenize(stream, state);
          }
        }
        if (ch == "*" && lastCh == "/") {
          state.tokenize = tokenComment(depth + 1);
          return state.tokenize(stream, state);
        }
        lastCh = ch;
      }
      return "comment";
    };
  }

  // Parser

  var cx = {state: null, stream: null, marked: null, cc: null};
  function pass() {
    for (var i = arguments.length - 1; i >= 0; i--) cx.cc.push(arguments[i]);
  }
  function cont() {
    pass.apply(null, arguments);
    return true;
  }

  function pushlex(type, info) {
    var result = function() {
      var state = cx.state;
      state.lexical = {indented: state.indented, column: cx.stream.column(),
                       type: type, prev: state.lexical, info: info};
    };
    result.lex = true;
    return result;
  }
  function poplex() {
    var state = cx.state;
    if (state.lexical.prev) {
      if (state.lexical.type == ")")
        state.indented = state.lexical.indented;
      state.lexical = state.lexical.prev;
    }
  }
  function typecx() { cx.state.keywords = typeKeywords; }
  function valcx() { cx.state.keywords = valKeywords; }
  poplex.lex = typecx.lex = valcx.lex = true;

  function commasep(comb, end) {
    function more(type) {
      if (type == ",") return cont(comb, more);
      if (type == end) return cont();
      return cont(more);
    }
    return function(type) {
      if (type == end) return cont();
      return pass(comb, more);
    };
  }

  function stat_of(comb, tag) {
    return cont(pushlex("stat", tag), comb, poplex, block);
  }
  function block(type) {
    if (type == "}") return cont();
    if (type == "let") return stat_of(letdef1, "let");
    if (type == "fn") return stat_of(fndef);
    if (type == "type") return cont(pushlex("stat"), tydef, endstatement, poplex, block);
    if (type == "enum") return stat_of(tagdef);
    if (type == "mod") return stat_of(mod);
    if (type == "trait") return stat_of(trait);
    if (type == "impl") return stat_of(impl);
    if (type == "open-attr") return cont(pushlex("]"), commasep(expression, "]"), poplex);
    if (type == "ignore" || type.match(/[\]\);,]/)) return cont(block);
    return pass(pushlex("stat"), expression, poplex, endstatement, block);
  }
  function endstatement(type) {
    if (type == ";") return cont();
    return pass();
  }
  function expression(type) {
    if (type == "atom" || type == "name") return cont(maybeop);
    if (type == "{") return cont(pushlex("}"), exprbrace, poplex);
    if (type.match(/[\[\(]/)) return matchBrackets(type, expression);
    if (type.match(/[\]\)\};,]/)) return pass();
    if (type == "if-style") return cont(expression, expression);
    if (type == "else-style" || type == "op") return cont(expression);
    if (type == "for") return cont(pattern, maybetype, inop, expression, expression);
    if (type == "match") return cont(expression, altbody);
    if (type == "fn") return cont(fndef);
    if (type == "macro") return cont(macro);
    return cont();
  }
  function maybeop(type) {
    if (content == ".") return cont(maybeprop);
    if (content == "::<"){return cont(typarams, maybeop);}
    if (type == "op" || content == ":") return cont(expression);
    if (type == "(" || type == "[") return matchBrackets(type, expression);
    return pass();
  }
  function maybeprop(type) {
    if (content.match(/^\w+$/)) {cx.marked = "variable"; return cont(maybeop);}
    return pass(expression);
  }
  function exprbrace(type) {
    if (type == "op") {
      if (content == "|") return cont(blockvars, poplex, pushlex("}", "block"), block);
      if (content == "||") return cont(poplex, pushlex("}", "block"), block);
    }
    if (content == "mut" || (content.match(/^\w+$/) && cx.stream.peek() == ":"
                                 && !cx.stream.match("::", false)))
      return pass(record_of(expression));
    return pass(block);
  }
  function record_of(comb) {
    function ro(type) {
      if (content == "mut" || content == "with") {cx.marked = "keyword"; return cont(ro);}
      if (content.match(/^\w*$/)) {cx.marked = "variable"; return cont(ro);}
      if (type == ":") return cont(comb, ro);
      if (type == "}") return cont();
      return cont(ro);
    }
    return ro;
  }
  function blockvars(type) {
    if (type == "name") {cx.marked = "def"; return cont(blockvars);}
    if (type == "op" && content == "|") return cont();
    return cont(blockvars);
  }

  function letdef1(type) {
    if (type.match(/[\]\)\};]/)) return cont();
    if (content == "=") return cont(expression, letdef2);
    if (type == ",") return cont(letdef1);
    return pass(pattern, maybetype, letdef1);
  }
  function letdef2(type) {
    if (type.match(/[\]\)\};,]/)) return pass(letdef1);
    else return pass(expression, letdef2);
  }
  function maybetype(type) {
    if (type == ":") return cont(typecx, rtype, valcx);
    return pass();
  }
  function inop(type) {
    if (type == "name" && content == "in") {cx.marked = "keyword"; return cont();}
    return pass();
  }
  function fndef(type) {
    if (content == "@" || content == "~") {cx.marked = "keyword"; return cont(fndef);}
    if (type == "name") {cx.marked = "def"; return cont(fndef);}
    if (content == "<") return cont(typarams, fndef);
    if (type == "{") return pass(expression);
    if (type == "(") return cont(pushlex(")"), commasep(argdef, ")"), poplex, fndef);
    if (type == "->") return cont(typecx, rtype, valcx, fndef);
    if (type == ";") return cont();
    return cont(fndef);
  }
  function tydef(type) {
    if (type == "name") {cx.marked = "def"; return cont(tydef);}
    if (content == "<") return cont(typarams, tydef);
    if (content == "=") return cont(typecx, rtype, valcx);
    return cont(tydef);
  }
  function tagdef(type) {
    if (type == "name") {cx.marked = "def"; return cont(tagdef);}
    if (content == "<") return cont(typarams, tagdef);
    if (content == "=") return cont(typecx, rtype, valcx, endstatement);
    if (type == "{") return cont(pushlex("}"), typecx, tagblock, valcx, poplex);
    return cont(tagdef);
  }
  function tagblock(type) {
    if (type == "}") return cont();
    if (type == "(") return cont(pushlex(")"), commasep(rtype, ")"), poplex, tagblock);
    if (content.match(/^\w+$/)) cx.marked = "def";
    return cont(tagblock);
  }
  function mod(type) {
    if (type == "name") {cx.marked = "def"; return cont(mod);}
    if (type == "{") return cont(pushlex("}"), block, poplex);
    return pass();
  }
  function trait(type) {
    if (type == "name") {cx.marked = "def"; return cont(trait);}
    if (content == "<") return cont(typarams, trait);
    if (type == "{") return cont(pushlex("}"), block, poplex);
    return pass();
  }
  function impl(type) {
    if (content == "<") return cont(typarams, impl);
    if (content == "of" || content == "for") {cx.marked = "keyword"; return cont(rtype, impl);}
    if (type == "name") {cx.marked = "def"; return cont(impl);}
    if (type == "{") return cont(pushlex("}"), block, poplex);
    return pass();
  }
  function typarams(type) {
    if (content == ">") return cont();
    if (content == ",") return cont(typarams);
    if (content == ":") return cont(rtype, typarams);
    return pass(rtype, typarams);
  }
  function argdef(type) {
    if (type == "name") {cx.marked = "def"; return cont(argdef);}
    if (type == ":") return cont(typecx, rtype, valcx);
    return pass();
  }
  function rtype(type) {
    if (type == "name") {cx.marked = "variable-3"; return cont(rtypemaybeparam); }
    if (content == "mut") {cx.marked = "keyword"; return cont(rtype);}
    if (type == "atom") return cont(rtypemaybeparam);
    if (type == "op" || type == "obj") return cont(rtype);
    if (type == "fn") return cont(fntype);
    if (type == "{") return cont(pushlex("{"), record_of(rtype), poplex);
    return matchBrackets(type, rtype);
  }
  function rtypemaybeparam(type) {
    if (content == "<") return cont(typarams);
    return pass();
  }
  function fntype(type) {
    if (type == "(") return cont(pushlex("("), commasep(rtype, ")"), poplex, fntype);
    if (type == "->") return cont(rtype);
    return pass();
  }
  function pattern(type) {
    if (type == "name") {cx.marked = "def"; return cont(patternmaybeop);}
    if (type == "atom") return cont(patternmaybeop);
    if (type == "op") return cont(pattern);
    if (type.match(/[\]\)\};,]/)) return pass();
    return matchBrackets(type, pattern);
  }
  function patternmaybeop(type) {
    if (type == "op" && content == ".") return cont();
    if (content == "to") {cx.marked = "keyword"; return cont(pattern);}
    else return pass();
  }
  function altbody(type) {
    if (type == "{") return cont(pushlex("}", "match"), altblock1, poplex);
    return pass();
  }
  function altblock1(type) {
    if (type == "}") return cont();
    if (type == "|") return cont(altblock1);
    if (content == "when") {cx.marked = "keyword"; return cont(expression, altblock2);}
    if (type.match(/[\]\);,]/)) return cont(altblock1);
    return pass(pattern, altblock2);
  }
  function altblock2(type) {
    if (type == "{") return cont(pushlex("}", "match"), block, poplex, altblock1);
    else return pass(altblock1);
  }

  function macro(type) {
    if (type.match(/[\[\(\{]/)) return matchBrackets(type, expression);
    return pass();
  }
  function matchBrackets(type, comb) {
    if (type == "[") return cont(pushlex("]"), commasep(comb, "]"), poplex);
    if (type == "(") return cont(pushlex(")"), commasep(comb, ")"), poplex);
    if (type == "{") return cont(pushlex("}"), commasep(comb, "}"), poplex);
    return cont();
  }

  function parse(state, stream, style) {
    var cc = state.cc;
    // Communicate our context to the combinators.
    // (Less wasteful than consing up a hundred closures on every call.)
    cx.state = state; cx.stream = stream; cx.marked = null, cx.cc = cc;

    while (true) {
      var combinator = cc.length ? cc.pop() : block;
      if (combinator(tcat)) {
        while(cc.length && cc[cc.length - 1].lex)
          cc.pop()();
        return cx.marked || style;
      }
    }
  }

  return {
    startState: function() {
      return {
        tokenize: tokenBase,
        cc: [],
        lexical: {indented: -indentUnit, column: 0, type: "top", align: false},
        keywords: valKeywords,
        indented: 0
      };
    },

    token: function(stream, state) {
      if (stream.sol()) {
        if (!state.lexical.hasOwnProperty("align"))
          state.lexical.align = false;
        state.indented = stream.indentation();
      }
      if (stream.eatSpace()) return null;
      tcat = content = null;
      var style = state.tokenize(stream, state);
      if (style == "comment") return style;
      if (!state.lexical.hasOwnProperty("align"))
        state.lexical.align = true;
      if (tcat == "prefix") return style;
      if (!content) content = stream.current();
      return parse(state, stream, style);
    },

    indent: function(state, textAfter) {
      if (state.tokenize != tokenBase) return 0;
      var firstChar = textAfter && textAfter.charAt(0), lexical = state.lexical,
          type = lexical.type, closing = firstChar == type;
      if (type == "stat") return lexical.indented + indentUnit;
      if (lexical.align) return lexical.column + (closing ? 0 : 1);
      return lexical.indented +
        (closing ? 0 : (lexical.info == "match" ? altIndentUnit : indentUnit));
    },

    electricChars: "{}"
  };
});

CodeMirror.defineMIME("text/x-rustsrc", "rust");
