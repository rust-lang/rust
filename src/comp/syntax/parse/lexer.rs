
import core::{vec, str, option, either};
import std::io;
import io::reader_util;
import option::{some, none};
import util::interner;
import util::interner::intern;
import codemap;
import driver::diagnostic;

type reader = @{
    cm: codemap::codemap,
    span_diagnostic: diagnostic::span_handler,
    src: @str,
    len: uint,
    mutable col: uint,
    mutable pos: uint,
    mutable curr: char,
    mutable chpos: uint,
    mutable strs: [str],
    filemap: codemap::filemap,
    interner: @interner::interner<str>
};

impl reader for reader {
    fn is_eof() -> bool { self.curr == -1 as char }
    fn get_str_from(start: uint) -> str unsafe {
        // I'm pretty skeptical about this subtraction. What if there's a
        // multi-byte character before the mark?
        ret str::unsafe::slice_bytes(*self.src, start - 1u, self.pos - 1u);
    }
    fn next() -> char {
        if self.pos < self.len {
            ret str::char_at(*self.src, self.pos);
        } else { ret -1 as char; }
    }
    fn bump() {
        if self.pos < self.len {
            self.col += 1u;
            self.chpos += 1u;
            if self.curr == '\n' {
                codemap::next_line(self.filemap, self.chpos, self.pos +
                                   self.filemap.start_pos.byte);
                self.col = 0u;
            }
            let next = str::char_range_at(*self.src, self.pos);
            self.pos = next.next;
            self.curr = next.ch;
        } else { self.curr = -1 as char; }
    }
    fn fatal(m: str) -> ! {
        self.span_diagnostic.span_fatal(
            ast_util::mk_sp(self.chpos, self.chpos),
            m)
    }
}

fn new_reader(cm: codemap::codemap,
              span_diagnostic: diagnostic::span_handler,
              filemap: codemap::filemap,
              itr: @interner::interner<str>) -> reader {
    let r = @{cm: cm,
              span_diagnostic: span_diagnostic,
              src: filemap.src, len: str::byte_len(*filemap.src),
              mutable col: 0u, mutable pos: 0u, mutable curr: -1 as char,
              mutable chpos: filemap.start_pos.ch, mutable strs: [],
              filemap: filemap, interner: itr};
    if r.pos < r.len {
        let next = str::char_range_at(*r.src, r.pos);
        r.pos = next.next;
        r.curr = next.ch;
    }
    ret r;
}

fn dec_digit_val(c: char) -> int { ret (c as int) - ('0' as int); }

fn hex_digit_val(c: char) -> int {
    if in_range(c, '0', '9') { ret (c as int) - ('0' as int); }
    if in_range(c, 'a', 'f') { ret (c as int) - ('a' as int) + 10; }
    if in_range(c, 'A', 'F') { ret (c as int) - ('A' as int) + 10; }
    fail;
}

fn bin_digit_value(c: char) -> int { if c == '0' { ret 0; } ret 1; }

fn is_whitespace(c: char) -> bool {
    ret c == ' ' || c == '\t' || c == '\r' || c == '\n';
}

fn may_begin_ident(c: char) -> bool { ret is_alpha(c) || c == '_'; }

fn in_range(c: char, lo: char, hi: char) -> bool { ret lo <= c && c <= hi; }

fn is_alpha(c: char) -> bool {
    ret in_range(c, 'a', 'z') || in_range(c, 'A', 'Z');
}

fn is_dec_digit(c: char) -> bool { ret in_range(c, '0', '9'); }

fn is_alnum(c: char) -> bool { ret is_alpha(c) || is_dec_digit(c); }

fn is_hex_digit(c: char) -> bool {
    ret in_range(c, '0', '9') || in_range(c, 'a', 'f') ||
            in_range(c, 'A', 'F');
}

fn is_bin_digit(c: char) -> bool { ret c == '0' || c == '1'; }

fn consume_whitespace_and_comments(rdr: reader) {
    while is_whitespace(rdr.curr) { rdr.bump(); }
    be consume_any_line_comment(rdr);
}

fn consume_any_line_comment(rdr: reader) {
    if rdr.curr == '/' {
        alt rdr.next() {
          '/' {
            while rdr.curr != '\n' && !rdr.is_eof() { rdr.bump(); }
            // Restart whitespace munch.

            be consume_whitespace_and_comments(rdr);
          }
          '*' { rdr.bump(); rdr.bump(); be consume_block_comment(rdr); }
          _ { ret; }
        }
    }
}

fn consume_block_comment(rdr: reader) {
    let level: int = 1;
    while level > 0 {
        if rdr.is_eof() { rdr.fatal("unterminated block comment"); }
        if rdr.curr == '/' && rdr.next() == '*' {
            rdr.bump();
            rdr.bump();
            level += 1;
        } else {
            if rdr.curr == '*' && rdr.next() == '/' {
                rdr.bump();
                rdr.bump();
                level -= 1;
            } else { rdr.bump(); }
        }
    }
    // restart whitespace munch.

    be consume_whitespace_and_comments(rdr);
}

fn scan_exponent(rdr: reader) -> option<str> {
    let c = rdr.curr;
    let rslt = "";
    if c == 'e' || c == 'E' {
        str::push_byte(rslt, c as u8);
        rdr.bump();
        c = rdr.curr;
        if c == '-' || c == '+' {
            str::push_byte(rslt, c as u8);
            rdr.bump();
        }
        let exponent = scan_digits(rdr, 10u);
        if str::byte_len(exponent) > 0u {
            ret some(rslt + exponent);
        } else { rdr.fatal("scan_exponent: bad fp literal"); }
    } else { ret none::<str>; }
}

fn scan_digits(rdr: reader, radix: uint) -> str {
    let rslt = "";
    while true {
        let c = rdr.curr;
        if c == '_' { rdr.bump(); cont; }
        alt char::maybe_digit(c) {
          some(d) if (d as uint) < radix {
            str::push_byte(rslt, c as u8);
            rdr.bump();
          }
          _ { break; }
        }
    }
    ret rslt;
}

fn scan_number(c: char, rdr: reader) -> token::token {
    let num_str, base = 10u, c = c, n = rdr.next();
    if c == '0' && n == 'x' {
        rdr.bump();
        rdr.bump();
        base = 16u;
    } else if c == '0' && n == 'b' {
        rdr.bump();
        rdr.bump();
        base = 2u;
    }
    num_str = scan_digits(rdr, base);
    c = rdr.curr;
    n = rdr.next();
    if c == 'u' || c == 'i' {
        let signed = c == 'i', tp = if signed { either::left(ast::ty_i) }
                                         else { either::right(ast::ty_u) };
        rdr.bump();
        c = rdr.curr;
        if c == '8' {
            rdr.bump();
            tp = if signed { either::left(ast::ty_i8) }
                      else { either::right(ast::ty_u8) };
        }
        n = rdr.next();
        if c == '1' && n == '6' {
            rdr.bump();
            rdr.bump();
            tp = if signed { either::left(ast::ty_i16) }
                      else { either::right(ast::ty_u16) };
        } else if c == '3' && n == '2' {
            rdr.bump();
            rdr.bump();
            tp = if signed { either::left(ast::ty_i32) }
                      else { either::right(ast::ty_u32) };
        } else if c == '6' && n == '4' {
            rdr.bump();
            rdr.bump();
            tp = if signed { either::left(ast::ty_i64) }
                      else { either::right(ast::ty_u64) };
        }
        let parsed = u64::from_str(num_str, base as u64);
        alt tp {
          either::left(t) { ret token::LIT_INT(parsed as i64, t); }
          either::right(t) { ret token::LIT_UINT(parsed, t); }
        }
    }
    let is_float = false;
    if rdr.curr == '.' && !(is_alpha(rdr.next()) || rdr.next() == '_') {
        is_float = true;
        rdr.bump();
        let dec_part = scan_digits(rdr, 10u);
        num_str += "." + dec_part;
    }
    alt scan_exponent(rdr) {
      some(s) {
        is_float = true;
        num_str += s;
      }
      none {}
    }
    if rdr.curr == 'f' {
        rdr.bump();
        c = rdr.curr;
        n = rdr.next();
        if c == '3' && n == '2' {
            rdr.bump();
            rdr.bump();
            ret token::LIT_FLOAT(intern(*rdr.interner, num_str),
                                 ast::ty_f32);
        } else if c == '6' && n == '4' {
            rdr.bump();
            rdr.bump();
            ret token::LIT_FLOAT(intern(*rdr.interner, num_str),
                                 ast::ty_f64);
            /* FIXME: if this is out of range for either a 32-bit or
            64-bit float, it won't be noticed till the back-end */
        } else {
            is_float = true;
        }
    }
    if is_float {
        ret token::LIT_FLOAT(interner::intern(*rdr.interner, num_str),
                             ast::ty_f);
    } else {
        let parsed = u64::from_str(num_str, base as u64);
        ret token::LIT_INT(parsed as i64, ast::ty_i);
    }
}

fn scan_numeric_escape(rdr: reader, n_hex_digits: uint) -> char {
    let accum_int = 0, i = n_hex_digits;
    while i != 0u {
        let n = rdr.curr;
        rdr.bump();
        if !is_hex_digit(n) {
            rdr.fatal(#fmt["illegal numeric character escape: %d", n as int]);
        }
        accum_int *= 16;
        accum_int += hex_digit_val(n);
        i -= 1u;
    }
    ret accum_int as char;
}

fn next_token(rdr: reader) -> {tok: token::token, chpos: uint, bpos: uint} {
    consume_whitespace_and_comments(rdr);
    let start_chpos = rdr.chpos;
    let start_bpos = rdr.pos;
    let tok = if rdr.is_eof() { token::EOF } else { next_token_inner(rdr) };
    ret {tok: tok, chpos: start_chpos, bpos: start_bpos};
}

fn next_token_inner(rdr: reader) -> token::token {
    let accum_str = "";
    let c = rdr.curr;
    if char::is_XID_start(c) || c == '_' {
        while char::is_XID_continue(c) {
            str::push_char(accum_str, c);
            rdr.bump();
            c = rdr.curr;
        }
        if str::eq(accum_str, "_") { ret token::UNDERSCORE; }
        let is_mod_name = c == ':' && rdr.next() == ':';

        // FIXME: perform NFKC normalization here.
        ret token::IDENT(interner::intern::<str>(*rdr.interner,
                                                 accum_str), is_mod_name);
    }
    if is_dec_digit(c) {
        ret scan_number(c, rdr);
    }
    fn binop(rdr: reader, op: token::binop) -> token::token {
        rdr.bump();
        if rdr.curr == '=' {
            rdr.bump();
            ret token::BINOPEQ(op);
        } else { ret token::BINOP(op); }
    }
    alt c {





      // One-byte tokens.
      ';' { rdr.bump(); ret token::SEMI; }
      ',' { rdr.bump(); ret token::COMMA; }
      '.' {
        rdr.bump();
        if rdr.curr == '.' && rdr.next() == '.' {
            rdr.bump();
            rdr.bump();
            ret token::ELLIPSIS;
        }
        ret token::DOT;
      }
      '(' { rdr.bump(); ret token::LPAREN; }
      ')' { rdr.bump(); ret token::RPAREN; }
      '{' { rdr.bump(); ret token::LBRACE; }
      '}' { rdr.bump(); ret token::RBRACE; }
      '[' { rdr.bump(); ret token::LBRACKET; }
      ']' { rdr.bump(); ret token::RBRACKET; }
      '@' { rdr.bump(); ret token::AT; }
      '#' {
        rdr.bump();
        if rdr.curr == '<' { rdr.bump(); ret token::POUND_LT; }
        if rdr.curr == '{' { rdr.bump(); ret token::POUND_LBRACE; }
        ret token::POUND;
      }
      '~' { rdr.bump(); ret token::TILDE; }
      ':' {
        rdr.bump();
        if rdr.curr == ':' {
            rdr.bump();
            ret token::MOD_SEP;
        } else { ret token::COLON; }
      }





      // Multi-byte tokens.
      '=' {
        rdr.bump();
        if rdr.curr == '=' {
            rdr.bump();
            ret token::EQEQ;
        } else { ret token::EQ; }
      }
      '!' {
        rdr.bump();
        if rdr.curr == '=' {
            rdr.bump();
            ret token::NE;
        } else { ret token::NOT; }
      }
      '<' {
        rdr.bump();
        alt rdr.curr {
          '=' { rdr.bump(); ret token::LE; }
          '<' { ret binop(rdr, token::LSL); }
          '-' {
            rdr.bump();
            alt rdr.curr {
              '>' { rdr.bump(); ret token::DARROW; }
              _ { ret token::LARROW; }
            }
          }
          _ { ret token::LT; }
        }
      }
      '>' {
        rdr.bump();
        alt rdr.curr {
          '=' { rdr.bump(); ret token::GE; }
          '>' {
            if rdr.next() == '>' {
                rdr.bump();
                ret binop(rdr, token::ASR);
            } else { ret binop(rdr, token::LSR); }
          }
          _ { ret token::GT; }
        }
      }
      '\'' {
        rdr.bump();
        let c2 = rdr.curr;
        rdr.bump();
        if c2 == '\\' {
            let escaped = rdr.curr;
            rdr.bump();
            alt escaped {
              'n' { c2 = '\n'; }
              'r' { c2 = '\r'; }
              't' { c2 = '\t'; }
              '\\' { c2 = '\\'; }
              '\'' { c2 = '\''; }
              'x' { c2 = scan_numeric_escape(rdr, 2u); }
              'u' { c2 = scan_numeric_escape(rdr, 4u); }
              'U' { c2 = scan_numeric_escape(rdr, 8u); }
              c2 {
                rdr.fatal(#fmt["unknown character escape: %d", c2 as int]);
              }
            }
        }
        if rdr.curr != '\'' {
            rdr.fatal("unterminated character constant");
        }
        rdr.bump(); // advance curr past token
        ret token::LIT_INT(c2 as i64, ast::ty_char);
      }
      '"' {
        let n = rdr.chpos;
        rdr.bump();
        while rdr.curr != '"' {
            if rdr.is_eof() {
                rdr.fatal(#fmt["unterminated double quote string: %s",
                             rdr.get_str_from(n)]);
            }

            let ch = rdr.curr;
            rdr.bump();
            alt ch {
              '\\' {
                let escaped = rdr.curr;
                rdr.bump();
                alt escaped {
                  'n' { str::push_byte(accum_str, '\n' as u8); }
                  'r' { str::push_byte(accum_str, '\r' as u8); }
                  't' { str::push_byte(accum_str, '\t' as u8); }
                  '\\' { str::push_byte(accum_str, '\\' as u8); }
                  '"' { str::push_byte(accum_str, '"' as u8); }
                  '\n' { consume_whitespace(rdr); }
                  'x' {
                    str::push_char(accum_str, scan_numeric_escape(rdr, 2u));
                  }
                  'u' {
                    str::push_char(accum_str, scan_numeric_escape(rdr, 4u));
                  }
                  'U' {
                    str::push_char(accum_str, scan_numeric_escape(rdr, 8u));
                  }
                  c2 {
                    rdr.fatal(#fmt["unknown string escape: %d", c2 as int]);
                  }
                }
              }
              _ { str::push_char(accum_str, ch); }
            }
        }
        rdr.bump();
        ret token::LIT_STR(interner::intern::<str>(*rdr.interner,
                                                   accum_str));
      }
      '-' {
        if rdr.next() == '>' {
            rdr.bump();
            rdr.bump();
            ret token::RARROW;
        } else { ret binop(rdr, token::MINUS); }
      }
      '&' {
        if rdr.next() == '&' {
            rdr.bump();
            rdr.bump();
            ret token::ANDAND;
        } else { ret binop(rdr, token::AND); }
      }
      '|' {
        alt rdr.next() {
          '|' { rdr.bump(); rdr.bump(); ret token::OROR; }
          _ { ret binop(rdr, token::OR); }
        }
      }
      '+' { ret binop(rdr, token::PLUS); }
      '*' { ret binop(rdr, token::STAR); }
      '/' { ret binop(rdr, token::SLASH); }
      '^' { ret binop(rdr, token::CARET); }
      '%' { ret binop(rdr, token::PERCENT); }
      c { rdr.fatal(#fmt["unkown start of token: %d", c as int]); }
    }
}

enum cmnt_style {
    isolated, // No code on either side of each line of the comment
    trailing, // Code exists to the left of the comment
    mixed, // Code before /* foo */ and after the comment
    blank_line, // Just a manual blank line "\n\n", for layout
}

type cmnt = {style: cmnt_style, lines: [str], pos: uint};

fn read_to_eol(rdr: reader) -> str {
    let val = "";
    while rdr.curr != '\n' && !rdr.is_eof() {
        str::push_char(val, rdr.curr);
        rdr.bump();
    }
    if rdr.curr == '\n' { rdr.bump(); }
    ret val;
}

fn read_one_line_comment(rdr: reader) -> str {
    let val = read_to_eol(rdr);
    assert (val[0] == '/' as u8 && val[1] == '/' as u8);
    ret val;
}

fn consume_whitespace(rdr: reader) {
    while is_whitespace(rdr.curr) && !rdr.is_eof() { rdr.bump(); }
}

fn consume_non_eol_whitespace(rdr: reader) {
    while is_whitespace(rdr.curr) && rdr.curr != '\n' && !rdr.is_eof() {
        rdr.bump();
    }
}

fn push_blank_line_comment(rdr: reader, &comments: [cmnt]) {
    #debug(">>> blank-line comment");
    let v: [str] = [];
    comments += [{style: blank_line, lines: v, pos: rdr.chpos}];
}

fn consume_whitespace_counting_blank_lines(rdr: reader, &comments: [cmnt]) {
    while is_whitespace(rdr.curr) && !rdr.is_eof() {
        if rdr.col == 0u && rdr.curr == '\n' {
            push_blank_line_comment(rdr, comments);
        }
        rdr.bump();
    }
}

fn read_line_comments(rdr: reader, code_to_the_left: bool) -> cmnt {
    #debug(">>> line comments");
    let p = rdr.chpos;
    let lines: [str] = [];
    while rdr.curr == '/' && rdr.next() == '/' {
        let line = read_one_line_comment(rdr);
        log(debug, line);
        lines += [line];
        consume_non_eol_whitespace(rdr);
    }
    #debug("<<< line comments");
    ret {style: if code_to_the_left { trailing } else { isolated },
         lines: lines,
         pos: p};
}

fn all_whitespace(s: str, begin: uint, end: uint) -> bool {
    let i: uint = begin;
    while i != end { if !is_whitespace(s[i] as char) { ret false; } i += 1u; }
    ret true;
}

fn trim_whitespace_prefix_and_push_line(&lines: [str],
                                        s: str, col: uint) unsafe {
    let s1;
    if all_whitespace(s, 0u, col) {
        if col < str::byte_len(s) {
            s1 = str::unsafe::slice_bytes(s, col, str::byte_len(s));
        } else { s1 = ""; }
    } else { s1 = s; }
    log(debug, "pushing line: " + s1);
    lines += [s1];
}

fn read_block_comment(rdr: reader, code_to_the_left: bool) -> cmnt {
    #debug(">>> block comment");
    let p = rdr.chpos;
    let lines: [str] = [];
    let col: uint = rdr.col;
    rdr.bump();
    rdr.bump();
    let curr_line = "/*";
    let level: int = 1;
    while level > 0 {
        #debug("=== block comment level %d", level);
        if rdr.is_eof() { rdr.fatal("unterminated block comment"); }
        if rdr.curr == '\n' {
            trim_whitespace_prefix_and_push_line(lines, curr_line, col);
            curr_line = "";
            rdr.bump();
        } else {
            str::push_char(curr_line, rdr.curr);
            if rdr.curr == '/' && rdr.next() == '*' {
                rdr.bump();
                rdr.bump();
                curr_line += "*";
                level += 1;
            } else {
                if rdr.curr == '*' && rdr.next() == '/' {
                    rdr.bump();
                    rdr.bump();
                    curr_line += "/";
                    level -= 1;
                } else { rdr.bump(); }
            }
        }
    }
    if str::byte_len(curr_line) != 0u {
        trim_whitespace_prefix_and_push_line(lines, curr_line, col);
    }
    let style = if code_to_the_left { trailing } else { isolated };
    consume_non_eol_whitespace(rdr);
    if !rdr.is_eof() && rdr.curr != '\n' && vec::len(lines) == 1u {
        style = mixed;
    }
    #debug("<<< block comment");
    ret {style: style, lines: lines, pos: p};
}

fn peeking_at_comment(rdr: reader) -> bool {
    ret rdr.curr == '/' && rdr.next() == '/' ||
            rdr.curr == '/' && rdr.next() == '*';
}

fn consume_comment(rdr: reader, code_to_the_left: bool, &comments: [cmnt]) {
    #debug(">>> consume comment");
    if rdr.curr == '/' && rdr.next() == '/' {
        comments += [read_line_comments(rdr, code_to_the_left)];
    } else if rdr.curr == '/' && rdr.next() == '*' {
        comments += [read_block_comment(rdr, code_to_the_left)];
    } else { fail; }
    #debug("<<< consume comment");
}

fn is_lit(t: token::token) -> bool {
    ret alt t {
          token::LIT_INT(_, _) { true }
          token::LIT_UINT(_, _) { true }
          token::LIT_FLOAT(_, _) { true }
          token::LIT_STR(_) { true }
          token::LIT_BOOL(_) { true }
          _ { false }
        }
}

type lit = {lit: str, pos: uint};

fn gather_comments_and_literals(cm: codemap::codemap,
                                span_diagnostic: diagnostic::span_handler,
                                path: str,
                                srdr: io::reader) ->
   {cmnts: [cmnt], lits: [lit]} {
    let src = @str::from_bytes(srdr.read_whole_stream());
    let itr = @interner::mk::<str>(str::hash, str::eq);
    let rdr = new_reader(cm, span_diagnostic,
                         codemap::new_filemap(path, src, 0u, 0u), itr);
    let comments: [cmnt] = [];
    let literals: [lit] = [];
    let first_read: bool = true;
    while !rdr.is_eof() {
        while true {
            let code_to_the_left = !first_read;
            consume_non_eol_whitespace(rdr);
            if rdr.curr == '\n' {
                code_to_the_left = false;
                consume_whitespace_counting_blank_lines(rdr, comments);
            }
            while peeking_at_comment(rdr) {
                consume_comment(rdr, code_to_the_left, comments);
                consume_whitespace_counting_blank_lines(rdr, comments);
            }
            break;
        }
        let tok = next_token(rdr);
        if is_lit(tok.tok) {
            literals += [{lit: rdr.get_str_from(tok.bpos), pos: tok.chpos}];
        }
        log(debug, "tok: " + token::to_str(rdr, tok.tok));
        first_read = false;
    }
    ret {cmnts: comments, lits: literals};
}

//
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
//
