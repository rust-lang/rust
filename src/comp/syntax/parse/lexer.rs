
import std::{io, int, vec, str, map, option};
import std::map::hashmap;
import std::option::{some, none};
import util::interner;
import util::interner::intern;
import codemap;

type reader =
    obj {
        fn is_eof() -> bool;
        fn curr() -> char;
        fn next() -> char;
        fn init();
        fn bump();
        fn get_str_from(uint) -> str;
        fn get_interner() -> @interner::interner<str>;
        fn get_chpos() -> uint;
        fn get_byte_pos() -> uint;
        fn get_col() -> uint;
        fn get_filemap() -> codemap::filemap;
        fn err(str);
    };

fn new_reader(cm: codemap::codemap, src: str, filemap: codemap::filemap,
              itr: @interner::interner<str>) -> reader {
    obj reader(cm: codemap::codemap,
               src: str,
               len: uint,
               mutable col: uint,
               mutable pos: uint,
               mutable ch: char,
               mutable chpos: uint,
               mutable strs: [str],
               fm: codemap::filemap,
               itr: @interner::interner<str>) {
        fn is_eof() -> bool { ret ch == -1 as char; }
        fn get_str_from(start: uint) -> str {
            // I'm pretty skeptical about this subtraction. What if there's a
            // multi-byte character before the mark?
            ret str::slice(src, start - 1u, pos - 1u);
        }
        fn get_chpos() -> uint { ret chpos; }
        fn get_byte_pos() -> uint { ret pos; }
        fn curr() -> char { ret ch; }
        fn next() -> char {
            if pos < len {
                ret str::char_at(src, pos);
            } else { ret -1 as char; }
        }
        fn init() {
            if pos < len {
                let next = str::char_range_at(src, pos);
                pos = next.next;
                ch = next.ch;
            }
        }
        fn bump() {
            if pos < len {
                col += 1u;
                chpos += 1u;
                if ch == '\n' {
                    codemap::next_line(fm, chpos, pos + fm.start_pos.byte);
                    col = 0u;
                }
                let next = str::char_range_at(src, pos);
                pos = next.next;
                ch = next.ch;
            } else { ch = -1 as char; }
        }
        fn get_interner() -> @interner::interner<str> { ret itr; }
        fn get_col() -> uint { ret col; }
        fn get_filemap() -> codemap::filemap { ret fm; }
        fn err(m: str) {
            codemap::emit_error(some(ast_util::mk_sp(chpos, chpos)), m, cm);
        }
    }
    let strs: [str] = [];
    let rd =
        reader(cm, src, str::byte_len(src), 0u, 0u, -1 as char,
               filemap.start_pos.ch, strs, filemap, itr);
    rd.init();
    ret rd;
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
    while is_whitespace(rdr.curr()) { rdr.bump(); }
    be consume_any_line_comment(rdr);
}

fn consume_any_line_comment(rdr: reader) {
    if rdr.curr() == '/' {
        alt rdr.next() {
          '/' {
            while rdr.curr() != '\n' && !rdr.is_eof() { rdr.bump(); }
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
        if rdr.is_eof() { rdr.err("unterminated block comment"); fail; }
        if rdr.curr() == '/' && rdr.next() == '*' {
            rdr.bump();
            rdr.bump();
            level += 1;
        } else {
            if rdr.curr() == '*' && rdr.next() == '/' {
                rdr.bump();
                rdr.bump();
                level -= 1;
            } else { rdr.bump(); }
        }
    }
    // restart whitespace munch.

    be consume_whitespace_and_comments(rdr);
}

fn digits_to_string(s: str) -> int {
    let accum_int: int = 0;
    for c: u8 in s { accum_int *= 10; accum_int += dec_digit_val(c as char); }
    ret accum_int;
}

fn scan_exponent(rdr: reader) -> option::t<str> {
    let c = rdr.curr();
    let rslt = "";
    if c == 'e' || c == 'E' {
        rslt += str::unsafe_from_bytes([c as u8]);
        rdr.bump();
        c = rdr.curr();
        if c == '-' || c == '+' {
            rslt += str::unsafe_from_bytes([c as u8]);
            rdr.bump();
        }
        let exponent = scan_dec_digits(rdr);
        if str::byte_len(exponent) > 0u {
            ret some(rslt + exponent);
        } else { rdr.err("scan_exponent: bad fp literal"); fail; }
    } else { ret none::<str>; }
}

fn scan_dec_digits(rdr: reader) -> str {
    let c = rdr.curr();
    let rslt: str = "";
    while is_dec_digit(c) || c == '_' {
        if c != '_' { rslt += str::unsafe_from_bytes([c as u8]); }
        rdr.bump();
        c = rdr.curr();
    }
    ret rslt;
}

fn scan_number(c: char, rdr: reader) -> token::token {
    let accum_int = 0;
    let num_str: str = "";
    let is_dec_integer: bool = false;
    let n = rdr.next();
    if c == '0' && n == 'x' {
        rdr.bump();
        rdr.bump();
        c = rdr.curr();
        while is_hex_digit(c) || c == '_' {
            if c != '_' { accum_int *= 16; accum_int += hex_digit_val(c); }
            rdr.bump();
            c = rdr.curr();
        }
    } else if c == '0' && n == 'b' {
        rdr.bump();
        rdr.bump();
        c = rdr.curr();
        while is_bin_digit(c) || c == '_' {
            if c != '_' { accum_int *= 2; accum_int += bin_digit_value(c); }
            rdr.bump();
            c = rdr.curr();
        }
    } else { num_str = scan_dec_digits(rdr); is_dec_integer = true; }
    if is_dec_integer { accum_int = digits_to_string(num_str); }
    c = rdr.curr();
    n = rdr.next();
    if c == 'u' || c == 'i' {
        let signed: bool = c == 'i';
        rdr.bump();
        c = rdr.curr();
        if c == '8' {
            rdr.bump();
            if signed {
                ret token::LIT_MACH_INT(ast::ty_i8, accum_int);
            } else { ret token::LIT_MACH_INT(ast::ty_u8, accum_int); }
        }
        n = rdr.next();
        if c == '1' && n == '6' {
            rdr.bump();
            rdr.bump();
            if signed {
                ret token::LIT_MACH_INT(ast::ty_i16, accum_int);
            } else { ret token::LIT_MACH_INT(ast::ty_u16, accum_int); }
        }
        if c == '3' && n == '2' {
            rdr.bump();
            rdr.bump();
            if signed {
                ret token::LIT_MACH_INT(ast::ty_i32, accum_int);
            } else { ret token::LIT_MACH_INT(ast::ty_u32, accum_int); }
        }
        if c == '6' && n == '4' {
            rdr.bump();
            rdr.bump();
            if signed {
                ret token::LIT_MACH_INT(ast::ty_i64, accum_int);
            } else { ret token::LIT_MACH_INT(ast::ty_u64, accum_int); }
        }
        if signed {
            ret token::LIT_INT(accum_int);
        } else {
            // FIXME: should cast in the target bit-width.
            ret token::LIT_UINT(accum_int as uint);
        }
    }
    let is_float = false;
    if rdr.curr() == '.' {
        is_float = true;
        rdr.bump();
        let dec_part = scan_dec_digits(rdr);
        num_str += "." + dec_part;
    }
    alt scan_exponent(rdr) {
      some(s) {
        is_float = true;
        num_str += s;
      }
      none. {}
    }
    if rdr.curr() == 'f' {
        rdr.bump();
        c = rdr.curr();
        n = rdr.next();
        if c == '3' && n == '2' {
            rdr.bump();
            rdr.bump();
            ret token::LIT_MACH_FLOAT(ast::ty_f32,
                                      intern(*rdr.get_interner(),
                                             num_str));
        } else if c == '6' && n == '4' {
            rdr.bump();
            rdr.bump();
            ret token::LIT_MACH_FLOAT(ast::ty_f64,
                                      intern(*rdr.get_interner(),
                                             num_str));
            /* FIXME: if this is out of range for either a 32-bit or
            64-bit float, it won't be noticed till the back-end */
        } else {
            is_float = true;
        }
    }
    if is_float {
        ret token::LIT_FLOAT(interner::intern::<str>(*rdr.get_interner(),
                                                     num_str));
    } else {
        ret token::LIT_INT(accum_int);
    }
}

fn scan_numeric_escape(rdr: reader, n_hex_digits: uint) -> char {
    let accum_int = 0;
    while n_hex_digits != 0u {
        let n = rdr.curr();
        rdr.bump();
        if !is_hex_digit(n) {
            rdr.err(#fmt["illegal numeric character escape: %d", n as int]);
            fail;
        }
        accum_int *= 16;
        accum_int += hex_digit_val(n);
        n_hex_digits -= 1u;
    }
    ret accum_int as char;
}

fn next_token(rdr: reader) -> {tok: token::token, chpos: uint, bpos: uint} {
    consume_whitespace_and_comments(rdr);
    let start_chpos = rdr.get_chpos();
    let start_bpos = rdr.get_byte_pos();
    let tok = if rdr.is_eof() { token::EOF } else { next_token_inner(rdr) };
    ret {tok: tok, chpos: start_chpos, bpos: start_bpos};
}

fn next_token_inner(rdr: reader) -> token::token {
    let accum_str = "";
    let c = rdr.curr();
    if is_alpha(c) || c == '_' {
        while is_alnum(c) || c == '_' {
            str::push_char(accum_str, c);
            rdr.bump();
            c = rdr.curr();
        }
        if str::eq(accum_str, "_") { ret token::UNDERSCORE; }
        let is_mod_name = c == ':' && rdr.next() == ':';
        ret token::IDENT(interner::intern::<str>(*rdr.get_interner(),
                                                 accum_str), is_mod_name);
    }
    if is_dec_digit(c) { ret scan_number(c, rdr); }
    fn binop(rdr: reader, op: token::binop) -> token::token {
        rdr.bump();
        if rdr.curr() == '=' {
            rdr.bump();
            ret token::BINOPEQ(op);
        } else { ret token::BINOP(op); }
    }
    alt c {





      // One-byte tokens.
      '?' {
        rdr.bump();
        ret token::QUES;
      }
      ';' { rdr.bump(); ret token::SEMI; }
      ',' { rdr.bump(); ret token::COMMA; }
      '.' {
        rdr.bump();
        if rdr.curr() == '.' && rdr.next() == '.' {
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
        if rdr.curr() == '<' { rdr.bump(); ret token::POUND_LT; }
        if rdr.curr() == '{' { rdr.bump(); ret token::POUND_LBRACE; }
        ret token::POUND;
      }
      '~' { rdr.bump(); ret token::TILDE; }
      ':' {
        rdr.bump();
        if rdr.curr() == ':' {
            rdr.bump();
            ret token::MOD_SEP;
        } else { ret token::COLON; }
      }





      // Multi-byte tokens.
      '=' {
        rdr.bump();
        if rdr.curr() == '=' {
            rdr.bump();
            ret token::EQEQ;
        } else { ret token::EQ; }
      }
      '!' {
        rdr.bump();
        if rdr.curr() == '=' {
            rdr.bump();
            ret token::NE;
        } else { ret token::NOT; }
      }
      '<' {
        rdr.bump();
        alt rdr.curr() {
          '=' { rdr.bump(); ret token::LE; }
          '<' { ret binop(rdr, token::LSL); }
          '-' {
            rdr.bump();
            alt rdr.curr() {
              '>' { rdr.bump(); ret token::DARROW; }
              _ { ret token::LARROW; }
            }
          }
          _ { ret token::LT; }
        }
      }
      '>' {
        rdr.bump();
        alt rdr.curr() {
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
        let c2 = rdr.curr();
        rdr.bump();
        if c2 == '\\' {
            let escaped = rdr.curr();
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
                rdr.err(#fmt["unknown character escape: %d", c2 as int]);
                fail;
              }
            }
        }
        if rdr.curr() != '\'' {
            rdr.err("unterminated character constant");
            fail;
        }
        rdr.bump(); // advance curr past token

        ret token::LIT_CHAR(c2);
      }
      '"' {
        rdr.bump();
        while rdr.curr() != '"' {
            let ch = rdr.curr();
            rdr.bump();
            alt ch {
              '\\' {
                let escaped = rdr.curr();
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
                    rdr.err(#fmt["unknown string escape: %d", c2 as int]);
                    fail;
                  }
                }
              }
              _ { str::push_char(accum_str, ch); }
            }
        }
        rdr.bump();
        ret token::LIT_STR(interner::intern::<str>(*rdr.get_interner(),
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
      c { rdr.err(#fmt["unkown start of token: %d", c as int]); fail; }
    }
}

tag cmnt_style {
    isolated; // No code on either side of each line of the comment
    trailing; // Code exists to the left of the comment
    mixed; // Code before /* foo */ and after the comment
    blank_line; // Just a manual blank line "\n\n", for layout
}

type cmnt = {style: cmnt_style, lines: [str], pos: uint};

fn read_to_eol(rdr: reader) -> str {
    let val = "";
    while rdr.curr() != '\n' && !rdr.is_eof() {
        str::push_char(val, rdr.curr());
        rdr.bump();
    }
    if rdr.curr() == '\n' { rdr.bump(); }
    ret val;
}

fn read_one_line_comment(rdr: reader) -> str {
    let val = read_to_eol(rdr);
    assert (val[0] == '/' as u8 && val[1] == '/' as u8);
    ret val;
}

fn consume_whitespace(rdr: reader) {
    while is_whitespace(rdr.curr()) && !rdr.is_eof() { rdr.bump(); }
}

fn consume_non_eol_whitespace(rdr: reader) {
    while is_whitespace(rdr.curr()) && rdr.curr() != '\n' && !rdr.is_eof() {
        rdr.bump();
    }
}

fn push_blank_line_comment(rdr: reader, &comments: [cmnt]) {
    log ">>> blank-line comment";
    let v: [str] = [];
    comments += [{style: blank_line, lines: v, pos: rdr.get_chpos()}];
}

fn consume_whitespace_counting_blank_lines(rdr: reader, &comments: [cmnt]) {
    while is_whitespace(rdr.curr()) && !rdr.is_eof() {
        if rdr.get_col() == 0u && rdr.curr() == '\n' {
            push_blank_line_comment(rdr, comments);
        }
        rdr.bump();
    }
}

fn read_line_comments(rdr: reader, code_to_the_left: bool) -> cmnt {
    log ">>> line comments";
    let p = rdr.get_chpos();
    let lines: [str] = [];
    while rdr.curr() == '/' && rdr.next() == '/' {
        let line = read_one_line_comment(rdr);
        log line;
        lines += [line];
        consume_non_eol_whitespace(rdr);
    }
    log "<<< line comments";
    ret {style: if code_to_the_left { trailing } else { isolated },
         lines: lines,
         pos: p};
}

fn all_whitespace(s: str, begin: uint, end: uint) -> bool {
    let i: uint = begin;
    while i != end { if !is_whitespace(s[i] as char) { ret false; } i += 1u; }
    ret true;
}

fn trim_whitespace_prefix_and_push_line(&lines: [str], s: str, col: uint) {
    let s1;
    if all_whitespace(s, 0u, col) {
        if col < str::byte_len(s) {
            s1 = str::slice(s, col, str::byte_len(s));
        } else { s1 = ""; }
    } else { s1 = s; }
    log "pushing line: " + s1;
    lines += [s1];
}

fn read_block_comment(rdr: reader, code_to_the_left: bool) -> cmnt {
    log ">>> block comment";
    let p = rdr.get_chpos();
    let lines: [str] = [];
    let col: uint = rdr.get_col();
    rdr.bump();
    rdr.bump();
    let curr_line = "/*";
    let level: int = 1;
    while level > 0 {
        log #fmt["=== block comment level %d", level];
        if rdr.is_eof() { rdr.err("unterminated block comment"); fail; }
        if rdr.curr() == '\n' {
            trim_whitespace_prefix_and_push_line(lines, curr_line, col);
            curr_line = "";
            rdr.bump();
        } else {
            str::push_char(curr_line, rdr.curr());
            if rdr.curr() == '/' && rdr.next() == '*' {
                rdr.bump();
                rdr.bump();
                curr_line += "*";
                level += 1;
            } else {
                if rdr.curr() == '*' && rdr.next() == '/' {
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
    if !rdr.is_eof() && rdr.curr() != '\n' && vec::len(lines) == 1u {
        style = mixed;
    }
    log "<<< block comment";
    ret {style: style, lines: lines, pos: p};
}

fn peeking_at_comment(rdr: reader) -> bool {
    ret rdr.curr() == '/' && rdr.next() == '/' ||
            rdr.curr() == '/' && rdr.next() == '*';
}

fn consume_comment(rdr: reader, code_to_the_left: bool, &comments: [cmnt]) {
    log ">>> consume comment";
    if rdr.curr() == '/' && rdr.next() == '/' {
        comments += [read_line_comments(rdr, code_to_the_left)];
    } else if rdr.curr() == '/' && rdr.next() == '*' {
        comments += [read_block_comment(rdr, code_to_the_left)];
    } else { fail; }
    log "<<< consume comment";
}

fn is_lit(t: token::token) -> bool {
    ret alt t {
          token::LIT_INT(_) { true }
          token::LIT_UINT(_) { true }
          token::LIT_MACH_INT(_, _) { true }
          token::LIT_FLOAT(_) { true }
          token::LIT_MACH_FLOAT(_, _) { true }
          token::LIT_STR(_) { true }
          token::LIT_CHAR(_) { true }
          token::LIT_BOOL(_) { true }
          _ { false }
        }
}

type lit = {lit: str, pos: uint};

fn gather_comments_and_literals(cm: codemap::codemap, path: str,
                                srdr: io::reader) ->
   {cmnts: [cmnt], lits: [lit]} {
    let src = str::unsafe_from_bytes(srdr.read_whole_stream());
    let itr = @interner::mk::<str>(str::hash, str::eq);
    let rdr = new_reader(cm, src, codemap::new_filemap(path, 0u, 0u), itr);
    let comments: [cmnt] = [];
    let literals: [lit] = [];
    let first_read: bool = true;
    while !rdr.is_eof() {
        while true {
            let code_to_the_left = !first_read;
            consume_non_eol_whitespace(rdr);
            if rdr.curr() == '\n' {
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
        log "tok: " + token::to_str(rdr, tok.tok);
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
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
//
