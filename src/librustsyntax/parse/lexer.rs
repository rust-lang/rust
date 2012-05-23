import util::interner;
import util::interner::intern;
import diagnostic;

export reader, new_reader, next_token, is_whitespace;

type reader = @{
    span_diagnostic: diagnostic::span_handler,
    src: @str,
    mut col: uint,
    mut pos: uint,
    mut curr: char,
    mut chpos: uint,
    filemap: codemap::filemap,
    interner: @interner::interner<str>
};

impl reader for reader {
    fn is_eof() -> bool { self.curr == -1 as char }
    fn get_str_from(start: uint) -> str unsafe {
        // I'm pretty skeptical about this subtraction. What if there's a
        // multi-byte character before the mark?
        ret str::slice(*self.src, start - 1u, self.pos - 1u);
    }
    fn next() -> char {
        if self.pos < (*self.src).len() {
            ret str::char_at(*self.src, self.pos);
        } else { ret -1 as char; }
    }
    fn bump() {
        if self.pos < (*self.src).len() {
            self.col += 1u;
            self.chpos += 1u;
            if self.curr == '\n' {
                codemap::next_line(self.filemap, self.chpos, self.pos);
                self.col = 0u;
            }
            let next = str::char_range_at(*self.src, self.pos);
            self.pos = next.next;
            self.curr = next.ch;
        } else {
            if (self.curr != -1 as char) {
                self.col += 1u;
                self.chpos += 1u;
                self.curr = -1 as char;
            }
        }
    }
    fn fatal(m: str) -> ! {
        self.span_diagnostic.span_fatal(
            ast_util::mk_sp(self.chpos, self.chpos),
            m)
    }
}

fn new_reader(span_diagnostic: diagnostic::span_handler,
              filemap: codemap::filemap,
              itr: @interner::interner<str>) -> reader {
    let r = @{span_diagnostic: span_diagnostic, src: filemap.src,
              mut col: 0u, mut pos: 0u, mut curr: -1 as char,
              mut chpos: filemap.start_pos.ch,
              filemap: filemap, interner: itr};
    if r.pos < (*filemap.src).len() {
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
    ret consume_any_line_comment(rdr);
}

fn consume_any_line_comment(rdr: reader) {
    if rdr.curr == '/' {
        alt rdr.next() {
          '/' {
            while rdr.curr != '\n' && !rdr.is_eof() { rdr.bump(); }
            // Restart whitespace munch.

            ret consume_whitespace_and_comments(rdr);
          }
          '*' { rdr.bump(); rdr.bump(); ret consume_block_comment(rdr); }
          _ { ret; }
        }
    }
}

fn consume_block_comment(rdr: reader) {
    let mut level: int = 1;
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

    ret consume_whitespace_and_comments(rdr);
}

fn scan_exponent(rdr: reader) -> option<str> {
    let mut c = rdr.curr;
    let mut rslt = "";
    if c == 'e' || c == 'E' {
        str::push_char(rslt, c);
        rdr.bump();
        c = rdr.curr;
        if c == '-' || c == '+' {
            str::push_char(rslt, c);
            rdr.bump();
        }
        let exponent = scan_digits(rdr, 10u);
        if str::len(exponent) > 0u {
            ret some(rslt + exponent);
        } else { rdr.fatal("scan_exponent: bad fp literal"); }
    } else { ret none::<str>; }
}

fn scan_digits(rdr: reader, radix: uint) -> str {
    let mut rslt = "";
    loop {
        let c = rdr.curr;
        if c == '_' { rdr.bump(); cont; }
        alt char::to_digit(c, radix) {
          some(d) {
            str::push_char(rslt, c);
            rdr.bump();
          }
          _ { ret rslt; }
        }
    };
}

fn scan_number(c: char, rdr: reader) -> token::token {
    let mut num_str, base = 10u, c = c, n = rdr.next();
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
        let signed = c == 'i';
        let mut tp = {
            if signed { either::left(ast::ty_i) }
            else { either::right(ast::ty_u) }
        };
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
        if str::len(num_str) == 0u {
            rdr.fatal("no valid digits found for number");
        }
        let parsed = option::get(u64::from_str_radix(num_str, base as u64));
        alt tp {
          either::left(t) { ret token::LIT_INT(parsed as i64, t); }
          either::right(t) { ret token::LIT_UINT(parsed, t); }
        }
    }
    let mut is_float = false;
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
            64-bit float, it won't be noticed till the back-end (Issue #2252)
            */
        } else {
            is_float = true;
        }
    }
    if is_float {
        ret token::LIT_FLOAT(interner::intern(*rdr.interner, num_str),
                             ast::ty_f);
    } else {
        if str::len(num_str) == 0u {
            rdr.fatal("no valid digits found for number");
        }
        let parsed = option::get(u64::from_str_radix(num_str, base as u64));
        ret token::LIT_INT(parsed as i64, ast::ty_i);
    }
}

fn scan_numeric_escape(rdr: reader, n_hex_digits: uint) -> char {
    let mut accum_int = 0, i = n_hex_digits;
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
    let mut accum_str = "";
    let mut c = rdr.curr;
    if (c >= 'a' && c <= 'z')
        || (c >= 'A' && c <= 'Z')
        || c == '_'
        || (c > 'z' && char::is_XID_start(c)) {
        while (c >= 'a' && c <= 'z')
            || (c >= 'A' && c <= 'Z')
            || (c >= '0' && c <= '9')
            || c == '_'
            || (c > 'z' && char::is_XID_continue(c)) {
            str::push_char(accum_str, c);
            rdr.bump();
            c = rdr.curr;
        }
        if str::eq(accum_str, "_") { ret token::UNDERSCORE; }
        let is_mod_name = c == ':' && rdr.next() == ':';

        // FIXME: perform NFKC normalization here. (Issue #2253)
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
      '#' { rdr.bump(); ret token::POUND; }
      '~' { rdr.bump(); ret token::TILDE; }
      ':' {
        rdr.bump();
        if rdr.curr == ':' {
            rdr.bump();
            ret token::MOD_SEP;
        } else { ret token::COLON; }
      }

      '$' { rdr.bump(); ret token::DOLLAR; }





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
          '<' { ret binop(rdr, token::SHL); }
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
          '>' { ret binop(rdr, token::SHR); }
          _ { ret token::GT; }
        }
      }
      '\'' {
        rdr.bump();
        let mut c2 = rdr.curr;
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
                  'n' { str::push_char(accum_str, '\n'); }
                  'r' { str::push_char(accum_str, '\r'); }
                  't' { str::push_char(accum_str, '\t'); }
                  '\\' { str::push_char(accum_str, '\\'); }
                  '"' { str::push_char(accum_str, '"'); }
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
      c { rdr.fatal(#fmt["unknown start of token: %d", c as int]); }
    }
}

fn consume_whitespace(rdr: reader) {
    while is_whitespace(rdr.curr) && !rdr.is_eof() { rdr.bump(); }
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
