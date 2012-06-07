import util::interner;
import util::interner::intern;
import diagnostic;
import ast::{tt_delim,tt_flat};

export reader, string_reader, new_string_reader, is_whitespace;
export tt_reader,  new_tt_reader, dup_tt_reader;
export nextch, is_eof, bump, get_str_from;
export string_reader_as_reader, tt_reader_as_reader;

iface reader {
    fn is_eof() -> bool;
    fn next_token() -> {tok: token::token, chpos: uint};
    fn fatal(str) -> !;
    fn chpos() -> uint;
    fn interner() -> @interner::interner<@str>;
}

enum tt_frame_up { /* to break a circularity */
    tt_frame_up(option<tt_frame>)
}

#[doc = "an unzipping of `token_tree`s"]
type tt_frame = @{
    readme: [ast::token_tree],
    mut idx: uint,
    up: tt_frame_up
};

type tt_reader = @{
    span_diagnostic: diagnostic::span_handler,
    interner: @interner::interner<@str>,
    mut cur: tt_frame,
    /* cached: */
    mut cur_tok: token::token,
    mut cur_chpos: uint
};

fn new_tt_reader(span_diagnostic: diagnostic::span_handler,
                 itr: @interner::interner<@str>, src: [ast::token_tree])
    -> tt_reader {
    let r = @{span_diagnostic: span_diagnostic, interner: itr,
              mut cur: @{readme: src, mut idx: 0u,
                         up: tt_frame_up(option::none)},
              mut cur_tok: token::EOF, /* dummy value, never read */
              mut cur_chpos: 0u /* dummy value, never read */
             };
    (r as reader).next_token(); /* get cur_tok and cur_chpos set up */
    ret r;
}

pure fn dup_tt_frame(&&f: tt_frame) -> tt_frame {
    @{readme: f.readme, mut idx: f.idx,
      up: alt f.up {
        tt_frame_up(o_f) {
          tt_frame_up(option::map(o_f, dup_tt_frame))
        }
      }
     }
}

pure fn dup_tt_reader(&&r: tt_reader) -> tt_reader {
    @{span_diagnostic: r.span_diagnostic, interner: r.interner,
      mut cur: dup_tt_frame(r.cur),
      mut cur_tok: r.cur_tok, mut cur_chpos: r.cur_chpos}
}

type string_reader = @{
    span_diagnostic: diagnostic::span_handler,
    src: @str,
    mut col: uint,
    mut pos: uint,
    mut curr: char,
    mut chpos: uint,
    filemap: codemap::filemap,
    interner: @interner::interner<@str>
};

fn new_string_reader(span_diagnostic: diagnostic::span_handler,
                     filemap: codemap::filemap,
                     itr: @interner::interner<@str>) -> string_reader {
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

impl string_reader_as_reader of reader for string_reader {
    fn is_eof() -> bool { is_eof(self) }
    fn next_token() -> {tok: token::token, chpos: uint} {
        consume_whitespace_and_comments(self);
        let start_chpos = self.chpos;
        let tok = if is_eof(self) {
            token::EOF
        } else {
            next_token_inner(self)
        };
        ret {tok: tok, chpos: start_chpos};
    }
    fn fatal(m: str) -> ! {
        self.span_diagnostic.span_fatal(
            ast_util::mk_sp(self.chpos, self.chpos), m)
    }
    fn chpos() -> uint { self.chpos }
    fn interner() -> @interner::interner<@str> { self.interner }
}

impl tt_reader_as_reader of reader for tt_reader {
    fn is_eof() -> bool { self.cur_tok == token::EOF }
    fn next_token() -> {tok: token::token, chpos: uint} {
        let ret_val = { tok: self.cur_tok, chpos: self.cur_chpos };
        if self.cur.idx >= vec::len(self.cur.readme) {
            /* done with this set; pop */
            alt self.cur.up {
              tt_frame_up(option::none) {
                self.cur_tok = token::EOF;
                ret ret_val;
              }
              tt_frame_up(option::some(tt_f)) {
                self.cur = tt_f;
                /* the above `if` would need to be a `while` if we didn't know
                that the last thing in a `tt_delim` is always a `tt_flat` */
                self.cur.idx += 1u;
              }
            }
        }
        /* if `tt_delim`s could be 0-length, we'd need to be able to switch
        between popping and pushing until we got to an actual `tt_flat` */
        loop { /* because it's easiest, this handles `tt_delim` not starting
                  with a `tt_flat`, even though it won't happen */
            alt self.cur.readme[self.cur.idx] {
              tt_delim(tts) {
                self.cur = @{readme: tts, mut idx: 0u,
                             up: tt_frame_up(option::some(self.cur)) };
              }
              tt_flat(chpos, tok) {
                self.cur_chpos = chpos; self.cur_tok = tok;
                self.cur.idx += 1u;
                ret ret_val;
              }
          }
        }
    }
    fn fatal(m: str) -> ! {
        self.span_diagnostic.span_fatal(
            ast_util::mk_sp(self.chpos(), self.chpos()), m);
    }
    fn chpos() -> uint { self.cur_chpos }
    fn interner() -> @interner::interner<@str> { self.interner }
}

fn get_str_from(rdr: string_reader, start: uint) -> str unsafe {
    // I'm pretty skeptical about this subtraction. What if there's a
    // multi-byte character before the mark?
    ret str::slice(*rdr.src, start - 1u, rdr.pos - 1u);
}

fn bump(rdr: string_reader) {
    if rdr.pos < (*rdr.src).len() {
        rdr.col += 1u;
        rdr.chpos += 1u;
        if rdr.curr == '\n' {
            codemap::next_line(rdr.filemap, rdr.chpos, rdr.pos);
            rdr.col = 0u;
        }
        let next = str::char_range_at(*rdr.src, rdr.pos);
        rdr.pos = next.next;
        rdr.curr = next.ch;
    } else {
        if (rdr.curr != -1 as char) {
            rdr.col += 1u;
            rdr.chpos += 1u;
            rdr.curr = -1 as char;
        }
    }
}
fn is_eof(rdr: string_reader) -> bool {
    rdr.curr == -1 as char
}
fn nextch(rdr: string_reader) -> char {
    if rdr.pos < (*rdr.src).len() {
        ret str::char_at(*rdr.src, rdr.pos);
    } else { ret -1 as char; }
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

fn consume_whitespace_and_comments(rdr: string_reader) {
    while is_whitespace(rdr.curr) { bump(rdr); }
    ret consume_any_line_comment(rdr);
}

fn consume_any_line_comment(rdr: string_reader) {
    if rdr.curr == '/' {
        alt nextch(rdr) {
          '/' {
            while rdr.curr != '\n' && !is_eof(rdr) { bump(rdr); }
            // Restart whitespace munch.

            ret consume_whitespace_and_comments(rdr);
          }
          '*' { bump(rdr); bump(rdr); ret consume_block_comment(rdr); }
          _ { ret; }
        }
    } else if rdr.curr == '#' {
        if nextch(rdr) == '!' {
            let cmap = codemap::new_codemap();
            (*cmap).files.push(rdr.filemap);
            let loc = codemap::lookup_char_pos_adj(cmap, rdr.chpos);
            if loc.line == 1u && loc.col == 0u {
                while rdr.curr != '\n' && !is_eof(rdr) { bump(rdr); }
                ret consume_whitespace_and_comments(rdr);
            }
        }
    }
}

fn consume_block_comment(rdr: string_reader) {
    let mut level: int = 1;
    while level > 0 {
        if is_eof(rdr) { rdr.fatal("unterminated block comment"); }
        if rdr.curr == '/' && nextch(rdr) == '*' {
            bump(rdr);
            bump(rdr);
            level += 1;
        } else {
            if rdr.curr == '*' && nextch(rdr) == '/' {
                bump(rdr);
                bump(rdr);
                level -= 1;
            } else { bump(rdr); }
        }
    }
    // restart whitespace munch.

    ret consume_whitespace_and_comments(rdr);
}

fn scan_exponent(rdr: string_reader) -> option<str> {
    let mut c = rdr.curr;
    let mut rslt = "";
    if c == 'e' || c == 'E' {
        str::push_char(rslt, c);
        bump(rdr);
        c = rdr.curr;
        if c == '-' || c == '+' {
            str::push_char(rslt, c);
            bump(rdr);
        }
        let exponent = scan_digits(rdr, 10u);
        if str::len(exponent) > 0u {
            ret some(rslt + exponent);
        } else { rdr.fatal("scan_exponent: bad fp literal"); }
    } else { ret none::<str>; }
}

fn scan_digits(rdr: string_reader, radix: uint) -> str {
    let mut rslt = "";
    loop {
        let c = rdr.curr;
        if c == '_' { bump(rdr); cont; }
        alt char::to_digit(c, radix) {
          some(d) {
            str::push_char(rslt, c);
            bump(rdr);
          }
          _ { ret rslt; }
        }
    };
}

fn scan_number(c: char, rdr: string_reader) -> token::token {
    let mut num_str, base = 10u, c = c, n = nextch(rdr);
    if c == '0' && n == 'x' {
        bump(rdr);
        bump(rdr);
        base = 16u;
    } else if c == '0' && n == 'b' {
        bump(rdr);
        bump(rdr);
        base = 2u;
    }
    num_str = scan_digits(rdr, base);
    c = rdr.curr;
    nextch(rdr);
    if c == 'u' || c == 'i' {
        let signed = c == 'i';
        let mut tp = {
            if signed { either::left(ast::ty_i) }
            else { either::right(ast::ty_u) }
        };
        bump(rdr);
        c = rdr.curr;
        if c == '8' {
            bump(rdr);
            tp = if signed { either::left(ast::ty_i8) }
                      else { either::right(ast::ty_u8) };
        }
        n = nextch(rdr);
        if c == '1' && n == '6' {
            bump(rdr);
            bump(rdr);
            tp = if signed { either::left(ast::ty_i16) }
                      else { either::right(ast::ty_u16) };
        } else if c == '3' && n == '2' {
            bump(rdr);
            bump(rdr);
            tp = if signed { either::left(ast::ty_i32) }
                      else { either::right(ast::ty_u32) };
        } else if c == '6' && n == '4' {
            bump(rdr);
            bump(rdr);
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
    if rdr.curr == '.' && !(is_alpha(nextch(rdr)) || nextch(rdr) == '_') {
        is_float = true;
        bump(rdr);
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
        bump(rdr);
        c = rdr.curr;
        n = nextch(rdr);
        if c == '3' && n == '2' {
            bump(rdr);
            bump(rdr);
            ret token::LIT_FLOAT(intern(*rdr.interner, @num_str),
                                 ast::ty_f32);
        } else if c == '6' && n == '4' {
            bump(rdr);
            bump(rdr);
            ret token::LIT_FLOAT(intern(*rdr.interner, @num_str),
                                 ast::ty_f64);
            /* FIXME: if this is out of range for either a 32-bit or
            64-bit float, it won't be noticed till the back-end (Issue #2252)
            */
        } else {
            is_float = true;
        }
    }
    if is_float {
        ret token::LIT_FLOAT(intern(*rdr.interner, @num_str),
                             ast::ty_f);
    } else {
        if str::len(num_str) == 0u {
            rdr.fatal("no valid digits found for number");
        }
        let parsed = option::get(u64::from_str_radix(num_str, base as u64));

        #debug["lexing %s as an unsuffixed integer literal",
               num_str];
        ret token::LIT_INT_UNSUFFIXED(parsed as i64);
    }
}

fn scan_numeric_escape(rdr: string_reader, n_hex_digits: uint) -> char {
    let mut accum_int = 0, i = n_hex_digits;
    while i != 0u {
        let n = rdr.curr;
        bump(rdr);
        if !is_hex_digit(n) {
            rdr.fatal(#fmt["illegal numeric character escape: %d", n as int]);
        }
        accum_int *= 16;
        accum_int += hex_digit_val(n);
        i -= 1u;
    }
    ret accum_int as char;
}

fn next_token_inner(rdr: string_reader) -> token::token {
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
            bump(rdr);
            c = rdr.curr;
        }
        if str::eq(accum_str, "_") { ret token::UNDERSCORE; }
        let is_mod_name = c == ':' && nextch(rdr) == ':';

        // FIXME: perform NFKC normalization here. (Issue #2253)
        ret token::IDENT(interner::intern(*rdr.interner,
                                          @accum_str), is_mod_name);
    }
    if is_dec_digit(c) {
        ret scan_number(c, rdr);
    }
    fn binop(rdr: string_reader, op: token::binop) -> token::token {
        bump(rdr);
        if rdr.curr == '=' {
            bump(rdr);
            ret token::BINOPEQ(op);
        } else { ret token::BINOP(op); }
    }
    alt c {





      // One-byte tokens.
      ';' { bump(rdr); ret token::SEMI; }
      ',' { bump(rdr); ret token::COMMA; }
      '.' {
        bump(rdr);
        if rdr.curr == '.' && nextch(rdr) == '.' {
            bump(rdr);
            bump(rdr);
            ret token::ELLIPSIS;
        }
        ret token::DOT;
      }
      '(' { bump(rdr); ret token::LPAREN; }
      ')' { bump(rdr); ret token::RPAREN; }
      '{' { bump(rdr); ret token::LBRACE; }
      '}' { bump(rdr); ret token::RBRACE; }
      '[' { bump(rdr); ret token::LBRACKET; }
      ']' { bump(rdr); ret token::RBRACKET; }
      '@' { bump(rdr); ret token::AT; }
      '#' { bump(rdr); ret token::POUND; }
      '~' { bump(rdr); ret token::TILDE; }
      ':' {
        bump(rdr);
        if rdr.curr == ':' {
            bump(rdr);
            ret token::MOD_SEP;
        } else { ret token::COLON; }
      }

      '$' { bump(rdr); ret token::DOLLAR; }





      // Multi-byte tokens.
      '=' {
        bump(rdr);
        if rdr.curr == '=' {
            bump(rdr);
            ret token::EQEQ;
        } else if rdr.curr == '>' {
            bump(rdr);
            ret token::FAT_ARROW;
        } else {
            ret token::EQ;
        }
      }
      '!' {
        bump(rdr);
        if rdr.curr == '=' {
            bump(rdr);
            ret token::NE;
        } else { ret token::NOT; }
      }
      '<' {
        bump(rdr);
        alt rdr.curr {
          '=' { bump(rdr); ret token::LE; }
          '<' { ret binop(rdr, token::SHL); }
          '-' {
            bump(rdr);
            alt rdr.curr {
              '>' { bump(rdr); ret token::DARROW; }
              _ { ret token::LARROW; }
            }
          }
          _ { ret token::LT; }
        }
      }
      '>' {
        bump(rdr);
        alt rdr.curr {
          '=' { bump(rdr); ret token::GE; }
          '>' { ret binop(rdr, token::SHR); }
          _ { ret token::GT; }
        }
      }
      '\'' {
        bump(rdr);
        let mut c2 = rdr.curr;
        bump(rdr);
        if c2 == '\\' {
            let escaped = rdr.curr;
            bump(rdr);
            alt escaped {
              'n' { c2 = '\n'; }
              'r' { c2 = '\r'; }
              't' { c2 = '\t'; }
              '\\' { c2 = '\\'; }
              '\'' { c2 = '\''; }
              '"' { c2 = '"'; }
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
        bump(rdr); // advance curr past token
        ret token::LIT_INT(c2 as i64, ast::ty_char);
      }
      '"' {
        let n = rdr.chpos;
        bump(rdr);
        while rdr.curr != '"' {
            if is_eof(rdr) {
                rdr.fatal(#fmt["unterminated double quote string: %s",
                               get_str_from(rdr, n)]);
            }

            let ch = rdr.curr;
            bump(rdr);
            alt ch {
              '\\' {
                let escaped = rdr.curr;
                bump(rdr);
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
        bump(rdr);
        ret token::LIT_STR(interner::intern(*rdr.interner,
                                            @accum_str));
      }
      '-' {
        if nextch(rdr) == '>' {
            bump(rdr);
            bump(rdr);
            ret token::RARROW;
        } else { ret binop(rdr, token::MINUS); }
      }
      '&' {
        if nextch(rdr) == '&' {
            bump(rdr);
            bump(rdr);
            ret token::ANDAND;
        } else { ret binop(rdr, token::AND); }
      }
      '|' {
        alt nextch(rdr) {
          '|' { bump(rdr); bump(rdr); ret token::OROR; }
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

fn consume_whitespace(rdr: string_reader) {
    while is_whitespace(rdr.curr) && !is_eof(rdr) { bump(rdr); }
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
