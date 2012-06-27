import util::interner::{interner,intern};
import diagnostic::span_handler;
import codemap::span;
import ext::tt::transcribe::{tt_reader,  new_tt_reader, dup_tt_reader,
                             tt_next_token};

export reader, string_reader, new_string_reader, is_whitespace;
export tt_reader,  new_tt_reader;
export nextch, is_eof, bump, get_str_from, new_low_level_string_reader;
export string_reader_as_reader, tt_reader_as_reader;

iface reader {
    fn is_eof() -> bool;
    fn next_token() -> {tok: token::token, sp: span};
    fn fatal(str) -> !;
    fn span_diag() -> span_handler;
    fn interner() -> @interner<@str>;
    fn peek() -> {tok: token::token, sp: span};
    fn dup() -> reader;
}

type string_reader = @{
    span_diagnostic: span_handler,
    src: @str,
    mut col: uint,
    mut pos: uint,
    mut curr: char,
    mut chpos: uint,
    filemap: codemap::filemap,
    interner: @interner<@str>,
    /* cached: */
    mut peek_tok: token::token,
    mut peek_span: span
};

fn new_string_reader(span_diagnostic: span_handler,
                     filemap: codemap::filemap,
                     itr: @interner<@str>) -> string_reader {
    let r = new_low_level_string_reader(span_diagnostic, filemap, itr);
    string_advance_token(r); /* fill in peek_* */
    ret r;
}

/* For comments.rs, which hackily pokes into 'pos' and 'curr' */
fn new_low_level_string_reader(span_diagnostic: span_handler,
                               filemap: codemap::filemap,
                               itr: @interner<@str>)
    -> string_reader {
    let r = @{span_diagnostic: span_diagnostic, src: filemap.src,
              mut col: 0u, mut pos: 0u, mut curr: -1 as char,
              mut chpos: filemap.start_pos.ch,
              filemap: filemap, interner: itr,
              /* dummy values; not read */
              mut peek_tok: token::EOF,
              mut peek_span: ast_util::mk_sp(0u,0u)};
    if r.pos < (*filemap.src).len() {
        let next = str::char_range_at(*r.src, r.pos);
        r.pos = next.next;
        r.curr = next.ch;
    }
    ret r;
}

fn dup_string_reader(&&r: string_reader) -> string_reader {
    @{span_diagnostic: r.span_diagnostic, src: r.src,
      mut col: r.col, mut pos: r.pos, mut curr: r.curr, mut chpos: r.chpos,
      filemap: r.filemap, interner: r.interner,
      mut peek_tok: r.peek_tok, mut peek_span: r.peek_span}
}

impl string_reader_as_reader of reader for string_reader {
    fn is_eof() -> bool { is_eof(self) }
    fn next_token() -> {tok: token::token, sp: span} {
        let ret_val = {tok: self.peek_tok, sp: self.peek_span};
        string_advance_token(self);
        ret ret_val;
    }
    fn fatal(m: str) -> ! {
        self.span_diagnostic.span_fatal(copy self.peek_span, m)
    }
    fn span_diag() -> span_handler { self.span_diagnostic }
    fn interner() -> @interner<@str> { self.interner }
    fn peek() -> {tok: token::token, sp: span} {
        {tok: self.peek_tok, sp: self.peek_span}
    }
    fn dup() -> reader { dup_string_reader(self) as reader }
}

impl tt_reader_as_reader of reader for tt_reader {
    fn is_eof() -> bool { self.cur_tok == token::EOF }
    fn next_token() -> {tok: token::token, sp: span} {
        /* weird resolve bug: if the following `if`, or any of its
        statements are removed, we get resolution errors */
        if false {
            let _ignore_me = 0;
            let _me_too = self.cur.readme[self.cur.idx];
        }
        tt_next_token(self)
    }
    fn fatal(m: str) -> ! {
        self.span_diagnostic.span_fatal(copy self.cur_span, m);
    }
    fn span_diag() -> span_handler { self.span_diagnostic }
    fn interner() -> @interner<@str> { self.interner }
    fn peek() -> {tok: token::token, sp: span} {
        { tok: self.cur_tok, sp: self.cur_span }
    }
    fn dup() -> reader { dup_tt_reader(self) as reader }
}

fn string_advance_token(&&r: string_reader) {
    for consume_whitespace_and_comments(r).each |comment| {
        r.peek_tok = comment.tok;
        r.peek_span = comment.sp;
        ret;
    }

    if is_eof(r) {
        r.peek_tok = token::EOF;
    } else {
        let start_chpos = r.chpos;
        r.peek_tok = next_token_inner(r);
        r.peek_span = ast_util::mk_sp(start_chpos, r.chpos);
    };

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

// might return a sugared-doc-attr
fn consume_whitespace_and_comments(rdr: string_reader)
                                -> option<{tok: token::token, sp: span}> {
    while is_whitespace(rdr.curr) { bump(rdr); }
    ret consume_any_line_comment(rdr);
}

// might return a sugared-doc-attr
fn consume_any_line_comment(rdr: string_reader)
                                -> option<{tok: token::token, sp: span}> {
    if rdr.curr == '/' {
        alt nextch(rdr) {
          '/' {
            bump(rdr);
            bump(rdr);
            // line comments starting with "///" or "//!" are doc-comments
            if rdr.curr == '/' || rdr.curr == '!' {
                let start_chpos = rdr.chpos - 2u;
                let mut acc = "//";
                while rdr.curr != '\n' && !is_eof(rdr) {
                    str::push_char(acc, rdr.curr);
                    bump(rdr);
                }
                ret some({
                    tok: token::DOC_COMMENT(intern(*rdr.interner, @acc)),
                    sp: ast_util::mk_sp(start_chpos, rdr.chpos)
                });
            } else {
                while rdr.curr != '\n' && !is_eof(rdr) { bump(rdr); }
                // Restart whitespace munch.
                ret consume_whitespace_and_comments(rdr);
            }
          }
          '*' { bump(rdr); bump(rdr); ret consume_block_comment(rdr); }
          _ {}
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
    ret none;
}

// might return a sugared-doc-attr
fn consume_block_comment(rdr: string_reader)
                                -> option<{tok: token::token, sp: span}> {

    // block comments starting with "/**" or "/*!" are doc-comments
    if rdr.curr == '*' || rdr.curr == '!' {
        let start_chpos = rdr.chpos - 2u;
        let mut acc = "/*";
        while !(rdr.curr == '*' && nextch(rdr) == '/') && !is_eof(rdr) {
            str::push_char(acc, rdr.curr);
            bump(rdr);
        }
        if is_eof(rdr) {
            rdr.fatal("unterminated block doc-comment");
        } else {
            acc += "*/";
            bump(rdr);
            bump(rdr);
            ret some({
                tok: token::DOC_COMMENT(intern(*rdr.interner, @acc)),
                sp: ast_util::mk_sp(start_chpos, rdr.chpos)
            });
        }
    }

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
            /* FIXME (#2252): if this is out of range for either a
            32-bit or 64-bit float, it won't be noticed till the
            back-end.  */
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
        ret token::IDENT(intern(*rdr.interner,
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
        ret token::LIT_STR(intern(*rdr.interner, @accum_str));
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
