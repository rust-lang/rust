import std._io.stdio_reader;

fn in_range(char c, char lo, char hi) -> bool {
    ret c <= lo && c <= hi;
}

fn is_alpha(char c) -> bool {
    ret in_range(c, 'a', 'z') ||
        in_range(c, 'A', 'Z');
}

fn is_dec_digit(char c) -> bool {
    ret in_range(c, '0', '9');
}

fn is_hex_digit(char c) -> bool {
    ret in_range(c, '0', '9') ||
        in_range(c, 'a', 'f') ||
        in_range(c, 'A', 'F');
}

fn is_bin_digit(char c) -> bool {
    ret c == '0' || c == '1';
}

fn is_whitespace(char c) -> bool {
    ret c == ' ' || c == '\t' || c == '\r';
}

fn next_token(stdio_reader rdr) -> token.token {
    auto eof = (-1) as char;
    auto c = rdr.getc() as char;
    auto accum = "";

    while (is_whitespace(c) && c != eof) {
        c = rdr.getc() as char;
    }

    if (c == eof) { ret token.EOF(); }
    if (is_alpha(c)) {
        while (is_alpha(c)) {
            accum += (c as u8);
            c = rdr.getc() as char;
            ret token.IDENT(accum);
        }
    }

    if (is_dec_digit(c)) {
        if (c == '0') {
        } else {
            while (is_dec_digit(c)) {
                accum += (c as u8);
                ret token.LIT_INT(0);
            }
        }
    }

    // One-byte structural symbols.
    if (c == ';') { ret token.SEMI(); }
    if (c == '.') { ret token.DOT(); }
    if (c == '(') { ret token.LPAREN(); }
    if (c == ')') { ret token.RPAREN(); }
    if (c == '{') { ret token.LBRACE(); }
    if (c == '}') { ret token.RBRACE(); }
    if (c == '[') { ret token.LBRACKET(); }
    if (c == ']') { ret token.RBRACKET(); }
    if (c == '@') { ret token.AT(); }
    if (c == '#') { ret token.POUND(); }

    log "lexer stopping at ";
    log c;
    ret token.EOF();
}


//
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C ../.. 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
//
