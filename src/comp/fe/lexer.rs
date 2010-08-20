import std._io.stdio_reader;

fn in_range(char c, char lo, char hi) -> bool {
    ret lo <= c && c <= hi;
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
    ret c == ' ' || c == '\t' || c == '\r' || c == '\n';
}

fn next_token(stdio_reader rdr) -> token.token {
    auto eof = (-1) as char;
    auto c = rdr.getc() as char;
    auto accum_str = "";
    auto accum_int = 0;

    while (is_whitespace(c) && c != eof) {
        c = rdr.getc() as char;
    }

    if (c == eof) { ret token.EOF(); }

    if (is_alpha(c)) {
        while (is_alpha(c)) {
            accum_str += (c as u8);
            c = rdr.getc() as char;
        }
        rdr.ungetc(c as int);
        ret token.IDENT(accum_str);
    }

    if (is_dec_digit(c)) {
        if (c == '0') {
        } else {
            while (is_dec_digit(c)) {
                accum_int *= 10;
                accum_int += (c as int) - ('0' as int);
                c = rdr.getc() as char;
            }
            rdr.ungetc(c as int);
            ret token.LIT_INT(accum_int);
        }
    }

    // One-byte structural symbols.
    alt (c) {
        case (';') { ret token.SEMI(); }
        case (',') { ret token.COMMA(); }
        case ('.') { ret token.DOT(); }
        case ('(') { ret token.LPAREN(); }
        case (')') { ret token.RPAREN(); }
        case ('{') { ret token.LBRACE(); }
        case ('}') { ret token.RBRACE(); }
        case ('[') { ret token.LBRACKET(); }
        case (']') { ret token.RBRACKET(); }
        case ('@') { ret token.AT(); }
        case ('#') { ret token.POUND(); }
        case ('=') {
            auto c2 = rdr.getc() as char;
            if (c2 == '=') {
                ret token.OP(token.EQEQ());
            } else {
                rdr.ungetc(c2 as int);
                ret token.OP(token.EQ());
            }
        }
    }

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
