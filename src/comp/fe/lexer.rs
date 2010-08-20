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
        accum += (c as u8);
    }
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
