import std._io.stdio_reader;

fn next_token(stdio_reader rdr) -> token.token {
    auto c = rdr.getc();
    log "got char";
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
