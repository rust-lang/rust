import std._io;

state type parser =
    state obj {
          state fn peek() -> token.token;
          state fn bump();
    };

fn new_parser(str path) -> parser {
    state obj stdio_parser(mutable token.token tok,
                           _io.stdio_reader rdr)
        {
            state fn peek() -> token.token {
                ret tok;
            }
            state fn bump() {
                tok = lexer.next_token(rdr);
            }
        }
    auto rdr = _io.new_stdio_reader(path);
    ret stdio_parser(lexer.next_token(rdr), rdr);
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
