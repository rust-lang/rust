import std._io;
import driver.session;
import util.common;

state type parser =
    state obj {
          state fn peek() -> token.token;
          state fn bump();
          io fn err(str s);
          fn get_session() -> session.session;
          fn get_span() -> common.span;
    };

state fn new_parser(session.session sess, str path) -> parser {
    state obj stdio_parser(session.session sess,
                           mutable token.token tok,
                           mutable common.pos lo,
                           mutable common.pos hi,
                           lexer.reader rdr)
        {
            state fn peek() -> token.token {
                ret tok;
            }

            state fn bump() {
                tok = lexer.next_token(rdr);
                lo = rdr.get_mark_pos();
                hi = rdr.get_curr_pos();
            }

            io fn err(str m) {
                auto span = rec(filename = rdr.get_filename(),
                                lo = lo, hi = hi);
                sess.span_err(span, m);
            }

            fn get_session() -> session.session {
                ret sess;
            }

            fn get_span() -> common.span {
                ret rec(filename = rdr.get_filename(),
                        lo = lo, hi = hi);
            }
        }
    auto srdr = _io.new_stdio_reader(path);
    auto rdr = lexer.new_reader(srdr, path);
    auto npos = rdr.get_curr_pos();
    ret stdio_parser(sess, lexer.next_token(rdr), npos, npos, rdr);
}

state fn expect(parser p, token.token t) {
    // FIXME: comparing tags would be good. One of these days.
    if (true /* p.peek() == t */) {
        p.bump();
    } else {
        let str s = "expecting ";
        s += token.to_str(t);
        s += ", found ";
        s += token.to_str(t);
        p.err(s);
    }
}

state fn parse_ident(parser p) -> ast.ident {
    alt (p.peek()) {
        case (token.IDENT(i)) { ret i; }
        case (_) {
            p.err("expecting ident");
            fail;
        }
    }
}

state fn parse_item(parser p) -> tup(ast.ident, ast.item) {
    alt (p.peek()) {
        case (token.FN()) {
            p.bump();
            auto id = parse_ident(p);
            expect(p, token.LPAREN());
            let vec[rec(ast.slot slot, ast.ident ident)] inputs = vec();
            let vec[@ast.stmt] body = vec();
            auto output = rec(ty = ast.ty_nil(), mode = ast.val() );
            let ast._fn f = rec(inputs = inputs,
                                output = output,
                                body = body);
            ret tup(id, ast.item_fn(@f));
        }
    }
    p.err("expecting item");
    fail;
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
