import std._io;
import driver.session;
import util.common;
import util.common.new_str_hash;

// FIXME: import std.util.option and use it here.
// import std.util.option;

tag option[T] {
  none;
  some(T);
}


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
                log token.to_str(tok);
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
    if (p.peek() == t) {
        p.bump();
    } else {
        let str s = "expecting ";
        s += token.to_str(t);
        s += ", found ";
        s += token.to_str(p.peek());
        p.err(s);
    }
}

state fn parse_ident(parser p) -> ast.ident {
    alt (p.peek()) {
        case (token.IDENT(?i)) { p.bump(); ret i; }
        case (_) {
            p.err("expecting ident");
            fail;
        }
    }
}

state fn parse_ty(parser p) -> ast.ty {
    alt (p.peek()) {
        case (token.INT) { p.bump(); ret ast.ty_int; }
        case (token.UINT) { p.bump(); ret ast.ty_int; }
        case (token.STR) { p.bump(); ret ast.ty_str; }
        case (token.CHAR) { p.bump(); ret ast.ty_char; }
        case (token.MACH(?tm)) { p.bump(); ret ast.ty_machine(tm); }
    }
    p.err("expecting type");
    fail;
}

state fn parse_slot(parser p) -> ast.slot {
    let ast.mode m = ast.val;
    if (p.peek() == token.BINOP(token.AND)) {
        m = ast.alias;
        p.bump();
    }
    let ast.ty t = parse_ty(p);
    ret rec(ty=t, mode=m);
}

state fn parse_seq[T](token.token bra,
                      token.token ket,
                      option[token.token] sep,
                      (state fn(parser) -> T) f,
                      parser p) -> vec[T] {
    let bool first = true;
    expect(p, bra);
    let vec[T] v = vec();
    while (p.peek() != ket) {
        alt(sep) {
            case (some[token.token](?t)) {
                if (first) {
                    first = false;
                } else {
                    expect(p, t);
                }
            }
            case (_) {
            }
        }
        // FIXME: v += f(p) doesn't work at the moment.
        let T t = f(p);
        v += vec(t);
    }
    expect(p, ket);
    ret v;
}

state fn parse_lit(parser p) -> @ast.lit {
    alt (p.peek()) {
        case (token.LIT_INT(?i)) {
            p.bump();
            ret @ast.lit_int(i);
        }
        case (token.LIT_UINT(?u)) {
            p.bump();
            ret @ast.lit_uint(u);
        }
        case (token.LIT_CHAR(?c)) {
            p.bump();
            ret @ast.lit_char(c);
        }
        case (token.LIT_BOOL(?b)) {
            p.bump();
            ret @ast.lit_bool(b);
        }
    }
    p.err("expected literal");
    fail;
}



state fn parse_bottom_expr(parser p) -> @ast.expr {
    alt (p.peek()) {
        case (token.LPAREN) {
            p.bump();
            auto e = parse_expr(p);
            expect(p, token.RPAREN);
            ret e;
        }

        case (_) {
            ret @ast.expr_lit(parse_lit(p));
        }
    }
}


state fn parse_negation_expr(parser p) -> @ast.expr {
    alt (p.peek()) {

        case (token.NOT) {
            auto e = parse_negation_expr(p);
            ret @ast.expr_unary(ast.not, e);
        }

        case (token.TILDE) {
            auto e = parse_negation_expr(p);
            ret @ast.expr_unary(ast.bitnot, e);
        }

        case (_) {
            ret parse_bottom_expr(p);
        }
    }
}

state fn parse_expr(parser p) -> @ast.expr {
    ret parse_negation_expr(p);
}

state fn parse_stmt(parser p) -> @ast.stmt {
    alt (p.peek()) {
        case (token.LOG) {
            p.bump();
            auto e = parse_expr(p);
            expect(p, token.SEMI);
            ret @ast.stmt_log(e);
        }
    }
    p.err("expected statement");
    fail;
}

state fn parse_block(parser p) -> ast.block {
    auto f = parse_stmt;
    // FIXME: passing parse_stmt as an lval doesn't work at the moment.
    ret parse_seq[@ast.stmt](token.LBRACE,
                             token.RBRACE,
                             none[token.token],
                             f, p);
}

state fn parse_slot_ident_pair(parser p) ->
    rec(ast.slot slot, ast.ident ident) {
    auto s = parse_slot(p);
    auto i =  parse_ident(p);
    ret rec(slot=s, ident=i);
}

state fn parse_fn(parser p) -> tup(ast.ident, ast.item) {
    expect(p, token.FN);
    auto id = parse_ident(p);
    auto pf = parse_slot_ident_pair;
    auto inputs =
        // FIXME: passing parse_slot_ident_pair as an lval doesn't work at the
        // moment.
        parse_seq[rec(ast.slot slot, ast.ident ident)]
        (token.LPAREN,
         token.RPAREN,
         some(token.COMMA),
         pf, p);

    auto output;
    if (p.peek() == token.RARROW) {
        p.bump();
        output = rec(ty=parse_ty(p), mode=ast.val);
    } else {
        output = rec(ty=ast.ty_nil, mode=ast.val);
    }

    auto body = parse_block(p);

    let ast._fn f = rec(inputs = inputs,
                        output = output,
                        body = body);

    ret tup(id, ast.item_fn(@f));
}

state fn parse_mod(parser p) -> tup(ast.ident, ast.item) {
    expect(p, token.MOD);
    auto id = parse_ident(p);
    expect(p, token.LBRACE);
    let ast._mod m = new_str_hash[ast.item]();
    while (p.peek() != token.RBRACE) {
        auto i = parse_item(p);
        m.insert(i._0, i._1);
    }
    expect(p, token.RBRACE);
    ret tup(id, ast.item_mod(@m));
}

state fn parse_item(parser p) -> tup(ast.ident, ast.item) {
    alt (p.peek()) {
        case (token.FN) {
            ret parse_fn(p);
        }
        case (token.MOD) {
            ret parse_mod(p);
        }
    }
    p.err("expectied item");
    fail;
}

state fn parse_crate(parser p) -> ast.crate {
    let ast._mod m = new_str_hash[ast.item]();
    while (p.peek() != token.EOF) {
        auto i = parse_item(p);
        m.insert(i._0, i._1);
    }
    ret rec(module=m);
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
