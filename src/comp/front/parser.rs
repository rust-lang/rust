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
        case (token.LIT_STR(?s)) {
            p.bump();
            ret @ast.lit_str(s);
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

        case (token.TUP) {
            p.bump();
            auto pf = parse_expr;
            auto es = parse_seq[@ast.expr](token.LPAREN,
                                           token.RPAREN,
                                           some(token.COMMA),
                                           pf, p);
            ret @ast.expr_tup(es);
        }

        case (token.VEC) {
            p.bump();
            auto pf = parse_expr;
            auto es = parse_seq[@ast.expr](token.LPAREN,
                                           token.RPAREN,
                                           some(token.COMMA),
                                           pf, p);
            ret @ast.expr_vec(es);
        }

        case (token.REC) {
            p.bump();
            state fn parse_entry(parser p) ->
                tup(ast.ident, @ast.expr) {
                auto i = parse_ident(p);
                expect(p, token.EQ);
                auto e = parse_expr(p);
                ret tup(i, e);
            }
            auto pf = parse_entry;
            auto es =
                parse_seq[tup(ast.ident, @ast.expr)](token.LPAREN,
                                                     token.RPAREN,
                                                     some(token.COMMA),
                                                     pf, p);
            ret @ast.expr_rec(es);
        }

        case (token.IDENT(?i)) {
            p.bump();
            ret @ast.expr_ident(i);
        }

        case (_) {
            ret @ast.expr_lit(parse_lit(p));
        }
    }
}

state fn parse_path_expr(parser p) -> @ast.expr {
    auto e = parse_bottom_expr(p);
    while (true) {
        alt (p.peek()) {
            case (token.DOT) {
                p.bump();
                alt (p.peek()) {

                    case (token.IDENT(?i)) {
                        p.bump();
                        e = @ast.expr_field(e, i);
                    }

                    case (token.LPAREN) {
                        auto ix = parse_bottom_expr(p);
                        e = @ast.expr_index(e, ix);
                    }
                }
            }
            case (_) {
                ret e;
            }
        }
    }
    ret e;
}

state fn parse_prefix_expr(parser p) -> @ast.expr {
    alt (p.peek()) {

        case (token.NOT) {
            auto e = parse_prefix_expr(p);
            ret @ast.expr_unary(ast.not, e);
        }

        case (token.TILDE) {
            auto e = parse_prefix_expr(p);
            ret @ast.expr_unary(ast.bitnot, e);
        }

        case (token.BINOP(?b)) {
            alt (b) {

                case (token.MINUS) {
                    auto e = parse_prefix_expr(p);
                    ret @ast.expr_unary(ast.neg, e);
                }

                case (token.STAR) {
                    auto e = parse_prefix_expr(p);
                    ret @ast.expr_unary(ast.deref, e);
                }

                case (_) {
                    ret parse_path_expr(p);
                }
            }
        }

        case (token.AT) {
            p.bump();
            auto e = parse_prefix_expr(p);
            ret @ast.expr_unary(ast.box, e);
        }

        case (_) {
            ret parse_path_expr(p);
        }
    }
}

state fn parse_binops(parser p,
                      (state fn(parser) -> @ast.expr) sub,
                      vec[tup(token.binop, ast.binop)] ops)
    -> @ast.expr {
    auto e = sub(p);
    auto more = true;
    while (more) {
        more = false;
        auto t = p.peek();
        alt (t) {
            case (token.BINOP(?op)) {
                for (tup(token.binop, ast.binop) pair in ops) {
                    if (pair._0 == op) {
                        e = @ast.expr_binary(pair._1, e, sub(p));
                        more = true;
                        t = p.peek();
                    }
                }
            }
            case (_) {
            }
        }
    }
    ret e;
}

state fn parse_binary_exprs(parser p,
                            (state fn(parser) -> @ast.expr) sub,
                            vec[tup(token.token, ast.binop)] ops)
    -> @ast.expr {
    auto e = sub(p);
    auto more = true;
    while (more) {
        more = false;
        auto t = p.peek();
        for (tup(token.token, ast.binop) pair in ops) {
            if (pair._0 == t) {
                e = @ast.expr_binary(pair._1, e, sub(p));
                more = true;
                t = p.peek();
            }
        }
    }
    ret e;
}

state fn parse_factor_expr(parser p) -> @ast.expr {
    auto sub = parse_prefix_expr;
    ret parse_binops(p, sub, vec(tup(token.STAR, ast.mul),
                                 tup(token.SLASH, ast.div),
                                 tup(token.PERCENT, ast.rem)));
}

state fn parse_term_expr(parser p) -> @ast.expr {
    auto sub = parse_factor_expr;
    ret parse_binops(p, sub, vec(tup(token.PLUS, ast.add),
                                 tup(token.MINUS, ast.sub)));
}

state fn parse_shift_expr(parser p) -> @ast.expr {
    auto sub = parse_term_expr;
    ret parse_binops(p, sub, vec(tup(token.LSL, ast.lsl),
                                 tup(token.LSR, ast.lsr),
                                 tup(token.ASR, ast.asr)));
}

state fn parse_bitand_expr(parser p) -> @ast.expr {
    auto sub = parse_shift_expr;
    ret parse_binops(p, sub, vec(tup(token.AND, ast.bitand)));
}

state fn parse_bitxor_expr(parser p) -> @ast.expr {
    auto sub = parse_bitand_expr;
    ret parse_binops(p, sub, vec(tup(token.CARET, ast.bitxor)));
}

state fn parse_bitor_expr(parser p) -> @ast.expr {
    auto sub = parse_bitxor_expr;
    ret parse_binops(p, sub, vec(tup(token.OR, ast.bitor)));
}

state fn parse_cast_expr(parser p) -> @ast.expr {
    auto e = parse_bitor_expr(p);
    while (true) {
        alt (p.peek()) {
            case (token.AS) {
                p.bump();
                auto t = parse_ty(p);
                e = @ast.expr_cast(e, t);
            }

            case (_) {
                ret e;
            }
        }
    }
    ret e;
}

state fn parse_relational_expr(parser p) -> @ast.expr {
    auto sub = parse_cast_expr;
    ret parse_binary_exprs(p, sub, vec(tup(token.LT, ast.lt),
                                       tup(token.LE, ast.le),
                                       tup(token.GE, ast.ge),
                                       tup(token.GT, ast.gt)));
}


state fn parse_equality_expr(parser p) -> @ast.expr {
    auto sub = parse_relational_expr;
    ret parse_binary_exprs(p, sub, vec(tup(token.EQEQ, ast.eq),
                                       tup(token.NE, ast.ne)));
}

state fn parse_and_expr(parser p) -> @ast.expr {
    auto sub = parse_equality_expr;
    ret parse_binary_exprs(p, sub, vec(tup(token.ANDAND, ast.and)));
}

state fn parse_or_expr(parser p) -> @ast.expr {
    auto sub = parse_and_expr;
    ret parse_binary_exprs(p, sub, vec(tup(token.OROR, ast.or)));
}

state fn parse_expr(parser p) -> @ast.expr {
    ret parse_or_expr(p);
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
    auto i = parse_ident(p);
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
