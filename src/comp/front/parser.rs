import std._io;
import std.util.option;
import std.util.some;
import std.util.none;
import std.map.hashmap;

import driver.session;
import util.common;
import util.common.append;
import util.common.span;
import util.common.new_str_hash;

state type parser =
    state obj {
          fn peek() -> token.token;
          io fn bump();
          io fn err(str s);
          fn get_session() -> session.session;
          fn get_span() -> common.span;
          fn next_def_id() -> ast.def_id;
    };

io fn new_parser(session.session sess,
                 ast.crate_num crate, str path) -> parser {
    state obj stdio_parser(session.session sess,
                           mutable token.token tok,
                           mutable common.pos lo,
                           mutable common.pos hi,
                           mutable ast.def_num def,
                           ast.crate_num crate,
                           lexer.reader rdr)
        {
            fn peek() -> token.token {
                // log token.to_str(tok);
                ret tok;
            }

            io fn bump() {
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

            fn next_def_id() -> ast.def_id {
                def += 1;
                ret tup(crate, def);
            }
        }
    auto srdr = _io.new_stdio_reader(path);
    auto rdr = lexer.new_reader(srdr, path);
    auto npos = rdr.get_curr_pos();
    ret stdio_parser(sess, lexer.next_token(rdr),
                     npos, npos, 0, crate, rdr);
}

io fn expect(parser p, token.token t) {
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

fn spanned[T](&span lo, &span hi, &T node) -> ast.spanned[T] {
    ret rec(node=node, span=rec(filename=lo.filename,
                                lo=lo.lo,
                                hi=hi.hi));
}

io fn parse_ident(parser p) -> ast.ident {
    alt (p.peek()) {
        case (token.IDENT(?i)) { p.bump(); ret i; }
        case (_) {
            p.err("expecting ident");
            fail;
        }
    }
}

io fn parse_possibly_mutable_ty(parser p) -> tup(bool, @ast.ty) {
    auto mut;
    if (p.peek() == token.MUTABLE) {
        p.bump();
        mut = true;
    } else {
        mut = false;
    }

    ret tup(mut, parse_ty(p));
}

io fn parse_ty(parser p) -> @ast.ty {
    auto lo = p.get_span();
    let ast.ty_ t;
    alt (p.peek()) {
        case (token.INT) { p.bump(); t = ast.ty_int; }
        case (token.UINT) { p.bump(); t = ast.ty_int; }
        case (token.STR) { p.bump(); t = ast.ty_str; }
        case (token.CHAR) { p.bump(); t = ast.ty_char; }
        case (token.MACH(?tm)) { p.bump(); t = ast.ty_machine(tm); }

        case (token.AT) { p.bump(); t = ast.ty_box(parse_ty(p)); }

        case (token.VEC) {
            p.bump();
            expect(p, token.LBRACKET);
            t = ast.ty_vec(parse_ty(p));
            expect(p, token.RBRACKET);
        }

        case (token.TUP) {
            p.bump();
            auto f = parse_possibly_mutable_ty; // FIXME: trans_const_lval bug
            auto elems = parse_seq[tup(bool, @ast.ty)](token.LPAREN,
                token.RPAREN, some(token.COMMA), f, p);
            t = ast.ty_tup(elems.node);
        }

        case (_) {
            p.err("expecting type");
            t = ast.ty_nil;
            fail;
        }
    }
    ret @spanned(lo, lo, t);
}

io fn parse_arg(parser p) -> ast.arg {
    let ast.mode m = ast.val;
    if (p.peek() == token.BINOP(token.AND)) {
        m = ast.alias;
        p.bump();
    }
    let @ast.ty t = parse_ty(p);
    let ast.ident i = parse_ident(p);
    ret rec(mode=m, ty=t, ident=i, id=p.next_def_id());
}

io fn parse_seq[T](token.token bra,
                      token.token ket,
                      option[token.token] sep,
                      (io fn(parser) -> T) f,
                      parser p) -> util.common.spanned[vec[T]] {
    let bool first = true;
    auto lo = p.get_span();
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
    auto hi = p.get_span();
    expect(p, ket);
    ret spanned(lo, hi, v);
}

io fn parse_lit(parser p) -> option[ast.lit] {
    auto lo = p.get_span();
    let ast.lit_ lit;
    alt (p.peek()) {
        case (token.LIT_INT(?i)) {
            p.bump();
            lit = ast.lit_int(i);
        }
        case (token.LIT_UINT(?u)) {
            p.bump();
            lit = ast.lit_uint(u);
        }
        case (token.LIT_CHAR(?c)) {
            p.bump();
            lit = ast.lit_char(c);
        }
        case (token.LIT_BOOL(?b)) {
            p.bump();
            lit = ast.lit_bool(b);
        }
        case (token.LIT_STR(?s)) {
            p.bump();
            lit = ast.lit_str(s);
        }
        case (_) {
            lit = ast.lit_nil;  // FIXME: typestate bug requires this
            ret none[ast.lit];
        }
    }
    ret some(spanned(lo, lo, lit));
}

io fn parse_name(parser p, ast.ident id) -> ast.name {

    auto lo = p.get_span();

    p.bump();

    let vec[@ast.ty] v = vec();
    let util.common.spanned[vec[@ast.ty]] tys = rec(node=v, span=lo);

    alt (p.peek()) {
        case (token.LBRACKET) {
            auto pf = parse_ty;
            tys = parse_seq[@ast.ty](token.LBRACKET,
                                     token.RBRACKET,
                                     some(token.COMMA),
                                     pf, p);
        }
        case (_) {
        }
    }
    ret spanned(lo, tys.span, rec(ident=id, types=tys.node));
}

io fn parse_possibly_mutable_expr(parser p) -> tup(bool, @ast.expr) {
    auto mut;
    if (p.peek() == token.MUTABLE) {
        p.bump();
        mut = true;
    } else {
        mut = false;
    }

    ret tup(mut, parse_expr(p));
}

io fn parse_bottom_expr(parser p) -> @ast.expr {

    auto lo = p.get_span();
    auto hi = lo;

    // FIXME: can only remove this sort of thing when both typestate and
    // alt-exhaustive-match checking are co-operating.
    auto lit = @spanned(lo, lo, ast.lit_nil);
    let ast.expr_ ex = ast.expr_lit(lit, none[@ast.ty]);

    alt (p.peek()) {

        case (token.IDENT(?i)) {
            auto n = parse_name(p, i);
            hi = n.span;
            ex = ast.expr_name(n, none[ast.def], none[@ast.ty]);
        }

        case (token.LPAREN) {
            p.bump();
            auto e = parse_expr(p);
            hi = p.get_span();
            expect(p, token.RPAREN);
            ret @spanned(lo, hi, e.node);
        }

        case (token.TUP) {
            p.bump();
            auto pf = parse_possibly_mutable_expr;
            auto es = parse_seq[tup(bool, @ast.expr)](token.LPAREN,
                                                      token.RPAREN,
                                                      some(token.COMMA),
                                                      pf, p);
            hi = es.span;
            ex = ast.expr_tup(es.node, none[@ast.ty]);
        }

        case (token.VEC) {
            p.bump();
            auto pf = parse_expr;
            auto es = parse_seq[@ast.expr](token.LPAREN,
                                           token.RPAREN,
                                           some(token.COMMA),
                                           pf, p);
            hi = es.span;
            ex = ast.expr_vec(es.node, none[@ast.ty]);
        }

        case (token.REC) {
            p.bump();
            io fn parse_entry(parser p) ->
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
            hi = es.span;
            ex = ast.expr_rec(es.node, none[@ast.ty]);
        }

        case (_) {
            alt (parse_lit(p)) {
                case (some[ast.lit](?lit)) {
                    hi = lit.span;
                    ex = ast.expr_lit(@lit, none[@ast.ty]);
                }
                case (none[ast.lit]) {
                    p.err("expecting expression");
                }
            }
        }
    }

    ret @spanned(lo, hi, ex);
}

io fn parse_path_expr(parser p) -> @ast.expr {
    auto lo = p.get_span();
    auto e = parse_bottom_expr(p);
    auto hi = e.span;
    while (true) {
        alt (p.peek()) {
            case (token.DOT) {
                p.bump();
                alt (p.peek()) {

                    case (token.IDENT(?i)) {
                        hi = p.get_span();
                        p.bump();
                        auto e_ = ast.expr_field(e, i, none[@ast.ty]);
                        e = @spanned(lo, hi, e_);
                    }

                    case (token.LPAREN) {
                        auto ix = parse_bottom_expr(p);
                        hi = ix.span;
                        auto e_ = ast.expr_index(e, ix, none[@ast.ty]);
                        e = @spanned(lo, hi, e_);
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

io fn parse_prefix_expr(parser p) -> @ast.expr {

    auto lo = p.get_span();
    auto hi = lo;

    // FIXME: can only remove this sort of thing when both typestate and
    // alt-exhaustive-match checking are co-operating.
    auto lit = @spanned(lo, lo, ast.lit_nil);
    let ast.expr_ ex = ast.expr_lit(lit, none[@ast.ty]);

    alt (p.peek()) {

        case (token.NOT) {
            p.bump();
            auto e = parse_prefix_expr(p);
            hi = e.span;
            ex = ast.expr_unary(ast.not, e, none[@ast.ty]);
        }

        case (token.TILDE) {
            p.bump();
            auto e = parse_prefix_expr(p);
            hi = e.span;
            ex = ast.expr_unary(ast.bitnot, e, none[@ast.ty]);
        }

        case (token.BINOP(?b)) {
            alt (b) {
                case (token.MINUS) {
                    p.bump();
                    auto e = parse_prefix_expr(p);
                    hi = e.span;
                    ex = ast.expr_unary(ast.neg, e, none[@ast.ty]);
                }

                case (token.STAR) {
                    p.bump();
                    auto e = parse_prefix_expr(p);
                    hi = e.span;
                    ex = ast.expr_unary(ast.deref, e, none[@ast.ty]);
                }

                case (_) {
                    ret parse_path_expr(p);
                }
            }
        }

        case (token.AT) {
            p.bump();
            auto e = parse_prefix_expr(p);
            hi = e.span;
            ex = ast.expr_unary(ast.box, e, none[@ast.ty]);
        }

        case (_) {
            ret parse_path_expr(p);
        }
    }
    ret @spanned(lo, hi, ex);
}

io fn parse_binops(parser p,
                   (io fn(parser) -> @ast.expr) sub,
                   vec[tup(token.binop, ast.binop)] ops)
    -> @ast.expr {
    auto lo = p.get_span();
    auto hi = lo;
    auto e = sub(p);
    auto more = true;
    while (more) {
        more = false;
        for (tup(token.binop, ast.binop) pair in ops) {
            alt (p.peek()) {
                case (token.BINOP(?op)) {
                    if (pair._0 == op) {
                        p.bump();
                        auto rhs = sub(p);
                        hi = rhs.span;
                        auto exp = ast.expr_binary(pair._1, e, rhs,
                                                   none[@ast.ty]);
                        e = @spanned(lo, hi, exp);
                        more = true;
                    }
                }
            }
        }
    }
    ret e;
}

io fn parse_binary_exprs(parser p,
                            (io fn(parser) -> @ast.expr) sub,
                            vec[tup(token.token, ast.binop)] ops)
    -> @ast.expr {
    auto lo = p.get_span();
    auto hi = lo;
    auto e = sub(p);
    auto more = true;
    while (more) {
        more = false;
        for (tup(token.token, ast.binop) pair in ops) {
            if (pair._0 == p.peek()) {
                p.bump();
                auto rhs = sub(p);
                hi = rhs.span;
                auto exp = ast.expr_binary(pair._1, e, rhs, none[@ast.ty]);
                e = @spanned(lo, hi, exp);
                more = true;
            }
        }
    }
    ret e;
}

io fn parse_factor_expr(parser p) -> @ast.expr {
    auto sub = parse_prefix_expr;
    ret parse_binops(p, sub, vec(tup(token.STAR, ast.mul),
                                 tup(token.SLASH, ast.div),
                                 tup(token.PERCENT, ast.rem)));
}

io fn parse_term_expr(parser p) -> @ast.expr {
    auto sub = parse_factor_expr;
    ret parse_binops(p, sub, vec(tup(token.PLUS, ast.add),
                                 tup(token.MINUS, ast.sub)));
}

io fn parse_shift_expr(parser p) -> @ast.expr {
    auto sub = parse_term_expr;
    ret parse_binops(p, sub, vec(tup(token.LSL, ast.lsl),
                                 tup(token.LSR, ast.lsr),
                                 tup(token.ASR, ast.asr)));
}

io fn parse_bitand_expr(parser p) -> @ast.expr {
    auto sub = parse_shift_expr;
    ret parse_binops(p, sub, vec(tup(token.AND, ast.bitand)));
}

io fn parse_bitxor_expr(parser p) -> @ast.expr {
    auto sub = parse_bitand_expr;
    ret parse_binops(p, sub, vec(tup(token.CARET, ast.bitxor)));
}

io fn parse_bitor_expr(parser p) -> @ast.expr {
    auto sub = parse_bitxor_expr;
    ret parse_binops(p, sub, vec(tup(token.OR, ast.bitor)));
}

io fn parse_cast_expr(parser p) -> @ast.expr {
    auto lo = p.get_span();
    auto e = parse_bitor_expr(p);
    auto hi = e.span;
    while (true) {
        alt (p.peek()) {
            case (token.AS) {
                p.bump();
                auto t = parse_ty(p);
                hi = t.span;
                e = @spanned(lo, hi, ast.expr_cast(e, t));
            }

            case (_) {
                ret e;
            }
        }
    }
    ret e;
}

io fn parse_relational_expr(parser p) -> @ast.expr {
    auto sub = parse_cast_expr;
    ret parse_binary_exprs(p, sub, vec(tup(token.LT, ast.lt),
                                       tup(token.LE, ast.le),
                                       tup(token.GE, ast.ge),
                                       tup(token.GT, ast.gt)));
}


io fn parse_equality_expr(parser p) -> @ast.expr {
    auto sub = parse_relational_expr;
    ret parse_binary_exprs(p, sub, vec(tup(token.EQEQ, ast.eq),
                                       tup(token.NE, ast.ne)));
}

io fn parse_and_expr(parser p) -> @ast.expr {
    auto sub = parse_equality_expr;
    ret parse_binary_exprs(p, sub, vec(tup(token.ANDAND, ast.and)));
}

io fn parse_or_expr(parser p) -> @ast.expr {
    auto sub = parse_and_expr;
    ret parse_binary_exprs(p, sub, vec(tup(token.OROR, ast.or)));
}

io fn parse_assign_expr(parser p) -> @ast.expr {
    auto lo = p.get_span();
    auto lhs = parse_or_expr(p);
    alt (p.peek()) {
        case (token.EQ) {
            p.bump();
            auto rhs = parse_expr(p);
            ret @spanned(lo, rhs.span,
                         ast.expr_assign(lhs, rhs, none[@ast.ty]));
        }
    }
    ret lhs;
}

io fn parse_if_expr(parser p) -> @ast.expr {
    auto lo = p.get_span();
    auto hi = lo;

    expect(p, token.IF);
    expect(p, token.LPAREN);
    auto cond = parse_expr(p);
    expect(p, token.RPAREN);
    auto thn = parse_block(p);
    let option[ast.block] els = none[ast.block];
    hi = thn.span;
    alt (p.peek()) {
        case (token.ELSE) {
            p.bump();
            auto eblk = parse_block(p);
            els = some(eblk);
            hi = eblk.span;
        }
    }
    ret @spanned(lo, hi, ast.expr_if(cond, thn, els, none[@ast.ty]));
}

io fn parse_expr(parser p) -> @ast.expr {
    alt (p.peek()) {
        case (token.LBRACE) {
            auto blk = parse_block(p);
            ret @spanned(blk.span, blk.span,
                         ast.expr_block(blk, none[@ast.ty]));
        }
        case (token.IF) {
            ret parse_if_expr(p);
        }
        case (_) {
            ret parse_assign_expr(p);
        }

    }
}

io fn parse_initializer(parser p) -> option[@ast.expr] {
    if (p.peek() == token.EQ) {
        p.bump();
        ret some(parse_expr(p));
    }

    ret none[@ast.expr];
}

io fn parse_let(parser p) -> @ast.decl {
    auto lo = p.get_span();

    expect(p, token.LET);
    auto ty = parse_ty(p);
    auto ident = parse_ident(p);
    auto init = parse_initializer(p);

    auto hi = p.get_span();
    expect(p, token.SEMI);

    let ast.local local = rec(ty = some(ty),
                              infer = false,
                              ident = ident,
                              init = init,
                              id = p.next_def_id());

    ret @spanned(lo, hi, ast.decl_local(@local));
}

io fn parse_auto(parser p) -> @ast.decl {
    auto lo = p.get_span();

    expect(p, token.AUTO);
    auto ident = parse_ident(p);
    auto init = parse_initializer(p);

    auto hi = p.get_span();
    expect(p, token.SEMI);

    let ast.local local = rec(ty = none[@ast.ty],
                              infer = true,
                              ident = ident,
                              init = init,
                              id = p.next_def_id());

    ret @spanned(lo, hi, ast.decl_local(@local));
}

io fn parse_stmt(parser p) -> @ast.stmt {
    auto lo = p.get_span();
    alt (p.peek()) {

        case (token.LOG) {
            p.bump();
            auto e = parse_expr(p);
            auto hi = p.get_span();
            expect(p, token.SEMI);
            ret @spanned(lo, hi, ast.stmt_log(e));
        }

        case (token.LET) {
            auto decl = parse_let(p);
            auto hi = p.get_span();
            ret @spanned(lo, hi, ast.stmt_decl(decl));
        }

        case (token.AUTO) {
            auto decl = parse_auto(p);
            auto hi = p.get_span();
            ret @spanned(lo, hi, ast.stmt_decl(decl));
        }

        // Handle the (few) block-expr stmts first.

        case (token.IF) {
            auto e = parse_expr(p);
            ret @spanned(lo, e.span, ast.stmt_expr(e));
        }

        case (token.LBRACE) {
            auto e = parse_expr(p);
            ret @spanned(lo, e.span, ast.stmt_expr(e));
        }


        // Remainder are line-expr stmts.

        case (_) {
            auto e = parse_expr(p);
            auto hi = p.get_span();
            expect(p, token.SEMI);
            ret @spanned(lo, hi, ast.stmt_expr(e));
        }
    }
    p.err("expected statement");
    fail;
}

io fn parse_block(parser p) -> ast.block {
    auto f = parse_stmt;
    // FIXME: passing parse_stmt as an lval doesn't work at the moment.
    auto stmts = parse_seq[@ast.stmt](token.LBRACE,
                                      token.RBRACE,
                                      none[token.token],
                                      f, p);
    auto index = new_str_hash[uint]();
    auto u = 0u;
    for (@ast.stmt s in stmts.node) {
        // FIXME: typestate bug requires we do this up top, not
        // down below loop. Sigh.
        u += 1u;
        alt (s.node) {
            case (ast.stmt_decl(?d)) {
                alt (d.node) {
                    case (ast.decl_local(?loc)) {
                        index.insert(loc.ident, u-1u);
                    }
                    case (ast.decl_item(?it)) {
                        alt (it.node) {
                            case (ast.item_fn(?i, _, _)) {
                                index.insert(i, u-1u);
                            }
                            case (ast.item_mod(?i, _, _)) {
                                index.insert(i, u-1u);
                            }
                            case (ast.item_ty(?i, _, _)) {
                                index.insert(i, u-1u);
                            }
                        }
                    }
                }
            }
        }
    }
    let ast.block_ b = rec(stmts=stmts.node, index=index);
    ret spanned(stmts.span, stmts.span, b);
}

io fn parse_fn(parser p) -> tup(ast.ident, @ast.item) {
    auto lo = p.get_span();
    expect(p, token.FN);
    auto id = parse_ident(p);
    auto pf = parse_arg;
    let util.common.spanned[vec[ast.arg]] inputs =
        // FIXME: passing parse_arg as an lval doesn't work at the
        // moment.
        parse_seq[ast.arg]
        (token.LPAREN,
         token.RPAREN,
         some(token.COMMA),
         pf, p);

    let @ast.ty output;
    if (p.peek() == token.RARROW) {
        p.bump();
        output = parse_ty(p);
    } else {
        output = @spanned(lo, inputs.span, ast.ty_nil);
    }

    auto body = parse_block(p);

    let ast._fn f = rec(inputs = inputs.node,
                        output = output,
                        body = body);

    auto item = ast.item_fn(id, f, p.next_def_id());
    ret tup(id, @spanned(lo, body.span, item));
}

io fn parse_mod_items(parser p, token.token term) -> ast._mod {
   let vec[@ast.item] items = vec();
    let hashmap[ast.ident,uint] index = new_str_hash[uint]();
    let uint u = 0u;
    while (p.peek() != term) {
        auto pair = parse_item(p);
        append[@ast.item](items, pair._1);
        index.insert(pair._0, u);
        u += 1u;
    }
    ret rec(items=items, index=index);
 }

io fn parse_mod(parser p) -> tup(ast.ident, @ast.item) {
    auto lo = p.get_span();
    expect(p, token.MOD);
    auto id = parse_ident(p);
    expect(p, token.LBRACE);
    auto m = parse_mod_items(p, token.RBRACE);
    auto hi = p.get_span();
    expect(p, token.RBRACE);
    auto item = ast.item_mod(id, m, p.next_def_id());
    ret tup(id, @spanned(lo, hi, item));
}

io fn parse_item(parser p) -> tup(ast.ident, @ast.item) {
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

io fn parse_crate(parser p) -> @ast.crate {
    auto lo = p.get_span();
    auto hi = lo;
    auto m = parse_mod_items(p, token.EOF);
    ret @spanned(lo, hi, rec(module=m));
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
