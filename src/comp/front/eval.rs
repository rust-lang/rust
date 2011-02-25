import std._vec;
import std._str;
import std.option;
import std.option.some;
import std.option.none;
import std.map.hashmap;

import driver.session;
import ast.ident;
import front.parser.parser;
import front.parser.spanned;
import front.parser.new_parser;
import front.parser.parse_mod_items;
import util.common;
import util.common.filename;
import util.common.append;
import util.common.span;
import util.common.new_str_hash;


// Simple dynamic-typed value type for eval_expr.
tag val {
    val_bool(bool);
    val_int(int);
    val_str(str);
}

type env = vec[tup(ident, val)];

fn mk_env() -> env {
    let env e = vec();
    ret e;
}

fn val_is_bool(val v) -> bool {
    alt (v) {
        case (val_bool(_)) { ret true; }
        case (_) { }
    }
    ret false;
}

fn val_is_int(val v) -> bool {
    alt (v) {
        case (val_bool(_)) { ret true; }
        case (_) { }
    }
    ret false;
}

fn val_is_str(val v) -> bool {
    alt (v) {
        case (val_str(_)) { ret true; }
        case (_) { }
    }
    ret false;
}

fn val_as_bool(val v) -> bool {
    alt (v) {
        case (val_bool(?b)) { ret b; }
        case (_) { }
    }
    fail;
}

fn val_as_int(val v) -> int {
    alt (v) {
        case (val_int(?i)) { ret i; }
        case (_) { }
    }
    fail;
}

fn val_as_str(val v) -> str {
    alt (v) {
        case (val_str(?s)) { ret s; }
        case (_) { }
    }
    fail;
}

fn lookup(session.session sess, env e, span sp, ident i) -> val {
    for (tup(ident, val) pair in e) {
        if (_str.eq(i, pair._0)) {
            ret pair._1;
        }
    }
    sess.span_err(sp, "unknown variable: " + i);
    fail;
}

fn eval_lit(session.session sess, env e, span sp, @ast.lit lit) -> val {
    alt (lit.node) {
        case (ast.lit_bool(?b)) { ret val_bool(b); }
        case (ast.lit_int(?i)) { ret val_int(i); }
        case (ast.lit_str(?s)) { ret val_str(s); }
        case (_) {
            sess.span_err(sp, "evaluating unsupported literal");
        }
    }
    fail;
}

fn eval_expr(session.session sess, env e, @ast.expr x) -> val {
    alt (x.node) {
        case (ast.expr_path(?pth, _, _)) {
            if (_vec.len[ident](pth.node.idents) == 1u &&
                _vec.len[@ast.ty](pth.node.types) == 0u) {
                ret lookup(sess, e, x.span, pth.node.idents.(0));
            }
            sess.span_err(x.span, "evaluating structured path-name");
        }

        case (ast.expr_lit(?lit, _)) {
            ret eval_lit(sess, e, x.span, lit);
        }

        case (ast.expr_unary(?op, ?a, _)) {
            auto av = eval_expr(sess, e, a);
            alt (op) {
                case (ast.not) {
                    if (val_is_bool(av)) {
                        ret val_bool(!val_as_bool(av));
                    }
                    sess.span_err(x.span, "bad types in '!' expression");
                }
                case (_) {
                    sess.span_err(x.span, "evaluating unsupported unop");
                }
            }
        }

        case (ast.expr_binary(?op, ?a, ?b, _)) {
            auto av = eval_expr(sess, e, a);
            auto bv = eval_expr(sess, e, b);
            alt (op) {
                case (ast.add) {
                    if (val_is_int(av) && val_is_int(bv)) {
                        ret val_int(val_as_int(av) + val_as_int(bv));
                    }
                    if (val_is_str(av) && val_is_str(bv)) {
                        ret val_str(val_as_str(av) + val_as_str(bv));
                    }
                    sess.span_err(x.span, "bad types in '+' expression");
                }

                case (ast.sub) {
                    if (val_is_int(av) && val_is_int(bv)) {
                        ret val_int(val_as_int(av) - val_as_int(bv));
                    }
                    sess.span_err(x.span, "bad types in '-' expression");
                }

                case (ast.mul) {
                    if (val_is_int(av) && val_is_int(bv)) {
                        ret val_int(val_as_int(av) * val_as_int(bv));
                    }
                    sess.span_err(x.span, "bad types in '*' expression");
                }

                case (ast.div) {
                    if (val_is_int(av) && val_is_int(bv)) {
                        ret val_int(val_as_int(av) / val_as_int(bv));
                    }
                    sess.span_err(x.span, "bad types in '/' expression");
                }

                case (ast.rem) {
                    if (val_is_int(av) && val_is_int(bv)) {
                        ret val_int(val_as_int(av) % val_as_int(bv));
                    }
                    sess.span_err(x.span, "bad types in '%' expression");
                }

                case (ast.and) {
                    if (val_is_bool(av) && val_is_bool(bv)) {
                        ret val_bool(val_as_bool(av) && val_as_bool(bv));
                    }
                    sess.span_err(x.span, "bad types in '&&' expression");
                }

                case (ast.or) {
                    if (val_is_bool(av) && val_is_bool(bv)) {
                        ret val_bool(val_as_bool(av) || val_as_bool(bv));
                    }
                    sess.span_err(x.span, "bad types in '||' expression");
                }

                case (ast.eq) {
                    ret val_bool(val_eq(sess, x.span, av, bv));
                }

                case (ast.ne) {
                    ret val_bool(! val_eq(sess, x.span, av, bv));
                }

                case (_) {
                    sess.span_err(x.span, "evaluating unsupported binop");
                }
            }
        }
        case (_) {
            sess.span_err(x.span, "evaluating unsupported expression");
        }
    }
    fail;
}

fn val_eq(session.session sess, span sp, val av, val bv) -> bool {
    if (val_is_bool(av) && val_is_bool(bv)) {
        ret val_as_bool(av) == val_as_bool(bv);
    }
    if (val_is_int(av) && val_is_int(bv)) {
        ret val_as_int(av) == val_as_int(bv);
    }
    if (val_is_str(av) && val_is_str(bv)) {
        ret _str.eq(val_as_str(av),
                    val_as_str(bv));
    }
    sess.span_err(sp, "bad types in comparison");
    fail;
}

impure fn eval_crate_directives(parser p,
                                env e,
                                vec[@ast.crate_directive] cdirs,
                                str prefix,
                                &mutable vec[@ast.view_item] view_items,
                                &mutable vec[@ast.item] items,
                                hashmap[ast.ident,
                                        ast.mod_index_entry] index) {

    for (@ast.crate_directive sub_cdir in cdirs) {
        eval_crate_directive(p, e, sub_cdir, prefix,
                             view_items, items, index);
    }
}


impure fn eval_crate_directives_to_mod(parser p,
                                       env e,
                                       vec[@ast.crate_directive] cdirs,
                                       str prefix) -> ast._mod {
    let vec[@ast.view_item] view_items = vec();
    let vec[@ast.item] items = vec();
    auto index = new_str_hash[ast.mod_index_entry]();

    eval_crate_directives(p, e, cdirs, prefix,
                          view_items, items, index);

    ret rec(view_items=view_items, items=items, index=index);
}


impure fn eval_crate_directive_block(parser p,
                                     env e,
                                     &ast.block blk,
                                     str prefix,
                                     &mutable vec[@ast.view_item] view_items,
                                     &mutable vec[@ast.item] items,
                                     hashmap[ast.ident,
                                             ast.mod_index_entry] index) {

    for (@ast.stmt s in blk.node.stmts) {
        alt (s.node) {
            case (ast.stmt_crate_directive(?cdir)) {
                eval_crate_directive(p, e, cdir, prefix,
                                     view_items, items, index);
            }
            case (_) {
                auto sess = p.get_session();
                sess.span_err(s.span,
                              "unsupported stmt in crate-directive block");
            }
        }
    }
}

impure fn eval_crate_directive_expr(parser p,
                                    env e,
                                    @ast.expr x,
                                    str prefix,
                                    &mutable vec[@ast.view_item] view_items,
                                    &mutable vec[@ast.item] items,
                                    hashmap[ast.ident,
                                            ast.mod_index_entry] index) {
    auto sess = p.get_session();

    alt (x.node) {

        case (ast.expr_if(?cond, ?thn, ?elifs, ?elopt, _)) {
            auto cv = eval_expr(sess, e, cond);
            if (!val_is_bool(cv)) {
                sess.span_err(x.span, "bad cond type in 'if'");
            }

            if (val_as_bool(cv)) {
                ret eval_crate_directive_block(p, e, thn, prefix,
                                               view_items, items,
                                               index);
            }

            for (tup(@ast.expr, ast.block) elif in elifs) {
                auto cv = eval_expr(sess, e, elif._0);
                if (!val_is_bool(cv)) {
                    sess.span_err(x.span, "bad cond type in 'else if'");
                }

                if (val_as_bool(cv)) {
                    ret eval_crate_directive_block(p, e, elif._1, prefix,
                                                   view_items, items,
                                                   index);
                }
            }

            alt (elopt) {
                case (some[ast.block](?els)) {
                    ret eval_crate_directive_block(p, e, els, prefix,
                                                   view_items, items,
                                                   index);
                }
                case (_) {
                    // Absent-else is ok.
                }
            }
        }

        case (ast.expr_alt(?v, ?arms, _)) {
            auto vv = eval_expr(sess, e, v);
            for (ast.arm arm in arms) {
                alt (arm.pat.node) {
                    case (ast.pat_lit(?lit, _)) {
                        auto pv = eval_lit(sess, e,
                                           arm.pat.span, lit);
                        if (val_eq(sess, arm.pat.span, vv, pv)) {
                            ret eval_crate_directive_block
                                (p, e, arm.block, prefix,
                                 view_items, items, index);
                        }
                    }
                    case (ast.pat_wild(_)) {
                        ret eval_crate_directive_block
                            (p, e, arm.block, prefix,
                             view_items, items, index);
                    }
                    case (_) {
                        sess.span_err(arm.pat.span,
                                      "bad pattern type in 'alt'");
                    }
                }
            }
            sess.span_err(x.span, "no cases matched in 'alt'");
        }

        case (_) {
            sess.span_err(x.span, "unsupported expr type");
        }
    }
}

impure fn eval_crate_directive(parser p,
                               env e,
                               @ast.crate_directive cdir,
                               str prefix,
                               &mutable vec[@ast.view_item] view_items,
                               &mutable vec[@ast.item] items,
                               hashmap[ast.ident,
                                       ast.mod_index_entry] index) {
    alt (cdir.node) {

        case (ast.cdir_let(?id, ?x, ?cdirs)) {
            auto v = eval_expr(p.get_session(), e, x);
            auto e0 = vec(tup(id, v)) + e;
            eval_crate_directives(p, e0, cdirs, prefix,
                                  view_items, items, index);
        }

        case (ast.cdir_expr(?x)) {
            eval_crate_directive_expr(p, e, x, prefix,
                                      view_items, items, index);
        }

        case (ast.cdir_src_mod(?id, ?file_opt)) {

            auto file_path = id + ".rs";
            alt (file_opt) {
                case (some[filename](?f)) {
                    file_path = f;
                }
                case (none[filename]) {}
            }

            auto full_path = prefix + std.os.path_sep() + file_path;

            auto p0 = new_parser(p.get_session(), 0, full_path);
            auto m0 = parse_mod_items(p0, token.EOF);
            auto im = ast.item_mod(id, m0, p.next_def_id());
            auto i = @spanned(cdir.span, cdir.span, im);
            ast.index_item(index, i);
            append[@ast.item](items, i);
        }

        case (ast.cdir_dir_mod(?id, ?dir_opt, ?cdirs)) {

            auto path = id;
            alt (dir_opt) {
                case (some[filename](?d)) {
                    path = d;
                }
                case (none[filename]) {}
            }

            auto full_path = prefix + std.os.path_sep() + path;
            auto m0 = eval_crate_directives_to_mod(p, e, cdirs, full_path);
            auto im = ast.item_mod(id, m0, p.next_def_id());
            auto i = @spanned(cdir.span, cdir.span, im);
            ast.index_item(index, i);
            append[@ast.item](items, i);
        }

        case (ast.cdir_view_item(?vi)) {
            append[@ast.view_item](view_items, vi);
            ast.index_view_item(index, vi);
        }

        case (ast.cdir_meta(?mi)) {}
        case (ast.cdir_syntax(?pth)) {}
        case (ast.cdir_auth(?pth, ?eff)) {}
    }
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
