
import std::vec;
import std::str;
import std::option;
import std::option::some;
import std::option::none;
import std::map::hashmap;
import driver::session;
import ast::ident;
import front::parser::parser;
import front::parser::spanned;
import front::parser::new_parser;
import front::parser::parse_inner_attrs_and_next;
import front::parser::parse_mod_items;
import util::common;
import util::common::filename;
import util::common::span;
import util::common::new_str_hash;


// Simple dynamic-typed value type for eval_expr.
tag val { val_bool(bool); val_int(int); val_str(str); }

tag eval_mode { mode_depend; mode_parse; }

type env = vec[tup(ident, val)];

type ctx =
    @rec(parser p,
         eval_mode mode,
         mutable vec[str] deps,
         session::session sess,
         mutable uint chpos,
         mutable int next_id);

fn mk_env() -> env { ret []; }

fn val_is_bool(val v) -> bool {
    alt (v) { case (val_bool(_)) { true } case (_) { false } }
}

fn val_is_int(val v) -> bool {
    alt (v) { case (val_int(_)) { true } case (_) { false } }
}

fn val_is_str(val v) -> bool {
    alt (v) { case (val_str(_)) { true } case (_) { false } }
}

fn val_as_bool(val v) -> bool {
    alt (v) { case (val_bool(?b)) { b } case (_) { fail } }
}

fn val_as_int(val v) -> int {
    alt (v) { case (val_int(?i)) { i } case (_) { fail } }
}

fn val_as_str(val v) -> str {
    alt (v) { case (val_str(?s)) { s } case (_) { fail } }
}

fn lookup(session::session sess, env e, span sp, ident i) -> val {
    for (tup(ident, val) pair in e) {
        if (str::eq(i, pair._0)) { ret pair._1; }
    }
    sess.span_fatal(sp, "unknown variable: " + i)
}

fn eval_lit(ctx cx, span sp, @ast::lit lit) -> val {
    alt (lit.node) {
        case (ast::lit_bool(?b)) { val_bool(b) }
        case (ast::lit_int(?i)) { val_int(i) }
        case (ast::lit_str(?s, _)) { val_str(s) }
        case (_) { cx.sess.span_fatal(sp, "evaluating unsupported literal") }
    }
}

fn eval_expr(ctx cx, env e, @ast::expr x) -> val {
    alt (x.node) {
        case (ast::expr_path(?pth)) {
            if (vec::len[ident](pth.node.idents) == 1u &&
                    vec::len[@ast::ty](pth.node.types) == 0u) {
                ret lookup(cx.sess, e, x.span, pth.node.idents.(0));
            }
            cx.sess.span_fatal(x.span, "evaluating structured path-name");
        }
        case (ast::expr_lit(?lit)) { ret eval_lit(cx, x.span, lit); }
        case (ast::expr_unary(?op, ?a)) {
            auto av = eval_expr(cx, e, a);
            alt (op) {
                case (ast::not) {
                    if (val_is_bool(av)) { ret val_bool(!val_as_bool(av)); }
                    cx.sess.span_fatal(x.span, "bad types in '!' expression");
                }
                case (_) {
                    cx.sess.span_fatal(x.span, "evaluating unsupported unop");
                }
            }
        }
        case (ast::expr_binary(?op, ?a, ?b)) {
            auto av = eval_expr(cx, e, a);
            auto bv = eval_expr(cx, e, b);
            alt (op) {
                case (ast::add) {
                    if (val_is_int(av) && val_is_int(bv)) {
                        ret val_int(val_as_int(av) + val_as_int(bv));
                    }
                    if (val_is_str(av) && val_is_str(bv)) {
                        ret val_str(val_as_str(av) + val_as_str(bv));
                    }
                    cx.sess.span_fatal(x.span, "bad types in '+' expression");
                }
                case (ast::sub) {
                    if (val_is_int(av) && val_is_int(bv)) {
                        ret val_int(val_as_int(av) - val_as_int(bv));
                    }
                    cx.sess.span_fatal(x.span, "bad types in '-' expression");
                }
                case (ast::mul) {
                    if (val_is_int(av) && val_is_int(bv)) {
                        ret val_int(val_as_int(av) * val_as_int(bv));
                    }
                    cx.sess.span_fatal(x.span, "bad types in '*' expression");
                }
                case (ast::div) {
                    if (val_is_int(av) && val_is_int(bv)) {
                        ret val_int(val_as_int(av) / val_as_int(bv));
                    }
                    cx.sess.span_fatal(x.span, "bad types in '/' expression");
                }
                case (ast::rem) {
                    if (val_is_int(av) && val_is_int(bv)) {
                        ret val_int(val_as_int(av) % val_as_int(bv));
                    }
                    cx.sess.span_fatal(x.span, "bad types in '%' expression");
                }
                case (ast::and) {
                    if (val_is_bool(av) && val_is_bool(bv)) {
                        ret val_bool(val_as_bool(av) && val_as_bool(bv));
                    }
                    cx.sess.span_fatal(x.span,
                                       "bad types in '&&' expression");
                }
                case (ast::or) {
                    if (val_is_bool(av) && val_is_bool(bv)) {
                        ret val_bool(val_as_bool(av) || val_as_bool(bv));
                    }
                    cx.sess.span_fatal(x.span,
                                       "bad types in '||' expression");
                }
                case (ast::eq) {
                    ret val_bool(val_eq(cx.sess, x.span, av, bv));
                }
                case (ast::ne) {
                    ret val_bool(!val_eq(cx.sess, x.span, av, bv));
                }
                case (_) {
                    cx.sess.span_fatal(x.span,
                                       "evaluating unsupported binop");
                }
            }
        }
        case (_) {
            cx.sess.span_fatal(x.span, "evaluating unsupported expression");
        }
    }
    fail;
}

fn val_eq(session::session sess, span sp, val av, val bv) -> bool {
    if (val_is_bool(av) && val_is_bool(bv)) {
        val_as_bool(av) == val_as_bool(bv)
    } else if (val_is_int(av) && val_is_int(bv)) {
        val_as_int(av) == val_as_int(bv)
    } else if (val_is_str(av) && val_is_str(bv)) {
        str::eq(val_as_str(av), val_as_str(bv))
    } else { sess.span_fatal(sp, "bad types in comparison") }
}

fn eval_crate_directives(ctx cx, env e, vec[@ast::crate_directive] cdirs,
                         str prefix, &mutable vec[@ast::view_item] view_items,
                         &mutable vec[@ast::item] items) {
    for (@ast::crate_directive sub_cdir in cdirs) {
        eval_crate_directive(cx, e, sub_cdir, prefix, view_items, items);
    }
}

fn eval_crate_directives_to_mod(ctx cx, env e,
                                vec[@ast::crate_directive] cdirs, str prefix)
   -> ast::_mod {
    let vec[@ast::view_item] view_items = [];
    let vec[@ast::item] items = [];
    eval_crate_directives(cx, e, cdirs, prefix, view_items, items);
    ret rec(view_items=view_items, items=items);
}

fn eval_crate_directive_block(ctx cx, env e, &ast::block blk, str prefix,
                              &mutable vec[@ast::view_item] view_items,
                              &mutable vec[@ast::item] items) {
    for (@ast::stmt s in blk.node.stmts) {
        alt (s.node) {
            case (ast::stmt_crate_directive(?cdir)) {
                eval_crate_directive(cx, e, cdir, prefix, view_items, items);
            }
            case (_) {
                cx.sess.span_fatal(s.span,
                                 "unsupported stmt in crate-directive block");
            }
        }
    }
}

fn eval_crate_directive_expr(ctx cx, env e, @ast::expr x, str prefix,
                             &mutable vec[@ast::view_item] view_items,
                             &mutable vec[@ast::item] items) {
    alt (x.node) {
        case (ast::expr_if(?cond, ?thn, ?elopt)) {
            auto cv = eval_expr(cx, e, cond);
            if (!val_is_bool(cv)) {
                cx.sess.span_fatal(x.span, "bad cond type in 'if'");
            }
            if (val_as_bool(cv)) {
                ret eval_crate_directive_block(cx, e, thn, prefix, view_items,
                                               items);
            }
            alt (elopt) {
                case (some(?els)) {
                    ret eval_crate_directive_expr(cx, e, els, prefix,
                                                  view_items, items);
                }
                case (_) {
                    // Absent-else is ok.

                }
            }
        }
        case (ast::expr_alt(?v, ?arms)) {
            auto vv = eval_expr(cx, e, v);
            for (ast::arm arm in arms) {
                alt (arm.pat.node) {
                    case (ast::pat_lit(?lit, _)) {
                        auto pv = eval_lit(cx, arm.pat.span, lit);
                        if (val_eq(cx.sess, arm.pat.span, vv, pv)) {
                            ret eval_crate_directive_block(cx, e, arm.block,
                                                           prefix, view_items,
                                                           items);
                        }
                    }
                    case (ast::pat_wild(_)) {
                        ret eval_crate_directive_block(cx, e, arm.block,
                                                       prefix, view_items,
                                                       items);
                    }
                    case (_) {
                        cx.sess.span_fatal(arm.pat.span,
                                         "bad pattern type in 'alt'");
                    }
                }
            }
            cx.sess.span_fatal(x.span, "no cases matched in 'alt'");
        }
        case (ast::expr_block(?block)) {
            ret eval_crate_directive_block(cx, e, block, prefix, view_items,
                                           items);
        }
        case (_) { cx.sess.span_fatal(x.span, "unsupported expr type"); }
    }
}

fn eval_crate_directive(ctx cx, env e, @ast::crate_directive cdir, str prefix,
                        &mutable vec[@ast::view_item] view_items,
                        &mutable vec[@ast::item] items) {
    alt (cdir.node) {
        case (ast::cdir_let(?id, ?x, ?cdirs)) {
            auto v = eval_expr(cx, e, x);
            auto e0 = [tup(id, v)] + e;
            eval_crate_directives(cx, e0, cdirs, prefix, view_items, items);
        }
        case (ast::cdir_expr(?x)) {
            eval_crate_directive_expr(cx, e, x, prefix, view_items, items);
        }
        case (ast::cdir_src_mod(?id, ?file_opt, ?attrs)) {
            auto file_path = id + ".rs";
            alt (file_opt) {
                case (some(?f)) { file_path = f; }
                case (none) { }
            }
            auto full_path = if (std::fs::path_is_absolute(file_path)) {
                file_path
            } else {
                prefix + std::fs::path_sep() + file_path
            };
            if (cx.mode == mode_depend) { cx.deps += [full_path]; ret; }
            auto p0 =
                new_parser(cx.sess, e, full_path, cx.chpos,
                           cx.next_id);
            auto inner_attrs = parse_inner_attrs_and_next(p0);
            auto mod_attrs = attrs + inner_attrs._0;
            auto first_item_outer_attrs = inner_attrs._1;
            auto m0 = parse_mod_items(p0, token::EOF, first_item_outer_attrs);

            auto i = front::parser::mk_item(p0, cdir.span.lo, cdir.span.hi,
                                            id, ast::item_mod(m0),
                                            mod_attrs);
            // Thread defids and chpos through the parsers
            cx.chpos = p0.get_chpos();
            cx.next_id = p0.next_id();
            vec::push[@ast::item](items, i);
        }
        case (ast::cdir_dir_mod(?id, ?dir_opt, ?cdirs, ?attrs)) {
            auto path = id;
            alt (dir_opt) { case (some(?d)) { path = d; } case (none) { } }
            auto full_path = if (std::fs::path_is_absolute(path)) {
                path
            } else {
                prefix + std::fs::path_sep() + path
            };
            auto m0 = eval_crate_directives_to_mod(cx, e, cdirs, full_path);
            auto i = @rec(ident=id,
                          attrs=attrs,
                          id=cx.next_id,
                          node=ast::item_mod(m0),
                          span=cdir.span);
            cx.next_id += 1;
            vec::push[@ast::item](items, i);
        }
        case (ast::cdir_view_item(?vi)) {
            vec::push[@ast::view_item](view_items, vi);
        }
        case (ast::cdir_syntax(?pth)) { }
        case (ast::cdir_auth(?pth, ?eff)) { }
    }
}
//
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
//
