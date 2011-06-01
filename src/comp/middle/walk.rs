import front::ast;

import std::option;
import std::option::some;
import std::option::none;

import util::common::span;

// FIXME: Should visit patterns as well.
type ast_visitor =
    rec(fn () -> bool                  keep_going,
        fn () -> bool                  want_crate_directives,
        fn (&ast::crate c)              visit_crate_pre,
        fn (&ast::crate c)              visit_crate_post,
        fn (&@ast::crate_directive cd)  visit_crate_directive_pre,
        fn (&@ast::crate_directive cd)  visit_crate_directive_post,
        fn (&@ast::view_item i)         visit_view_item_pre,
        fn (&@ast::view_item i)         visit_view_item_post,
        fn (&@ast::native_item i)       visit_native_item_pre,
        fn (&@ast::native_item i)       visit_native_item_post,
        fn (&@ast::item i)              visit_item_pre,
        fn (&@ast::item i)              visit_item_post,
        fn (&@ast::method m)            visit_method_pre,
        fn (&@ast::method m)            visit_method_post,
        fn (&ast::block b)              visit_block_pre,
        fn (&ast::block b)              visit_block_post,
        fn (&@ast::stmt s)              visit_stmt_pre,
        fn (&@ast::stmt s)              visit_stmt_post,
        fn (&ast::arm a)                visit_arm_pre,
        fn (&ast::arm a)                visit_arm_post,
        fn (&@ast::decl d)              visit_decl_pre,
        fn (&@ast::decl d)              visit_decl_post,
        fn (&@ast::expr e)              visit_expr_pre,
        fn (&@ast::expr e)              visit_expr_post,
        fn (&@ast::ty t)                visit_ty_pre,
        fn (&@ast::ty t)                visit_ty_post,
        fn (&ast::_fn f, &span sp, &ast::ident name, 
            &ast::def_id d_id, &ast::ann a)  visit_fn_pre,
        fn (&ast::_fn f, &span sp, &ast::ident name,
            &ast::def_id d_id, &ast::ann a)  visit_fn_post);

fn walk_crate(&ast_visitor v, &ast::crate c) {
    if (!v.keep_going()) { ret; }
    v.visit_crate_pre(c);
    walk_mod(v, c.node.module);
    v.visit_crate_post(c);
}

fn walk_crate_directive(&ast_visitor v, @ast::crate_directive cd) {
    if (!v.keep_going()) { ret; }
    if (!v.want_crate_directives()) { ret; }
    v.visit_crate_directive_pre(cd);
    alt (cd.node) {
        case (ast::cdir_let(_, ?e, ?cdirs)) {
            walk_expr(v, e);
            for (@ast::crate_directive cdir in cdirs) {
                walk_crate_directive(v, cdir);
            }
        }
        case (ast::cdir_src_mod(_, _)) {}
        case (ast::cdir_dir_mod(_, _, ?cdirs)) {
            for (@ast::crate_directive cdir in cdirs) {
                walk_crate_directive(v, cdir);
            }
        }
        case (ast::cdir_view_item(?vi)) {
            walk_view_item(v, vi);
        }
        case (ast::cdir_meta(_)) {}
        case (ast::cdir_syntax(_)) {}
        case (ast::cdir_auth(_, _)) {}
    }
    v.visit_crate_directive_post(cd);
}

fn walk_mod(&ast_visitor v, &ast::_mod m) {
    if (!v.keep_going()) { ret; }
    for (@ast::view_item vi in m.view_items) {
        walk_view_item(v, vi);
    }
    for (@ast::item i in m.items) {
        walk_item(v, i);
    }
}

fn walk_view_item(&ast_visitor v, @ast::view_item vi) {
    if (!v.keep_going()) { ret; }
    v.visit_view_item_pre(vi);
    v.visit_view_item_post(vi);
}

fn walk_item(&ast_visitor v, @ast::item i) {
    if (!v.keep_going()) { ret; }
    v.visit_item_pre(i);
    alt (i.node) {
        case (ast::item_const(_, ?t, ?e, _, _)) {
            walk_ty(v, t);
            walk_expr(v, e);
        }
        case (ast::item_fn(?nm, ?f, _, ?d, ?a)) {
            walk_fn(v, f, i.span, nm, d, a);
        }
        case (ast::item_mod(_, ?m, _)) {
            walk_mod(v, m);
        }
        case (ast::item_native_mod(_, ?nm, _)) {
            walk_native_mod(v, nm);
        }
        case (ast::item_ty(_, ?t, _, _, _)) {
            walk_ty(v, t);
        }
        case (ast::item_tag(_, ?variants, _, _, _)) {
            for (ast::variant vr in variants) {
                for (ast::variant_arg va in vr.node.args) {
                    walk_ty(v, va.ty);
                }
            }
        }
        case (ast::item_obj(_, ?ob, _, _, _)) {
            for (ast::obj_field f in ob.fields) {
                walk_ty(v, f.ty);
            }
            for (@ast::method m in ob.methods) {
                v.visit_method_pre(m);
                walk_fn(v, m.node.meth, m.span,
                        m.node.ident, m.node.id, m.node.ann);
                v.visit_method_post(m);
            }
            alt (ob.dtor) {
                case (none) {}
                case (some(?m)) {
                    walk_fn(v, m.node.meth, m.span, m.node.ident, m.node.id,
                            m.node.ann);
                }
            }
        }

    }
    v.visit_item_post(i);
}

fn walk_ty(&ast_visitor v, @ast::ty t) {
    if (!v.keep_going()) { ret; }
    v.visit_ty_pre(t);
    alt (t.node) {
        case (ast::ty_nil) {}
        case (ast::ty_bot) {}
        case (ast::ty_bool) {}
        case (ast::ty_int) {}
        case (ast::ty_uint) {}
        case (ast::ty_float) {}
        case (ast::ty_machine(_)) {}
        case (ast::ty_char) {}
        case (ast::ty_str) {}
        case (ast::ty_box(?mt)) { walk_ty(v, mt.ty); }
        case (ast::ty_vec(?mt)) { walk_ty(v, mt.ty); }
        case (ast::ty_task) {}
        case (ast::ty_port(?t)) { walk_ty(v, t); }
        case (ast::ty_chan(?t)) { walk_ty(v, t); }
        case (ast::ty_tup(?mts)) {
            for (ast::mt mt in mts) {
                walk_ty(v, mt.ty);
            }
        }
        case (ast::ty_rec(?flds)) {
            for (ast::ty_field f in flds) {
                walk_ty(v, f.mt.ty);
            }
        }
        case (ast::ty_fn(_, ?args, ?out, _)) {
            for (ast::ty_arg a in args) {
                walk_ty(v, a.ty);
            }
            walk_ty(v, out);
        }
        case (ast::ty_obj(?tmeths)) {
            for (ast::ty_method m in tmeths) {
                for (ast::ty_arg a in m.inputs) {
                    walk_ty(v, a.ty);
                }
                walk_ty(v, m.output);
            }
        }
        case (ast::ty_path(?p, _)) {
            for (@ast::ty tp in p.node.types) {
                walk_ty(v, tp);
            }
        }
        case (ast::ty_type) {}
        case (ast::ty_constr(?t, _)) { walk_ty(v, t); }
    }
    v.visit_ty_post(t);
}

fn walk_pat(&ast_visitor v, &@ast::pat p) {
    alt (p.node) {
        case (ast::pat_tag(?path, ?children, _)) {
            for (@ast::ty tp in path.node.types) {
                walk_ty(v, tp);
            }
            for (@ast::pat child in children) {
                walk_pat(v, child);
            }
        }
        case (_) {}
    }
}

fn walk_native_mod(&ast_visitor v, &ast::native_mod nm) {
    if (!v.keep_going()) { ret; }
    for (@ast::view_item vi in nm.view_items) {
        walk_view_item(v, vi);
    }
    for (@ast::native_item ni in nm.items) {
        walk_native_item(v, ni);
    }
}

fn walk_native_item(&ast_visitor v, @ast::native_item ni) {
    if (!v.keep_going()) { ret; }
    v.visit_native_item_pre(ni);
    alt (ni.node) {
        case (ast::native_item_fn(_, _, ?fd, _, _, _)) {
            walk_fn_decl(v, fd);
        }
        case (ast::native_item_ty(_, _)) {
        }
    }
    v.visit_native_item_post(ni);
}

fn walk_fn_decl(&ast_visitor v, &ast::fn_decl fd) {
    for (ast::arg a in fd.inputs) {
        walk_ty(v, a.ty);
    }
    walk_ty(v, fd.output);
}

fn walk_fn(&ast_visitor v, &ast::_fn f, &span sp, &ast::ident i,
           &ast::def_id d, &ast::ann a) {
    if (!v.keep_going()) { ret; }
    v.visit_fn_pre(f, sp, i, d, a);
    walk_fn_decl(v, f.decl);
    walk_block(v, f.body);
    v.visit_fn_post(f, sp, i, d, a);
}

fn walk_block(&ast_visitor v, &ast::block b) {
    if (!v.keep_going()) { ret; }
    v.visit_block_pre(b);
    for (@ast::stmt s in b.node.stmts) {
        walk_stmt(v, s);
    }
    walk_expr_opt(v, b.node.expr);
    v.visit_block_post(b);
}

fn walk_stmt(&ast_visitor v, @ast::stmt s) {
    if (!v.keep_going()) { ret; }
    v.visit_stmt_pre(s);
    alt (s.node) {
        case (ast::stmt_decl(?d, _)) {
            walk_decl(v, d);
        }
        case (ast::stmt_expr(?e, _)) {
            walk_expr(v, e);
        }
        case (ast::stmt_crate_directive(?cdir)) {
            walk_crate_directive(v, cdir);
        }
    }
    v.visit_stmt_post(s);
}

fn walk_decl(&ast_visitor v, @ast::decl d) {
    if (!v.keep_going()) { ret; }
    v.visit_decl_pre(d);
    alt (d.node) {
        case (ast::decl_local(?loc)) {
            alt (loc.ty) {
                case (none) {}
                case (some(?t)) { walk_ty(v, t); }
            }
            alt (loc.init) {
                case (none) {}
                case (some(?i)) {
                    walk_expr(v, i.expr);
                }
            }
        }
        case (ast::decl_item(?it)) {
            walk_item(v, it);
        }
    }
    v.visit_decl_post(d);
}

fn walk_expr_opt(&ast_visitor v, option::t[@ast::expr] eo) {
    alt (eo) {
        case (none) {}
        case (some(?e)) {
            walk_expr(v, e);
        }
    }
}

fn walk_exprs(&ast_visitor v, vec[@ast::expr] exprs) {
    for (@ast::expr e in exprs) {
        walk_expr(v, e);
    }
}

fn walk_expr(&ast_visitor v, @ast::expr e) {
    if (!v.keep_going()) { ret; }
    v.visit_expr_pre(e);
    alt (e.node) {
        case (ast::expr_vec(?es, _, _)) {
            walk_exprs(v, es);
        }
        case (ast::expr_tup(?elts, _)) {
            for (ast::elt e in elts) {
                walk_expr(v, e.expr);
            }
        }
        case (ast::expr_rec(?flds, ?base, _)) {
            for (ast::field f in flds) {
                walk_expr(v, f.node.expr);
            }
            walk_expr_opt(v, base);
        }
        case (ast::expr_call(?callee, ?args, _)) {
            walk_expr(v, callee);
            walk_exprs(v, args);
        }
        case (ast::expr_self_method(_, _)) { }
        case (ast::expr_bind(?callee, ?args, _)) {
            walk_expr(v, callee);
            for (option::t[@ast::expr] eo in args) {
                walk_expr_opt(v, eo);
            }
        }
        case (ast::expr_spawn(_, _, ?callee, ?args, _)) {
            walk_expr(v, callee);
            walk_exprs(v, args);
        }
        case (ast::expr_binary(_, ?a, ?b, _)) {
            walk_expr(v, a);
            walk_expr(v, b);
        }
        case (ast::expr_unary(_, ?a, _)) {
            walk_expr(v, a);
        }
        case (ast::expr_lit(_, _)) { }
        case (ast::expr_cast(?x, ?t, _)) {
            walk_expr(v, x);
            walk_ty(v, t);
        }
        case (ast::expr_if(?x, ?b, ?eo, _)) {
            walk_expr(v, x);
            walk_block(v, b);
            walk_expr_opt(v, eo);
        }
        case (ast::expr_while(?x, ?b, _)) {
            walk_expr(v, x);
            walk_block(v, b);
        }
        case (ast::expr_for(?dcl, ?x, ?b, _)) {
            walk_decl(v, dcl);
            walk_expr(v, x);
            walk_block(v, b);
        }
        case (ast::expr_for_each(?dcl, ?x, ?b, _)) {
            walk_decl(v, dcl);
            walk_expr(v, x);
            walk_block(v, b);
        }
        case (ast::expr_do_while(?b, ?x, _)) {
            walk_block(v, b);
            walk_expr(v, x);
        }
        case (ast::expr_alt(?x, ?arms, _)) {
            walk_expr(v, x);
            for (ast::arm a in arms) {
                walk_pat(v, a.pat);
                v.visit_arm_pre(a);
                walk_block(v, a.block);
                v.visit_arm_post(a);
            }
        }
        case (ast::expr_block(?b, _)) {
            walk_block(v, b);
        }
        case (ast::expr_assign(?a, ?b, _)) {
            walk_expr(v, a);
            walk_expr(v, b);
        }
        case (ast::expr_move(?a, ?b, _)) {
            walk_expr(v, a);
            walk_expr(v, b);
        }
        case (ast::expr_assign_op(_, ?a, ?b, _)) {
            walk_expr(v, a);
            walk_expr(v, b);
        }
        case (ast::expr_send(?a, ?b, _)) {
            walk_expr(v, a);
            walk_expr(v, b);
        }
        case (ast::expr_recv(?a, ?b, _)) {
            walk_expr(v, a);
            walk_expr(v, b);
        }
        case (ast::expr_field(?x, _, _)) {
            walk_expr(v, x);
        }
        case (ast::expr_index(?a, ?b, _)) {
            walk_expr(v, a);
            walk_expr(v, b);
        }
        case (ast::expr_path(?p, _)) {
            for (@ast::ty tp in p.node.types) {
                walk_ty(v, tp);
            }
        }
        case (ast::expr_ext(_, ?args, ?body, ?expansion, _)) {
            // Only walk expansion, not args/body.
            walk_expr(v, expansion);
        }
        case (ast::expr_fail(_)) { }
        case (ast::expr_break(_)) { }
        case (ast::expr_cont(_)) { }
        case (ast::expr_ret(?eo, _)) {
            walk_expr_opt(v, eo);
        }
        case (ast::expr_put(?eo, _)) {
            walk_expr_opt(v, eo);
        }
        case (ast::expr_be(?x, _)) {
            walk_expr(v, x);
        }
        case (ast::expr_log(_,?x, _)) {
            walk_expr(v, x);
        }
        case (ast::expr_check(?x, _)) {
            walk_expr(v, x);
        }
        case (ast::expr_assert(?x, _)) {
            walk_expr(v, x);
        }
        case (ast::expr_port(_)) { }
        case (ast::expr_chan(?x, _)) {
            walk_expr(v, x);
        }

        case (ast::expr_anon_obj(?anon_obj,_,_,_)) { 

            // Fields
            let option::t[vec[ast::obj_field]] fields 
                = none[vec[ast::obj_field]];

            alt (anon_obj.fields) {
                case (none) { }
                case (some(?fields)) {
                    for (ast::obj_field f in fields) {
                        walk_ty(v, f.ty);
                    }
                }
            }

            // with_obj
            let option::t[@ast::expr] with_obj = none[@ast::expr];
            alt (anon_obj.with_obj) {
                case (none) { }
                case (some(?e)) {
                    walk_expr(v, e);
                }
            }

            // Methods
            for (@ast::method m in anon_obj.methods) {
                v.visit_method_pre(m);
                walk_fn(v, m.node.meth, m.span, m.node.ident, 
                        m.node.id, m.node.ann);
                v.visit_method_post(m);

            }
        }
    }
    v.visit_expr_post(e);
}

fn def_keep_going() -> bool { ret true; }
fn def_want_crate_directives() -> bool { ret false; }
fn def_visit_crate(&ast::crate c) { }
fn def_visit_crate_directive(&@ast::crate_directive c) { }
fn def_visit_view_item(&@ast::view_item vi) { }
fn def_visit_native_item(&@ast::native_item ni) { }
fn def_visit_item(&@ast::item i) { }
fn def_visit_method(&@ast::method m) { }
fn def_visit_block(&ast::block b) { }
fn def_visit_stmt(&@ast::stmt s) { }
fn def_visit_arm(&ast::arm a) { }
fn def_visit_decl(&@ast::decl d) { }
fn def_visit_expr(&@ast::expr e) { }
fn def_visit_ty(&@ast::ty t) { }
fn def_visit_fn(&ast::_fn f, &span sp, &ast::ident i, &ast::def_id d,
                &ast::ann a) { }

fn default_visitor() -> ast_visitor {

    auto d_keep_going = def_keep_going;
    auto d_want_crate_directives = def_want_crate_directives;
    auto d_visit_crate = def_visit_crate;
    auto d_visit_crate_directive = def_visit_crate_directive;
    auto d_visit_view_item = def_visit_view_item;
    auto d_visit_native_item = def_visit_native_item;
    auto d_visit_item = def_visit_item;
    auto d_visit_method = def_visit_method;
    auto d_visit_block = def_visit_block;
    auto d_visit_stmt = def_visit_stmt;
    auto d_visit_arm = def_visit_arm;
    auto d_visit_decl = def_visit_decl;
    auto d_visit_expr = def_visit_expr;
    auto d_visit_ty = def_visit_ty;
    auto d_visit_fn = def_visit_fn;

    ret rec(keep_going = d_keep_going,
            want_crate_directives = d_want_crate_directives,
            visit_crate_pre = d_visit_crate,
            visit_crate_post = d_visit_crate,
            visit_crate_directive_pre = d_visit_crate_directive,
            visit_crate_directive_post = d_visit_crate_directive,
            visit_view_item_pre = d_visit_view_item,
            visit_view_item_post = d_visit_view_item,
            visit_native_item_pre = d_visit_native_item,
            visit_native_item_post = d_visit_native_item,
            visit_item_pre = d_visit_item,
            visit_item_post = d_visit_item,
            visit_method_pre = d_visit_method,
            visit_method_post = d_visit_method,
            visit_block_pre = d_visit_block,
            visit_block_post = d_visit_block,
            visit_stmt_pre = d_visit_stmt,
            visit_stmt_post = d_visit_stmt,
            visit_arm_pre = d_visit_arm,
            visit_arm_post = d_visit_arm,
            visit_decl_pre = d_visit_decl,
            visit_decl_post = d_visit_decl,
            visit_expr_pre = d_visit_expr,
            visit_expr_post = d_visit_expr,
            visit_ty_pre = d_visit_ty,
            visit_ty_post = d_visit_ty,
            visit_fn_pre = d_visit_fn,
            visit_fn_post = d_visit_fn);
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
