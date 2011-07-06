
import ast;
import ast::ty_param;
import ast::respan;
import std::option;
import std::option::some;
import std::option::none;
import codemap::span;

type ast_visitor =
    rec(fn() -> bool  keep_going,
        fn() -> bool  want_crate_directives,
        fn(&ast::crate)  visit_crate_pre,
        fn(&ast::crate)  visit_crate_post,
        fn(&@ast::crate_directive)  visit_crate_directive_pre,
        fn(&@ast::crate_directive)  visit_crate_directive_post,
        fn(&@ast::view_item)  visit_view_item_pre,
        fn(&@ast::view_item)  visit_view_item_post,
        fn(&@ast::native_item)  visit_native_item_pre,
        fn(&@ast::native_item)  visit_native_item_post,
        fn(&@ast::item)  visit_item_pre,
        fn(&@ast::item)  visit_item_post,
        fn(&@ast::method)  visit_method_pre,
        fn(&@ast::method)  visit_method_post,
        fn(&ast::block)  visit_block_pre,
        fn(&ast::block)  visit_block_post,
        fn(&@ast::stmt)  visit_stmt_pre,
        fn(&@ast::stmt)  visit_stmt_post,
        fn(&ast::arm)  visit_arm_pre,
        fn(&ast::arm)  visit_arm_post,
        fn(&@ast::pat)  visit_pat_pre,
        fn(&@ast::pat)  visit_pat_post,
        fn(&@ast::decl)  visit_decl_pre,
        fn(&@ast::decl)  visit_decl_post,
        fn(&@ast::local)  visit_local_pre,
        fn(&@ast::local)  visit_local_post,
        fn(&@ast::expr)  visit_expr_pre,
        fn(&@ast::expr)  visit_expr_post,
        fn(&@ast::ty)  visit_ty_pre,
        fn(&@ast::ty)  visit_ty_post,
        fn(&@ast::constr)  visit_constr,
        fn(&ast::_fn, &ast::ty_param[], &span, &ast::fn_ident,
           ast::node_id) visit_fn_pre,
        fn(&ast::_fn, &ast::ty_param[], &span, &ast::fn_ident,
           ast::node_id) visit_fn_post);

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
        case (ast::cdir_src_mod(_, _, _)) { }
        case (ast::cdir_dir_mod(_, _, ?cdirs, _)) {
            for (@ast::crate_directive cdir in cdirs) {
                walk_crate_directive(v, cdir);
            }
        }
        case (ast::cdir_view_item(?vi)) { walk_view_item(v, vi); }
        case (ast::cdir_syntax(_)) { }
        case (ast::cdir_auth(_, _)) { }
    }
    v.visit_crate_directive_post(cd);
}

fn walk_mod(&ast_visitor v, &ast::_mod m) {
    if (!v.keep_going()) { ret; }
    for (@ast::view_item vi in m.view_items) { walk_view_item(v, vi); }
    for (@ast::item i in m.items) { walk_item(v, i); }
}

fn walk_view_item(&ast_visitor v, @ast::view_item vi) {
    if (!v.keep_going()) { ret; }
    v.visit_view_item_pre(vi);
    v.visit_view_item_post(vi);
}

fn walk_local(&ast_visitor v, @ast::local loc) {
    v.visit_local_pre(loc);
    alt (loc.node.ty) { case (none) { } case (some(?t)) { walk_ty(v, t); } }
    alt (loc.node.init) {
        case (none) { }
        case (some(?i)) { walk_expr(v, i.expr); }
    }
    v.visit_local_post(loc);
}

fn walk_item(&ast_visitor v, @ast::item i) {
    if (!v.keep_going()) { ret; }
    v.visit_item_pre(i);
    alt (i.node) {
        case (ast::item_const(?t, ?e)) { walk_ty(v, t); walk_expr(v, e); }
        case (ast::item_fn(?f, ?tps)) {
            walk_fn(v, f, tps, i.span, some(i.ident), i.id);
        }
        case (ast::item_mod(?m)) { walk_mod(v, m); }
        case (ast::item_native_mod(?nm)) { walk_native_mod(v, nm); }
        case (ast::item_ty(?t, _)) { walk_ty(v, t); }
        case (ast::item_res(?f, ?dtor_id, ?tps, _)) {
            walk_fn(v, f, tps, i.span, some(i.ident), dtor_id);
        }
        case (ast::item_tag(?variants, _)) {
            for (ast::variant vr in variants) {
                for (ast::variant_arg va in vr.node.args) {
                    walk_ty(v, va.ty);
                }
            }
        }
        case (ast::item_obj(?ob, _, _)) {
            for (ast::obj_field f in ob.fields) { walk_ty(v, f.ty); }
            for (@ast::method m in ob.methods) {
                v.visit_method_pre(m);
                // Methods don't have ty params?
                walk_fn(v, m.node.meth, ~[], m.span,
                        some(m.node.ident), m.node.id);
                v.visit_method_post(m);
            }
            alt (ob.dtor) {
                case (none) { }
                case (some(?m)) {
                    walk_fn(v, m.node.meth, ~[], m.span,
                            some(m.node.ident), m.node.id);
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
        case (ast::ty_nil) { }
        case (ast::ty_bot) { }
        case (ast::ty_bool) { }
        case (ast::ty_int) { }
        case (ast::ty_uint) { }
        case (ast::ty_float) { }
        case (ast::ty_machine(_)) { }
        case (ast::ty_char) { }
        case (ast::ty_str) { }
        case (ast::ty_istr) { }
        case (ast::ty_box(?mt)) { walk_ty(v, mt.ty); }
        case (ast::ty_vec(?mt)) { walk_ty(v, mt.ty); }
        case (ast::ty_ivec(?mt)) { walk_ty(v, mt.ty); }
        case (ast::ty_ptr(?mt)) { walk_ty(v, mt.ty); }
        case (ast::ty_task) { }
        case (ast::ty_port(?t)) { walk_ty(v, t); }
        case (ast::ty_chan(?t)) { walk_ty(v, t); }
        case (ast::ty_tup(?mts)) {
            for (ast::mt mt in mts) { walk_ty(v, mt.ty); }
        }
        case (ast::ty_rec(?flds)) {
            for (ast::ty_field f in flds) { walk_ty(v, f.node.mt.ty); }
        }
        case (ast::ty_fn(_, ?args, ?out, _, ?constrs)) {
            for (ast::ty_arg a in args) { walk_ty(v, a.node.ty); }
            for (@ast::constr c in constrs) { v.visit_constr(c); }
            walk_ty(v, out);
        }
        case (ast::ty_obj(?tmeths)) {
            for (ast::ty_method m in tmeths) {
                for (ast::ty_arg a in m.node.inputs) {
                    walk_ty(v, a.node.ty);
                }
                walk_ty(v, m.node.output);
            }
        }
        case (ast::ty_path(?p, _)) {
            for (@ast::ty tp in p.node.types) { walk_ty(v, tp); }
        }
        case (ast::ty_type) { }
        case (ast::ty_constr(?t, _)) { walk_ty(v, t); }
    }
    v.visit_ty_post(t);
}

fn walk_pat(&ast_visitor v, &@ast::pat p) {
    v.visit_pat_pre(p);
    alt (p.node) {
        case (ast::pat_tag(?path, ?children)) {
            for (@ast::ty tp in path.node.types) { walk_ty(v, tp); }
            for (@ast::pat child in children) { walk_pat(v, child); }
        }
        case (ast::pat_rec(?fields, _)) {
            for (ast::field_pat f in fields) { walk_pat(v, f.pat); }
        }
        case (_) { }
    }
    v.visit_pat_post(p);
}

fn walk_native_mod(&ast_visitor v, &ast::native_mod nm) {
    if (!v.keep_going()) { ret; }
    for (@ast::view_item vi in nm.view_items) { walk_view_item(v, vi); }
    for (@ast::native_item ni in nm.items) { walk_native_item(v, ni); }
}

fn walk_native_item(&ast_visitor v, @ast::native_item ni) {
    if (!v.keep_going()) { ret; }
    v.visit_native_item_pre(ni);
    alt (ni.node) {
        case (ast::native_item_fn(_, ?fd, _)) {
            walk_fn_decl(v, fd);
        }
        case (ast::native_item_ty) { }
    }
    v.visit_native_item_post(ni);
}

fn walk_fn_decl(&ast_visitor v, &ast::fn_decl fd) {
    for (ast::arg a in fd.inputs) { walk_ty(v, a.ty); }
    for (@ast::constr c in fd.constraints) { v.visit_constr(c); }
    walk_ty(v, fd.output);
}

fn walk_fn(&ast_visitor v, &ast::_fn f, &ast::ty_param[] tps,
           &span sp, &ast::fn_ident i, ast::node_id d) {
    if (!v.keep_going()) { ret; }
    v.visit_fn_pre(f, tps, sp, i, d);
    walk_fn_decl(v, f.decl);
    walk_block(v, f.body);
    v.visit_fn_post(f, tps, sp, i, d);
}

fn walk_block(&ast_visitor v, &ast::block b) {
    if (!v.keep_going()) { ret; }
    v.visit_block_pre(b);
    for (@ast::stmt s in b.node.stmts) { walk_stmt(v, s); }
    walk_expr_opt(v, b.node.expr);
    v.visit_block_post(b);
}

fn walk_stmt(&ast_visitor v, @ast::stmt s) {
    if (!v.keep_going()) { ret; }
    v.visit_stmt_pre(s);
    alt (s.node) {
        case (ast::stmt_decl(?d, _)) { walk_decl(v, d); }
        case (ast::stmt_expr(?e, _)) { walk_expr(v, e); }
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
        case (ast::decl_local(?loc)) { walk_local(v, loc); }
        case (ast::decl_item(?it)) { walk_item(v, it); }
    }
    v.visit_decl_post(d);
}

fn walk_expr_opt(&ast_visitor v, option::t[@ast::expr] eo) {
    alt (eo) { case (none) { } case (some(?e)) { walk_expr(v, e); } }
}

fn walk_exprs(&ast_visitor v, &(@ast::expr)[] exprs) {
    for (@ast::expr e in exprs) { walk_expr(v, e); }
}

fn walk_expr(&ast_visitor v, @ast::expr e) {
    if (!v.keep_going()) { ret; }
    v.visit_expr_pre(e);
    alt (e.node) {
        case (ast::expr_vec(?es, _, _)) { walk_exprs(v, es); }
        case (ast::expr_tup(?elts)) {
            for (ast::elt e in elts) { walk_expr(v, e.expr); }
        }
        case (ast::expr_rec(?flds, ?base)) {
            for (ast::field f in flds) { walk_expr(v, f.node.expr); }
            walk_expr_opt(v, base);
        }
        case (ast::expr_call(?callee, ?args)) {
            walk_expr(v, callee);
            walk_exprs(v, args);
        }
        case (ast::expr_self_method(_)) { }
        case (ast::expr_bind(?callee, ?args)) {
            walk_expr(v, callee);
            for (option::t[@ast::expr] eo in args) { walk_expr_opt(v, eo); }
        }
        case (ast::expr_spawn(_, _, ?callee, ?args)) {
            walk_expr(v, callee);
            walk_exprs(v, args);
        }
        case (ast::expr_binary(_, ?a, ?b)) {
            walk_expr(v, a);
            walk_expr(v, b);
        }
        case (ast::expr_unary(_, ?a)) { walk_expr(v, a); }
        case (ast::expr_lit(_)) { }
        case (ast::expr_cast(?x, ?t)) { walk_expr(v, x); walk_ty(v, t); }
        case (ast::expr_if(?x, ?b, ?eo)) {
            walk_expr(v, x);
            walk_block(v, b);
            walk_expr_opt(v, eo);
        }
        case (ast::expr_if_check(?x, ?b, ?eo)) {
            walk_expr(v, x);
            walk_block(v, b);
            walk_expr_opt(v, eo);
        }
        case (ast::expr_ternary(?c, ?t, ?e)) {
            walk_expr(v, c);
            walk_expr(v, t);
            walk_expr(v, e);
        }
        case (ast::expr_while(?x, ?b)) {
            walk_expr(v, x);
            walk_block(v, b);
        }
        case (ast::expr_for(?dcl, ?x, ?b)) {
            walk_local(v, dcl);
            walk_expr(v, x);
            walk_block(v, b);
        }
        case (ast::expr_for_each(?dcl, ?x, ?b)) {
            walk_local(v, dcl);
            walk_expr(v, x);
            walk_block(v, b);
        }
        case (ast::expr_do_while(?b, ?x)) {
            walk_block(v, b);
            walk_expr(v, x);
        }
        case (ast::expr_alt(?x, ?arms)) {
            walk_expr(v, x);
            for (ast::arm a in arms) {
                for (@ast::pat p in a.pats) { walk_pat(v, p); }
                v.visit_arm_pre(a);
                walk_block(v, a.block);
                v.visit_arm_post(a);
            }
        }
        case (ast::expr_fn(?f)) {
            walk_fn(v, f, ~[], e.span, none, e.id);
        }
        case (ast::expr_block(?b)) { walk_block(v, b); }
        case (ast::expr_assign(?a, ?b)) {
            walk_expr(v, a);
            walk_expr(v, b);
        }
        case (ast::expr_move(?a, ?b)) { walk_expr(v, a); walk_expr(v, b); }
        case (ast::expr_swap(?a, ?b)) { walk_expr(v, a); walk_expr(v, b); }
        case (ast::expr_assign_op(_, ?a, ?b)) {
            walk_expr(v, a);
            walk_expr(v, b);
        }
        case (ast::expr_send(?a, ?b)) { walk_expr(v, a); walk_expr(v, b); }
        case (ast::expr_recv(?a, ?b)) { walk_expr(v, a); walk_expr(v, b); }
        case (ast::expr_field(?x, _)) { walk_expr(v, x); }
        case (ast::expr_index(?a, ?b)) {
            walk_expr(v, a);
            walk_expr(v, b);
        }
        case (ast::expr_path(?p)) {
            for (@ast::ty tp in p.node.types) { walk_ty(v, tp); }
        }
        case (ast::expr_ext(_, ?args, _)) {
            for (@ast::expr e in args) {
                walk_expr(v, e);
            }
        }
        case (ast::expr_fail(?eo)) { walk_expr_opt(v, eo); }
        case (ast::expr_break) { }
        case (ast::expr_cont) { }
        case (ast::expr_ret(?eo)) { walk_expr_opt(v, eo); }
        case (ast::expr_put(?eo)) { walk_expr_opt(v, eo); }
        case (ast::expr_be(?x)) { walk_expr(v, x); }
        case (ast::expr_log(_, ?x)) { walk_expr(v, x); }
        case (ast::expr_check(_, ?x)) { walk_expr(v, x); }
        case (ast::expr_assert(?x)) { walk_expr(v, x); }
        case (ast::expr_port(_)) { }
        case (ast::expr_chan(?x)) { walk_expr(v, x); }
        case (ast::expr_anon_obj(?anon_obj, _)) {
            // Fields

            alt (anon_obj.fields) {
                case (none) { }
                case (some(?fields)) {
                    for (ast::anon_obj_field f in fields) { 
                        walk_ty(v, f.ty);
                        walk_expr(v, f.expr);
                    }
                }
            }
            // with_obj

            alt (anon_obj.with_obj) {
                case (none) { }
                case (some(?e)) { walk_expr(v, e); }
            }

            // Methods
            for (@ast::method m in anon_obj.methods) {
                v.visit_method_pre(m);
                walk_fn(v, m.node.meth, ~[], m.span, some(m.node.ident),
                        m.node.id);
                v.visit_method_post(m);
            }
        }
        case (ast::expr_embeded_type(?ty)) {
            walk_ty(v, ty);
        }
        case (ast::expr_embeded_block(?blk)) {
            walk_block(v, blk);
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

fn def_visit_pat(&@ast::pat p) { }

fn def_visit_decl(&@ast::decl d) { }

fn def_visit_local(&@ast::local l) { }

fn def_visit_expr(&@ast::expr e) { }

fn def_visit_ty(&@ast::ty t) { }

fn def_visit_constr(&@ast::constr c) { }

fn def_visit_fn(&ast::_fn f, &ast::ty_param[] tps,
  &span sp, &ast::fn_ident i, ast::node_id d) { }

fn default_visitor() -> ast_visitor {
    ret rec(keep_going=def_keep_going,
            want_crate_directives=def_want_crate_directives,
            visit_crate_pre=def_visit_crate,
            visit_crate_post=def_visit_crate,
            visit_crate_directive_pre=def_visit_crate_directive,
            visit_crate_directive_post=def_visit_crate_directive,
            visit_view_item_pre=def_visit_view_item,
            visit_view_item_post=def_visit_view_item,
            visit_native_item_pre=def_visit_native_item,
            visit_native_item_post=def_visit_native_item,
            visit_item_pre=def_visit_item,
            visit_item_post=def_visit_item,
            visit_method_pre=def_visit_method,
            visit_method_post=def_visit_method,
            visit_block_pre=def_visit_block,
            visit_block_post=def_visit_block,
            visit_stmt_pre=def_visit_stmt,
            visit_stmt_post=def_visit_stmt,
            visit_arm_pre=def_visit_arm,
            visit_arm_post=def_visit_arm,
            visit_pat_pre=def_visit_pat,
            visit_pat_post=def_visit_pat,
            visit_decl_pre=def_visit_decl,
            visit_decl_post=def_visit_decl,
            visit_local_pre=def_visit_local,
            visit_local_post=def_visit_local,
            visit_expr_pre=def_visit_expr,
            visit_expr_post=def_visit_expr,
            visit_ty_pre=def_visit_ty,
            visit_ty_post=def_visit_ty,
            visit_constr=def_visit_constr,
            visit_fn_pre=def_visit_fn,
            visit_fn_post=def_visit_fn);
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
