
import ast;
import ast::ty_param;
import ast::respan;
import std::option;
import std::option::some;
import std::option::none;
import codemap::span;

type ast_visitor =
    {keep_going: fn() -> bool ,
     want_crate_directives: fn() -> bool ,
     visit_crate_pre: fn(&ast::crate) ,
     visit_crate_post: fn(&ast::crate) ,
     visit_crate_directive_pre: fn(&@ast::crate_directive) ,
     visit_crate_directive_post: fn(&@ast::crate_directive) ,
     visit_view_item_pre: fn(&@ast::view_item) ,
     visit_view_item_post: fn(&@ast::view_item) ,
     visit_native_item_pre: fn(&@ast::native_item) ,
     visit_native_item_post: fn(&@ast::native_item) ,
     visit_item_pre: fn(&@ast::item) ,
     visit_item_post: fn(&@ast::item) ,
     visit_method_pre: fn(&@ast::method) ,
     visit_method_post: fn(&@ast::method) ,
     visit_block_pre: fn(&ast::blk) ,
     visit_block_post: fn(&ast::blk) ,
     visit_stmt_pre: fn(&@ast::stmt) ,
     visit_stmt_post: fn(&@ast::stmt) ,
     visit_arm_pre: fn(&ast::arm) ,
     visit_arm_post: fn(&ast::arm) ,
     visit_pat_pre: fn(&@ast::pat) ,
     visit_pat_post: fn(&@ast::pat) ,
     visit_decl_pre: fn(&@ast::decl) ,
     visit_decl_post: fn(&@ast::decl) ,
     visit_local_pre: fn(&@ast::local) ,
     visit_local_post: fn(&@ast::local) ,
     visit_expr_pre: fn(&@ast::expr) ,
     visit_expr_post: fn(&@ast::expr) ,
     visit_ty_pre: fn(&@ast::ty) ,
     visit_ty_post: fn(&@ast::ty) ,
     visit_constr: fn(&@ast::constr) ,
     visit_fn_pre:
         fn(&ast::_fn, &ast::ty_param[], &span, &ast::fn_ident, ast::node_id)
             ,
     visit_fn_post:
         fn(&ast::_fn, &ast::ty_param[], &span, &ast::fn_ident, ast::node_id)
             };

fn walk_crate(v: &ast_visitor, c: &ast::crate) {
    if !v.keep_going() { ret; }
    v.visit_crate_pre(c);
    walk_mod(v, c.node.module);
    v.visit_crate_post(c);
}

fn walk_crate_directive(v: &ast_visitor, cd: @ast::crate_directive) {
    if !v.keep_going() { ret; }
    if !v.want_crate_directives() { ret; }
    v.visit_crate_directive_pre(cd);
    alt cd.node {
      ast::cdir_src_mod(_, _, _) { }
      ast::cdir_dir_mod(_, _, cdirs, _) {
        for cdir: @ast::crate_directive  in cdirs {
            walk_crate_directive(v, cdir);
        }
      }
      ast::cdir_view_item(vi) { walk_view_item(v, vi); }
      ast::cdir_syntax(_) { }
      ast::cdir_auth(_, _) { }
    }
    v.visit_crate_directive_post(cd);
}

fn walk_mod(v: &ast_visitor, m: &ast::_mod) {
    if !v.keep_going() { ret; }
    for vi: @ast::view_item  in m.view_items { walk_view_item(v, vi); }
    for i: @ast::item  in m.items { walk_item(v, i); }
}

fn walk_view_item(v: &ast_visitor, vi: @ast::view_item) {
    if !v.keep_going() { ret; }
    v.visit_view_item_pre(vi);
    v.visit_view_item_post(vi);
}

fn walk_local(v: &ast_visitor, loc: @ast::local) {
    v.visit_local_pre(loc);
    alt loc.node.ty { none. { } some(t) { walk_ty(v, t); } }
    alt loc.node.init { none. { } some(i) { walk_expr(v, i.expr); } }
    v.visit_local_post(loc);
}

fn walk_item(v: &ast_visitor, i: @ast::item) {
    if !v.keep_going() { ret; }
    v.visit_item_pre(i);
    alt i.node {
      ast::item_const(t, e) { walk_ty(v, t); walk_expr(v, e); }
      ast::item_fn(f, tps) {
        walk_fn(v, f, tps, i.span, some(i.ident), i.id);
      }
      ast::item_mod(m) { walk_mod(v, m); }
      ast::item_native_mod(nm) { walk_native_mod(v, nm); }
      ast::item_ty(t, _) { walk_ty(v, t); }
      ast::item_res(f, dtor_id, tps, _) {
        walk_fn(v, f, tps, i.span, some(i.ident), dtor_id);
      }
      ast::item_tag(variants, _) {
        for vr: ast::variant  in variants {
            for va: ast::variant_arg  in vr.node.args { walk_ty(v, va.ty); }
        }
      }
      ast::item_obj(ob, _, _) {
        for f: ast::obj_field  in ob.fields { walk_ty(v, f.ty); }
        for m: @ast::method  in ob.methods {
            v.visit_method_pre(m);
            // Methods don't have ty params?
            walk_fn(v, m.node.meth, ~[], m.span, some(m.node.ident),
                    m.node.id);
            v.visit_method_post(m);
        }
        alt ob.dtor {
          none. { }
          some(m) {
            walk_fn(v, m.node.meth, ~[], m.span, some(m.node.ident),
                    m.node.id);
          }
        }
      }
    }
    v.visit_item_post(i);
}

fn walk_ty(v: &ast_visitor, t: @ast::ty) {
    if !v.keep_going() { ret; }
    v.visit_ty_pre(t);
    alt t.node {
      ast::ty_nil. { }
      ast::ty_bot. { }
      ast::ty_bool. { }
      ast::ty_int. { }
      ast::ty_uint. { }
      ast::ty_float. { }
      ast::ty_machine(_) { }
      ast::ty_char. { }
      ast::ty_str. { }
      ast::ty_istr. { }
      ast::ty_box(mt) { walk_ty(v, mt.ty); }
      ast::ty_vec(mt) { walk_ty(v, mt.ty); }
      ast::ty_ivec(mt) { walk_ty(v, mt.ty); }
      ast::ty_ptr(mt) { walk_ty(v, mt.ty); }
      ast::ty_task. { }
      ast::ty_port(t) { walk_ty(v, t); }
      ast::ty_chan(t) { walk_ty(v, t); }
      ast::ty_rec(flds) {
        for f: ast::ty_field  in flds { walk_ty(v, f.node.mt.ty); }
      }
      ast::ty_fn(_, args, out, _, constrs) {
        for a: ast::ty_arg  in args { walk_ty(v, a.node.ty); }
        for c: @ast::constr  in constrs { v.visit_constr(c); }
        walk_ty(v, out);
      }
      ast::ty_obj(tmeths) {
        for m: ast::ty_method  in tmeths {
            for a: ast::ty_arg  in m.node.inputs { walk_ty(v, a.node.ty); }
            walk_ty(v, m.node.output);
        }
      }
      ast::ty_path(p, _) {
        for tp: @ast::ty  in p.node.types { walk_ty(v, tp); }
      }
      ast::ty_type. { }
      ast::ty_constr(t, _) { walk_ty(v, t); }
    }
    v.visit_ty_post(t);
}

fn walk_pat(v: &ast_visitor, p: &@ast::pat) {
    v.visit_pat_pre(p);
    alt p.node {
      ast::pat_tag(path, children) {
        for tp: @ast::ty  in path.node.types { walk_ty(v, tp); }
        for child: @ast::pat  in children { walk_pat(v, child); }
      }
      ast::pat_rec(fields, _) {
        for f: ast::field_pat  in fields { walk_pat(v, f.pat); }
      }
      ast::pat_box(inner) { walk_pat(v, inner); }
      _ { }
    }
    v.visit_pat_post(p);
}

fn walk_native_mod(v: &ast_visitor, nm: &ast::native_mod) {
    if !v.keep_going() { ret; }
    for vi: @ast::view_item  in nm.view_items { walk_view_item(v, vi); }
    for ni: @ast::native_item  in nm.items { walk_native_item(v, ni); }
}

fn walk_native_item(v: &ast_visitor, ni: @ast::native_item) {
    if !v.keep_going() { ret; }
    v.visit_native_item_pre(ni);
    alt ni.node {
      ast::native_item_fn(_, fd, _) { walk_fn_decl(v, fd); }
      ast::native_item_ty. { }
    }
    v.visit_native_item_post(ni);
}

fn walk_fn_decl(v: &ast_visitor, fd: &ast::fn_decl) {
    for a: ast::arg  in fd.inputs { walk_ty(v, a.ty); }
    for c: @ast::constr  in fd.constraints { v.visit_constr(c); }
    walk_ty(v, fd.output);
}

fn walk_fn(v: &ast_visitor, f: &ast::_fn, tps: &ast::ty_param[], sp: &span,
           i: &ast::fn_ident, d: ast::node_id) {
    if !v.keep_going() { ret; }
    v.visit_fn_pre(f, tps, sp, i, d);
    walk_fn_decl(v, f.decl);
    walk_block(v, f.body);
    v.visit_fn_post(f, tps, sp, i, d);
}

fn walk_block(v: &ast_visitor, b: &ast::blk) {
    if !v.keep_going() { ret; }
    v.visit_block_pre(b);
    for s: @ast::stmt  in b.node.stmts { walk_stmt(v, s); }
    walk_expr_opt(v, b.node.expr);
    v.visit_block_post(b);
}

fn walk_stmt(v: &ast_visitor, s: @ast::stmt) {
    if !v.keep_going() { ret; }
    v.visit_stmt_pre(s);
    alt s.node {
      ast::stmt_decl(d, _) { walk_decl(v, d); }
      ast::stmt_expr(e, _) { walk_expr(v, e); }
      ast::stmt_crate_directive(cdir) { walk_crate_directive(v, cdir); }
    }
    v.visit_stmt_post(s);
}

fn walk_decl(v: &ast_visitor, d: @ast::decl) {
    if !v.keep_going() { ret; }
    v.visit_decl_pre(d);
    alt d.node {
      ast::decl_local(locs) {
        for loc: @ast::local  in locs { walk_local(v, loc); }
      }
      ast::decl_item(it) { walk_item(v, it); }
    }
    v.visit_decl_post(d);
}

fn walk_expr_opt(v: &ast_visitor, eo: option::t[@ast::expr]) {
    alt eo { none. { } some(e) { walk_expr(v, e); } }
}

fn walk_exprs(v: &ast_visitor, exprs: &(@ast::expr)[]) {
    for e: @ast::expr  in exprs { walk_expr(v, e); }
}

fn walk_mac(v: &ast_visitor, mac: ast::mac) {
    alt mac.node {
      ast::mac_invoc(pth, args, body) { walk_exprs(v, args); }
      ast::mac_embed_type(ty) { walk_ty(v, ty); }
      ast::mac_embed_block(blk) { walk_block(v, blk); }
      ast::mac_ellipsis. { }
    }
}

fn walk_expr(v: &ast_visitor, e: @ast::expr) {
    if !v.keep_going() { ret; }
    v.visit_expr_pre(e);
    alt e.node {
      ast::expr_vec(es, _, _) { walk_exprs(v, es); }
      ast::expr_rec(flds, base) {
        for f: ast::field  in flds { walk_expr(v, f.node.expr); }
        walk_expr_opt(v, base);
      }
      ast::expr_call(callee, args) {
        walk_expr(v, callee);
        walk_exprs(v, args);
      }
      ast::expr_self_method(_) { }
      ast::expr_bind(callee, args) {
        walk_expr(v, callee);
        for eo: option::t[@ast::expr]  in args { walk_expr_opt(v, eo); }
      }
      ast::expr_spawn(_, _, callee, args) {
        walk_expr(v, callee);
        walk_exprs(v, args);
      }
      ast::expr_binary(_, a, b) { walk_expr(v, a); walk_expr(v, b); }
      ast::expr_unary(_, a) { walk_expr(v, a); }
      ast::expr_lit(_) { }
      ast::expr_cast(x, t) { walk_expr(v, x); walk_ty(v, t); }
      ast::expr_if(x, b, eo) {
        walk_expr(v, x);
        walk_block(v, b);
        walk_expr_opt(v, eo);
      }
      ast::expr_if_check(x, b, eo) {
        walk_expr(v, x);
        walk_block(v, b);
        walk_expr_opt(v, eo);
      }
      ast::expr_ternary(c, t, e) {
        walk_expr(v, c);
        walk_expr(v, t);
        walk_expr(v, e);
      }
      ast::expr_while(x, b) { walk_expr(v, x); walk_block(v, b); }
      ast::expr_for(dcl, x, b) {
        walk_local(v, dcl);
        walk_expr(v, x);
        walk_block(v, b);
      }
      ast::expr_for_each(dcl, x, b) {
        walk_local(v, dcl);
        walk_expr(v, x);
        walk_block(v, b);
      }
      ast::expr_do_while(b, x) { walk_block(v, b); walk_expr(v, x); }
      ast::expr_alt(x, arms) {
        walk_expr(v, x);
        for a: ast::arm  in arms {
            for p: @ast::pat  in a.pats { walk_pat(v, p); }
            v.visit_arm_pre(a);
            walk_block(v, a.block);
            v.visit_arm_post(a);
        }
      }
      ast::expr_fn(f) { walk_fn(v, f, ~[], e.span, none, e.id); }
      ast::expr_block(b) { walk_block(v, b); }
      ast::expr_assign(a, b) { walk_expr(v, a); walk_expr(v, b); }
      ast::expr_move(a, b) { walk_expr(v, a); walk_expr(v, b); }
      ast::expr_swap(a, b) { walk_expr(v, a); walk_expr(v, b); }
      ast::expr_assign_op(_, a, b) { walk_expr(v, a); walk_expr(v, b); }
      ast::expr_send(a, b) { walk_expr(v, a); walk_expr(v, b); }
      ast::expr_recv(a, b) { walk_expr(v, a); walk_expr(v, b); }
      ast::expr_field(x, _) { walk_expr(v, x); }
      ast::expr_index(a, b) { walk_expr(v, a); walk_expr(v, b); }
      ast::expr_path(p) {
        for tp: @ast::ty  in p.node.types { walk_ty(v, tp); }
      }
      ast::expr_fail(eo) { walk_expr_opt(v, eo); }
      ast::expr_break. { }
      ast::expr_cont. { }
      ast::expr_ret(eo) { walk_expr_opt(v, eo); }
      ast::expr_put(eo) { walk_expr_opt(v, eo); }
      ast::expr_be(x) { walk_expr(v, x); }
      ast::expr_log(_, x) { walk_expr(v, x); }
      ast::expr_check(_, x) { walk_expr(v, x); }
      ast::expr_assert(x) { walk_expr(v, x); }
      ast::expr_port(_) { }
      ast::expr_chan(x) { walk_expr(v, x); }
      ast::expr_anon_obj(anon_obj) {
        // Fields

        alt anon_obj.fields {
          none. { }
          some(fields) {
            for f: ast::anon_obj_field  in fields {
                walk_ty(v, f.ty);
                walk_expr(v, f.expr);
            }
          }
        }
        // with_obj

        alt anon_obj.with_obj { none. { } some(e) { walk_expr(v, e); } }


        // Methods
        for m: @ast::method  in anon_obj.methods {
            v.visit_method_pre(m);
            walk_fn(v, m.node.meth, ~[], m.span, some(m.node.ident),
                    m.node.id);
            v.visit_method_post(m);
        }
      }
      ast::expr_mac(mac) { walk_mac(v, mac); }
    }
    v.visit_expr_post(e);
}

fn def_keep_going() -> bool { ret true; }

fn def_want_crate_directives() -> bool { ret false; }

fn def_visit_crate(c: &ast::crate) { }

fn def_visit_crate_directive(c: &@ast::crate_directive) { }

fn def_visit_view_item(vi: &@ast::view_item) { }

fn def_visit_native_item(ni: &@ast::native_item) { }

fn def_visit_item(i: &@ast::item) { }

fn def_visit_method(m: &@ast::method) { }

fn def_visit_block(b: &ast::blk) { }

fn def_visit_stmt(s: &@ast::stmt) { }

fn def_visit_arm(a: &ast::arm) { }

fn def_visit_pat(p: &@ast::pat) { }

fn def_visit_decl(d: &@ast::decl) { }

fn def_visit_local(l: &@ast::local) { }

fn def_visit_expr(e: &@ast::expr) { }

fn def_visit_ty(t: &@ast::ty) { }

fn def_visit_constr(c: &@ast::constr) { }

fn def_visit_fn(f: &ast::_fn, tps: &ast::ty_param[], sp: &span,
                i: &ast::fn_ident, d: ast::node_id) {
}

fn default_visitor() -> ast_visitor {
    ret {keep_going: def_keep_going,
         want_crate_directives: def_want_crate_directives,
         visit_crate_pre: def_visit_crate,
         visit_crate_post: def_visit_crate,
         visit_crate_directive_pre: def_visit_crate_directive,
         visit_crate_directive_post: def_visit_crate_directive,
         visit_view_item_pre: def_visit_view_item,
         visit_view_item_post: def_visit_view_item,
         visit_native_item_pre: def_visit_native_item,
         visit_native_item_post: def_visit_native_item,
         visit_item_pre: def_visit_item,
         visit_item_post: def_visit_item,
         visit_method_pre: def_visit_method,
         visit_method_post: def_visit_method,
         visit_block_pre: def_visit_block,
         visit_block_post: def_visit_block,
         visit_stmt_pre: def_visit_stmt,
         visit_stmt_post: def_visit_stmt,
         visit_arm_pre: def_visit_arm,
         visit_arm_post: def_visit_arm,
         visit_pat_pre: def_visit_pat,
         visit_pat_post: def_visit_pat,
         visit_decl_pre: def_visit_decl,
         visit_decl_post: def_visit_decl,
         visit_local_pre: def_visit_local,
         visit_local_post: def_visit_local,
         visit_expr_pre: def_visit_expr,
         visit_expr_post: def_visit_expr,
         visit_ty_pre: def_visit_ty,
         visit_ty_post: def_visit_ty,
         visit_constr: def_visit_constr,
         visit_fn_pre: def_visit_fn,
         visit_fn_post: def_visit_fn};
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
