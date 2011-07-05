
import ast::*;
import std::option;
import std::option::some;
import std::option::none;
import codemap::span;


// Context-passing AST walker. Each overridden visit method has full control
// over what happens with its node, it can do its own traversal of the node's
// children (potentially passing in different contexts to each), call
// visit::visit_* to apply the default traversal algorithm (again, it can
// override the context), or prevent deeper traversal by doing nothing.

// Lots of redundant indirection and refcounting. Our typesystem doesn't do
// circular types, so the visitor record can not hold functions that take
// visitors. A vt tag is used to break the cycle.
tag vt[E] { vtor(visitor[E]); }

fn vt[E](&vt[E] x) -> visitor[E] { alt (x) { case (vtor(?v)) { ret v; } } }

type visitor[E] =
    @rec(fn(&_mod, &span, &E, &vt[E])  visit_mod,
         fn(&@view_item, &E, &vt[E])  visit_view_item,
         fn(&@native_item, &E, &vt[E])  visit_native_item,
         fn(&@item, &E, &vt[E])  visit_item,
         fn(&@local, &E, &vt[E])  visit_local,
         fn(&block, &E, &vt[E])  visit_block,
         fn(&@stmt, &E, &vt[E])  visit_stmt,
         fn(&arm, &E, &vt[E])  visit_arm,
         fn(&@pat, &E, &vt[E])  visit_pat,
         fn(&@decl, &E, &vt[E])  visit_decl,
         fn(&@expr, &E, &vt[E])  visit_expr,
         fn(&@ty, &E, &vt[E])  visit_ty,
         fn(&@constr, &E, &vt[E])  visit_constr,
         fn(&_fn, &vec[ty_param], &span, &fn_ident, node_id, &E, &vt[E])
             visit_fn);

fn default_visitor[E]() -> visitor[E] {
    ret @rec(visit_mod=bind visit_mod[E](_, _, _, _),
             visit_view_item=bind visit_view_item[E](_, _, _),
             visit_native_item=bind visit_native_item[E](_, _, _),
             visit_item=bind visit_item[E](_, _, _),
             visit_local=bind visit_local[E](_, _, _),
             visit_block=bind visit_block[E](_, _, _),
             visit_stmt=bind visit_stmt[E](_, _, _),
             visit_arm=bind visit_arm[E](_, _, _),
             visit_pat=bind visit_pat[E](_, _, _),
             visit_decl=bind visit_decl[E](_, _, _),
             visit_expr=bind visit_expr[E](_, _, _),
             visit_ty=bind visit_ty[E](_, _, _),
             visit_constr=bind visit_constr[E](_, _, _),
             visit_fn=bind visit_fn[E](_, _, _, _, _, _, _));
}

fn visit_crate[E](&crate c, &E e, &vt[E] v) {
    vt(v).visit_mod(c.node.module, c.span, e, v);
}

fn visit_crate_directive[E](&@crate_directive cd, &E e, &vt[E] v) {
    alt (cd.node) {
        case (cdir_src_mod(_, _, _)) { }
        case (cdir_dir_mod(_, _, ?cdirs, _)) {
            for (@crate_directive cdir in cdirs) {
                visit_crate_directive(cdir, e, v);
            }
        }
        case (cdir_view_item(?vi)) { vt(v).visit_view_item(vi, e, v); }
        case (cdir_syntax(_)) { }
        case (cdir_auth(_, _)) { }
    }
}

fn visit_mod[E](&_mod m, &span sp, &E e, &vt[E] v) {
    for (@view_item vi in m.view_items) { vt(v).visit_view_item(vi, e, v); }
    for (@item i in m.items) { vt(v).visit_item(i, e, v); }
}

fn visit_view_item[E](&@view_item vi, &E e, &vt[E] v) { }

fn visit_local[E](&@local loc, &E e, &vt[E] v) {
    alt (loc.node.ty) {
        case (none) { }
        case (some(?t)) { vt(v).visit_ty(t, e, v); }
    }
    alt (loc.node.init) {
        case (none) { }
        case (some(?i)) { vt(v).visit_expr(i.expr, e, v); }
    }
}

fn visit_item[E](&@item i, &E e, &vt[E] v) {
    alt (i.node) {
        case (item_const(?t, ?ex)) {
            vt(v).visit_ty(t, e, v);
            vt(v).visit_expr(ex, e, v);
        }
        case (item_fn(?f, ?tp)) {
            vt(v).visit_fn(f, tp, i.span, some(i.ident), i.id, e, v);
        }
        case (item_mod(?m)) { vt(v).visit_mod(m, i.span, e, v); }
        case (item_native_mod(?nm)) {
            for (@view_item vi in nm.view_items) {
                vt(v).visit_view_item(vi, e, v);
            }
            for (@native_item ni in nm.items) {
                vt(v).visit_native_item(ni, e, v);
            }
        }
        case (item_ty(?t, _)) { vt(v).visit_ty(t, e, v); }
        case (item_res(?f, ?dtor_id, ?tps, _)) {
            vt(v).visit_fn(f, tps, i.span, some(i.ident), dtor_id, e, v);
        }
        case (item_tag(?variants, _)) {
            for (variant vr in variants) {
                for (variant_arg va in vr.node.args) {
                    vt(v).visit_ty(va.ty, e, v);
                }
            }
        }
        case (item_obj(?ob, _, _)) {
            for (obj_field f in ob.fields) { vt(v).visit_ty(f.ty, e, v); }
            for (@method m in ob.methods) {
                vt(v).visit_fn(m.node.meth, [], m.span, some(m.node.ident),
                               m.node.id, e, v);
            }
            alt (ob.dtor) {
                case (none) { }
                case (some(?m)) {
                    vt(v).visit_fn(m.node.meth, [], m.span,
                                   some(m.node.ident),
                                   m.node.id, e, v);
                }
            }
        }
    }
}

fn visit_ty[E](&@ty t, &E e, &vt[E] v) {
    alt (t.node) {
        case (ty_nil)           { /* no-op */ }
        case (ty_bot)           { /* no-op */ }
        case (ty_bool)          { /* no-op */ }
        case (ty_int)           { /* no-op */ }
        case (ty_float)         { /* no-op */ }
        case (ty_uint)          { /* no-op */ }
        case (ty_machine(_))    { /* no-op */ }
        case (ty_char)          { /* no-op */ }
        case (ty_str)           { /* no-op */ }
        case (ty_istr)          { /* no-op */ }
        case (ty_box(?mt))      { vt(v).visit_ty(mt.ty, e, v); }
        case (ty_vec(?mt))      { vt(v).visit_ty(mt.ty, e, v); }
        case (ty_ivec(?mt))     { vt(v).visit_ty(mt.ty, e, v); }
        case (ty_ptr(?mt))      { vt(v).visit_ty(mt.ty, e, v); }
        case (ty_port(?t))      { vt(v).visit_ty(t, e, v); }
        case (ty_chan(?t))      { vt(v).visit_ty(t, e, v); }
        case (ty_task)          { /* no-op */ }
        case (ty_tup(?mts)) {
            for (mt mt in mts) { vt(v).visit_ty(mt.ty, e, v); }
        }
        case (ty_rec(?flds)) {
            for (ty_field f in flds) { vt(v).visit_ty(f.node.mt.ty, e, v); }
        }
        case (ty_fn(_, ?args, ?out, _, ?constrs)) {
            for (ty_arg a in args) { vt(v).visit_ty(a.node.ty, e, v); }
            for (@constr c in constrs) { vt(v).visit_constr(c, e, v); }
            vt(v).visit_ty(out, e, v);
        }
        case (ty_obj(?tmeths)) {
            for (ty_method m in tmeths) {
                for (ty_arg a in m.node.inputs) {
                    vt(v).visit_ty(a.node.ty, e, v);
                }
                vt(v).visit_ty(m.node.output, e, v);
            }
        }
        case (ty_path(?p, _)) {
            for (@ty tp in p.node.types) { vt(v).visit_ty(tp, e, v); }
        }
        case (ty_type)          { /* no-op */ }
        case (ty_constr(?t, _)) { vt(v).visit_ty(t, e, v); }
    }
}

fn visit_constr[E](&@constr c, &E e, &vt[E] v) {
    // default

}

fn visit_pat[E](&@pat p, &E e, &vt[E] v) {
    alt (p.node) {
        case (pat_tag(?path, ?children)) {
            for (@ty tp in path.node.types) { vt(v).visit_ty(tp, e, v); }
            for (@pat child in children) { vt(v).visit_pat(child, e, v); }
        }
        case (_) { }
    }
}

fn visit_native_item[E](&@native_item ni, &E e, &vt[E] v) {
    alt (ni.node) {
        case (native_item_fn(_, ?fd, _)) { visit_fn_decl(fd, e, v); }
        case (native_item_ty) { }
    }
}

fn visit_fn_decl[E](&fn_decl fd, &E e, &vt[E] v) {
    for (arg a in fd.inputs) { vt(v).visit_ty(a.ty, e, v); }
    for (@constr c in fd.constraints) { vt(v).visit_constr(c, e, v); }
    vt(v).visit_ty(fd.output, e, v);
}

fn visit_fn[E](&_fn f, &vec[ty_param] tp, &span sp, &fn_ident i,
               node_id id, &E e, &vt[E] v) {
    visit_fn_decl(f.decl, e, v);
    vt(v).visit_block(f.body, e, v);
}

fn visit_block[E](&block b, &E e, &vt[E] v) {
    for (@stmt s in b.node.stmts) { vt(v).visit_stmt(s, e, v); }
    visit_expr_opt(b.node.expr, e, v);
}

fn visit_stmt[E](&@stmt s, &E e, &vt[E] v) {
    alt (s.node) {
        case (stmt_decl(?d, _)) { vt(v).visit_decl(d, e, v); }
        case (stmt_expr(?ex, _)) { vt(v).visit_expr(ex, e, v); }
        case (stmt_crate_directive(?cd)) { visit_crate_directive(cd, e, v); }
    }
}

fn visit_decl[E](&@decl d, &E e, &vt[E] v) {
    alt (d.node) {
        case (decl_local(?loc)) {
            vt(v).visit_local(loc, e, v);
        }
        case (decl_item(?it)) { vt(v).visit_item(it, e, v); }
    }
}

fn visit_expr_opt[E](option::t[@expr] eo, &E e, &vt[E] v) {
    alt (eo) {
        case (none) { }
        case (some(?ex)) { vt(v).visit_expr(ex, e, v); }
    }
}

fn visit_exprs[E](vec[@expr] exprs, &E e, &vt[E] v) {
    for (@expr ex in exprs) { vt(v).visit_expr(ex, e, v); }
}

fn visit_expr[E](&@expr ex, &E e, &vt[E] v) {
    alt (ex.node) {
        case (expr_vec(?es, _, _)) { visit_exprs(es, e, v); }
        case (expr_tup(?elts)) {
            for (elt el in elts) { vt(v).visit_expr(el.expr, e, v); }
        }
        case (expr_rec(?flds, ?base)) {
            for (field f in flds) { vt(v).visit_expr(f.node.expr, e, v); }
            visit_expr_opt(base, e, v);
        }
        case (expr_call(?callee, ?args)) {
            vt(v).visit_expr(callee, e, v);
            visit_exprs(args, e, v);
        }
        case (expr_self_method(_)) { }
        case (expr_bind(?callee, ?args)) {
            vt(v).visit_expr(callee, e, v);
            for (option::t[@expr] eo in args) { visit_expr_opt(eo, e, v); }
        }
        case (expr_spawn(_, _, ?callee, ?args)) {
            vt(v).visit_expr(callee, e, v);
            visit_exprs(args, e, v);
        }
        case (expr_binary(_, ?a, ?b)) {
            vt(v).visit_expr(a, e, v);
            vt(v).visit_expr(b, e, v);
        }
        case (expr_unary(_, ?a)) { vt(v).visit_expr(a, e, v); }
        case (expr_lit(_)) { }
        case (expr_cast(?x, ?t)) {
            vt(v).visit_expr(x, e, v);
            vt(v).visit_ty(t, e, v);
        }
        case (expr_if(?x, ?b, ?eo)) {
            vt(v).visit_expr(x, e, v);
            vt(v).visit_block(b, e, v);
            visit_expr_opt(eo, e, v);
        }
        case (expr_if_check(?x, ?b, ?eo)) {
            vt(v).visit_expr(x, e, v);
            vt(v).visit_block(b, e, v);
            visit_expr_opt(eo, e, v);
        }
        case (expr_ternary(?c, ?t, ?el)) {
            vt(v).visit_expr(c, e, v);
            vt(v).visit_expr(t, e, v);
            vt(v).visit_expr(el, e, v);
        }
        case (expr_while(?x, ?b)) {
            vt(v).visit_expr(x, e, v);
            vt(v).visit_block(b, e, v);
        }
        case (expr_for(?dcl, ?x, ?b)) {
            vt(v).visit_local(dcl, e, v);
            vt(v).visit_expr(x, e, v);
            vt(v).visit_block(b, e, v);
        }
        case (expr_for_each(?dcl, ?x, ?b)) {
            vt(v).visit_local(dcl, e, v);
            vt(v).visit_expr(x, e, v);
            vt(v).visit_block(b, e, v);
        }
        case (expr_do_while(?b, ?x)) {
            vt(v).visit_block(b, e, v);
            vt(v).visit_expr(x, e, v);
        }
        case (expr_alt(?x, ?arms)) {
            vt(v).visit_expr(x, e, v);
            for (arm a in arms) { vt(v).visit_arm(a, e, v); }
        }
        case (expr_fn(?f)) {
            vt(v).visit_fn(f, [], ex.span, none, ex.id, e, v);
        }
        case (expr_block(?b)) { vt(v).visit_block(b, e, v); }
        case (expr_assign(?a, ?b)) {
            vt(v).visit_expr(b, e, v);
            vt(v).visit_expr(a, e, v);
        }
        case (expr_move(?a, ?b)) {
            vt(v).visit_expr(b, e, v);
            vt(v).visit_expr(a, e, v);
        }
        case (expr_swap(?a, ?b)) {
            vt(v).visit_expr(a, e, v);
            vt(v).visit_expr(b, e, v);
        }
        case (expr_assign_op(_, ?a, ?b)) {
            vt(v).visit_expr(b, e, v);
            vt(v).visit_expr(a, e, v);
        }
        case (expr_send(?a, ?b)) {
            vt(v).visit_expr(a, e, v);
            vt(v).visit_expr(b, e, v);
        }
        case (expr_recv(?a, ?b)) {
            vt(v).visit_expr(a, e, v);
            vt(v).visit_expr(b, e, v);
        }
        case (expr_field(?x, _)) { vt(v).visit_expr(x, e, v); }
        case (expr_index(?a, ?b)) {
            vt(v).visit_expr(a, e, v);
            vt(v).visit_expr(b, e, v);
        }
        case (expr_path(?p)) {
            for (@ty tp in p.node.types) { vt(v).visit_ty(tp, e, v); }
        }
        case (expr_ext(_, _, _, ?expansion)) {
            vt(v).visit_expr(expansion, e, v);
        }
        case (expr_fail(?eo)) {
            visit_expr_opt(eo, e, v);
        }
        case (expr_break) { }
        case (expr_cont) { }
        case (expr_ret(?eo)) { visit_expr_opt(eo, e, v); }
        case (expr_put(?eo)) { visit_expr_opt(eo, e, v); }
        case (expr_be(?x)) { vt(v).visit_expr(x, e, v); }
        case (expr_log(_, ?x)) { vt(v).visit_expr(x, e, v); }
        case (expr_check(_, ?x)) { vt(v).visit_expr(x, e, v); }
        case (expr_assert(?x)) { vt(v).visit_expr(x, e, v); }
        case (expr_port(_)) { }
        case (expr_chan(?x)) { vt(v).visit_expr(x, e, v); }
        case (expr_anon_obj(?anon_obj, _, _)) {
            alt (anon_obj.fields) {
                case (none) { }
                case (some(?fields)) {
                    for (anon_obj_field f in fields) {
                        vt(v).visit_ty(f.ty, e, v);
                        vt(v).visit_expr(f.expr, e, v);
                    }
                }
            }
            alt (anon_obj.with_obj) {
                case (none) { }
                case (some(?ex)) { vt(v).visit_expr(ex, e, v); }
            }
            for (@method m in anon_obj.methods) {
                vt(v).visit_fn(m.node.meth, [], m.span, some(m.node.ident),
                               m.node.id, e, v);
            }
        }
    }
}

fn visit_arm[E](&arm a, &E e, &vt[E] v) {
    vt(v).visit_pat(a.pat, e, v);
    vt(v).visit_block(a.block, e, v);
}
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
