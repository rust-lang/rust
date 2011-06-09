import front::ast::*;
import std::option;
import std::option::some;
import std::option::none;
import util::common::span;

// Lots of redundant indirection and refcounting. Our typesystem doesn't do
// circular types, so the visitor record can not hold functions that take
// visitors. A tag breaks the cycle.
tag vt[E] { vtor(visitor[E]); }
fn vt[E](&vt[E] x) -> visitor[E] {
    alt (x) { case (vtor(?v)) { ret v; } }
}

type visitor[E] =
    @rec(fn(&_mod m, &span sp, &E e, &vt[E] v) visit_mod,
         fn(&@native_item i, &E e, &vt[E] v)   visit_native_item,
         fn(&@item i, &E e, &vt[E] v)          visit_item,
         fn(&block b, &E e, &vt[E] v)          visit_block,
         fn(&@stmt s, &E e, &vt[E] v)          visit_stmt,
         fn(&arm a, &E e, &vt[E] v)            visit_arm,
         fn(&@pat p, &E e, &vt[E] v)           visit_pat,
         fn(&@decl d, &E e, &vt[E] v)          visit_decl,
         fn(&@expr ex, &E e, &vt[E] v)         visit_expr,
         fn(&@ty t, &E e, &vt[E] v)            visit_ty,
         fn(&_fn f, &span sp, &ident name, 
            &def_id d_id, &ann a,
            &E e, &vt[E] v)                    visit_fn);

fn default_visitor[E]() -> visitor[E] {
    ret @rec(visit_mod = bind visit_mod[E](_, _, _, _),
             visit_native_item = bind visit_native_item[E](_, _, _),
             visit_item = bind visit_item[E](_, _, _),
             visit_block = bind visit_block[E](_, _, _),
             visit_stmt = bind visit_stmt[E](_, _, _),
             visit_arm = bind visit_arm[E](_, _, _),
             visit_pat = bind visit_pat[E](_, _, _),
             visit_decl = bind visit_decl[E](_, _, _),
             visit_expr = bind visit_expr[E](_, _, _),
             visit_ty = bind visit_ty[E](_, _, _),
             visit_fn = bind visit_fn[E](_, _, _, _, _, _, _));
}

fn visit_crate[E](&crate c, &E e, &vt[E] v) {
    vt(v).visit_mod(c.node.module, c.span, e, v);
}

fn visit_mod[E](&_mod m, &span sp, &E e, &vt[E] v) {
    for (@item i in m.items) {
        vt(v).visit_item(i, e, v);
    }
}

fn visit_item[E](&@item i, &E e, &vt[E] v) {
    alt (i.node) {
        case (item_const(_, ?t, ?ex, _, _)) {
            vt(v).visit_ty(t, e, v);
            vt(v).visit_expr(ex, e, v);
        }
        case (item_fn(?nm, ?f, _, ?d, ?a)) {
            vt(v).visit_fn(f, i.span, nm, d, a, e, v);
        }
        case (item_mod(_, ?m, _)) {
            vt(v).visit_mod(m, i.span, e, v);
        }
        case (item_native_mod(_, ?nm, _)) {
            for (@native_item ni in nm.items) {
                vt(v).visit_native_item(ni, e, v);
            }
        }
        case (item_ty(_, ?t, _, _, _)) {
            vt(v).visit_ty(t, e, v);
        }
        case (item_tag(_, ?variants, _, _, _)) {
            for (variant vr in variants) {
                for (variant_arg va in vr.node.args) {
                    vt(v).visit_ty(va.ty, e, v);
                }
            }
        }
        case (item_obj(_, ?ob, _, _, _)) {
            for (obj_field f in ob.fields) {
                vt(v).visit_ty(f.ty, e, v);
            }
            for (@method m in ob.methods) {
                vt(v).visit_fn(m.node.meth, m.span, m.node.ident, m.node.id,
                           m.node.ann, e, v);
            }
            alt (ob.dtor) {
                case (none) {}
                case (some(?m)) {
                    vt(v).visit_fn(m.node.meth, m.span, m.node.ident,
                                   m.node.id, m.node.ann, e, v);
                }
            }
        }

    }
}

fn visit_ty[E](&@ty t, &E e, &vt[E] v) {
    alt (t.node) {
        case (ty_box(?mt)) { vt(v).visit_ty(mt.ty, e, v); }
        case (ty_vec(?mt)) { vt(v).visit_ty(mt.ty, e, v); }
        case (ty_ptr(?mt)) { vt(v).visit_ty(mt.ty, e, v); }
        case (ty_port(?t)) { vt(v).visit_ty(t, e, v); }
        case (ty_chan(?t)) { vt(v).visit_ty(t, e, v); }
        case (ty_tup(?mts)) {
            for (mt mt in mts) {
                vt(v).visit_ty(mt.ty, e, v);
            }
        }
        case (ty_rec(?flds)) {
            for (ty_field f in flds) {
                vt(v).visit_ty(f.node.mt.ty, e, v);
            }
        }
        case (ty_fn(_, ?args, ?out, _, _)) {
            for (ty_arg a in args) {
                vt(v).visit_ty(a.node.ty, e, v);
            }
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
            for (@ty tp in p.node.types) {
                vt(v).visit_ty(tp, e, v);
            }
        }
        case (ty_constr(?t, _)) { vt(v).visit_ty(t, e, v); }
        case (_) {}
    }
}

fn visit_pat[E](&@pat p, &E e, &vt[E] v) {
    alt (p.node) {
        case (pat_tag(?path, ?children, _)) {
            for (@pat child in children) {
                vt(v).visit_pat(child, e, v);
            }
        }
        case (_) {}
    }
}

fn visit_native_item[E](&@native_item ni, &E e, &vt[E] v) {
    alt (ni.node) {
        case (native_item_fn(_, _, ?fd, _, _, _)) {
            visit_fn_decl(fd, e, v);
        }
        case (native_item_ty(_, _)) {}
    }
}

fn visit_fn_decl[E](&fn_decl fd, &E e, &vt[E] v) {
    for (arg a in fd.inputs) {
        vt(v).visit_ty(a.ty, e, v);
    }
    vt(v).visit_ty(fd.output, e, v);
}

fn visit_fn[E](&_fn f, &span sp, &ident i, &def_id d, &ann a,
               &E e, &vt[E] v) {
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
        case (stmt_crate_directive(?cdir)) {}
    }
}

fn visit_decl[E](&@decl d, &E e, &vt[E] v) {
    alt (d.node) {
        case (decl_local(?loc)) {
            alt (loc.ty) {
                case (none) {}
                case (some(?t)) { vt(v).visit_ty(t, e, v); }
            }
            alt (loc.init) {
                case (none) {}
                case (some(?i)) { vt(v).visit_expr(i.expr, e, v); }
            }
        }
        case (decl_item(?it)) { vt(v).visit_item(it, e, v); }
    }
}

fn visit_expr_opt[E](option::t[@expr] eo, &E e, &vt[E] v) {
    alt (eo) {
        case (none) {}
        case (some(?ex)) { vt(v).visit_expr(ex, e, v);
        }
    }
}

fn visit_exprs[E](vec[@expr] exprs, &E e, &vt[E] v) {
    for (@expr ex in exprs) { vt(v).visit_expr(ex, e, v); }
}

fn visit_expr[E](&@expr ex, &E e, &vt[E] v) {
    alt (ex.node) {
        case (expr_vec(?es, _, _)) {
            visit_exprs(es, e, v);
        }
        case (expr_tup(?elts, _)) {
            for (elt el in elts) { vt(v).visit_expr(el.expr, e, v); }
        }
        case (expr_rec(?flds, ?base, _)) {
            for (field f in flds) { vt(v).visit_expr(f.node.expr, e, v); }
            visit_expr_opt(base, e, v);
        }
        case (expr_call(?callee, ?args, _)) {
            vt(v).visit_expr(callee, e, v);
            visit_exprs(args, e, v);
        }
        case (expr_self_method(_, _)) { }
        case (expr_bind(?callee, ?args, _)) {
            vt(v).visit_expr(callee, e, v);
            for (option::t[@expr] eo in args) { visit_expr_opt(eo, e, v); }
        }
        case (expr_spawn(_, _, ?callee, ?args, _)) {
            vt(v).visit_expr(callee, e, v);
            visit_exprs(args, e, v);
        }
        case (expr_binary(_, ?a, ?b, _)) {
            vt(v).visit_expr(a, e, v);
            vt(v).visit_expr(b, e, v);
        }
        case (expr_unary(_, ?a, _)) {
            vt(v).visit_expr(a, e, v);
        }
        case (expr_lit(_, _)) { }
        case (expr_cast(?x, ?t, _)) {
            vt(v).visit_expr(x, e, v);
            vt(v).visit_ty(t, e, v);
        }
        case (expr_if(?x, ?b, ?eo, _)) {
            vt(v).visit_expr(x, e, v);
            vt(v).visit_block(b, e, v);
            visit_expr_opt(eo, e, v);
        }
        case (expr_while(?x, ?b, _)) {
            vt(v).visit_expr(x, e, v);
            vt(v).visit_block(b, e, v);
        }
        case (expr_for(?dcl, ?x, ?b, _)) {
            vt(v).visit_decl(dcl, e, v);
            vt(v).visit_expr(x, e, v);
            vt(v).visit_block(b, e, v);
        }
        case (expr_for_each(?dcl, ?x, ?b, _)) {
            vt(v).visit_decl(dcl, e, v);
            vt(v).visit_expr(x, e, v);
            vt(v).visit_block(b, e, v);
        }
        case (expr_do_while(?b, ?x, _)) {
            vt(v).visit_block(b, e, v);
            vt(v).visit_expr(x, e, v);
        }
        case (expr_alt(?x, ?arms, _)) {
            vt(v).visit_expr(x, e, v);
            for (arm a in arms) {
                vt(v).visit_arm(a, e, v);
            }
        }
        case (expr_block(?b, _)) {
            vt(v).visit_block(b, e, v);
        }
        case (expr_assign(?a, ?b, _)) {
            vt(v).visit_expr(b, e, v);
            vt(v).visit_expr(a, e, v);
        }
        case (expr_move(?a, ?b, _)) {
            vt(v).visit_expr(b, e, v);
            vt(v).visit_expr(a, e, v);
        }
        case (expr_assign_op(_, ?a, ?b, _)) {
            vt(v).visit_expr(b, e, v);
            vt(v).visit_expr(a, e, v);
        }
        case (expr_send(?a, ?b, _)) {
            vt(v).visit_expr(a, e, v);
            vt(v).visit_expr(b, e, v);
        }
        case (expr_recv(?a, ?b, _)) {
            vt(v).visit_expr(a, e, v);
            vt(v).visit_expr(b, e, v);
        }
        case (expr_field(?x, _, _)) {
            vt(v).visit_expr(x, e, v);
        }
        case (expr_index(?a, ?b, _)) {
            vt(v).visit_expr(a, e, v);
            vt(v).visit_expr(b, e, v);
        }
        case (expr_path(?p, _)) {
            for (@ty tp in p.node.types) {
                vt(v).visit_ty(tp, e, v);
            }
        }
        case (expr_ext(_, _, _, ?expansion, _)) {
            vt(v).visit_expr(expansion, e, v);
        }
        case (expr_fail(_, _)) { }
        case (expr_break(_)) { }
        case (expr_cont(_)) { }
        case (expr_ret(?eo, _)) {
            visit_expr_opt(eo, e, v);
        }
        case (expr_put(?eo, _)) {
            visit_expr_opt(eo, e, v);
        }
        case (expr_be(?x, _)) {
            vt(v).visit_expr(x, e, v);
        }
        case (expr_log(_,?x, _)) {
            vt(v).visit_expr(x, e, v);
        }
        case (expr_check(?x, _)) {
            vt(v).visit_expr(x, e, v);
        }
        case (expr_assert(?x, _)) {
            vt(v).visit_expr(x, e, v);
        }
        case (expr_port(_)) { }
        case (expr_chan(?x, _)) {
            vt(v).visit_expr(x, e, v);
        }

        case (expr_anon_obj(?anon_obj,_,_,_)) { 
            alt (anon_obj.fields) {
                case (none) { }
                case (some(?fields)) {
                    for (obj_field f in fields) {
                        vt(v).visit_ty(f.ty, e, v);
                    }
                }
            }
            alt (anon_obj.with_obj) {
                case (none) { }
                case (some(?ex)) {
                    vt(v).visit_expr(ex, e, v);
                }
            }
            for (@method m in anon_obj.methods) {
                vt(v).visit_fn(m.node.meth, m.span, m.node.ident, 
                           m.node.id, m.node.ann, e, v);
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
