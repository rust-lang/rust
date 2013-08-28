// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use abi::AbiSet;
use ast::*;
use ast;
use codemap::span;
use parse;
use opt_vec;
use opt_vec::OptVec;

// Context-passing AST walker. Each overridden visit method has full control
// over what happens with its node, it can do its own traversal of the node's
// children (potentially passing in different contexts to each), call
// visit::visit_* to apply the default traversal algorithm (again, it can
// override the context), or prevent deeper traversal by doing nothing.
//
// Note: it is an important invariant that the default visitor walks the body
// of a function in "execution order" (more concretely, reverse post-order
// with respect to the CFG implied by the AST), meaning that if AST node A may
// execute before AST node B, then A is visited first.  The borrow checker in
// particular relies on this property.

// Our typesystem doesn't do circular types, so the visitor record can not
// hold functions that take visitors. A vt enum is used to break the cycle.
pub enum vt<E> { mk_vt(visitor<E>), }

pub enum fn_kind<'self> {
    // fn foo() or extern "Abi" fn foo()
    fk_item_fn(ident, &'self Generics, purity, AbiSet),

    // fn foo(&self)
    fk_method(ident, &'self Generics, &'self method),

    // @fn(x, y) { ... }
    fk_anon(ast::Sigil),

    // |x, y| ...
    fk_fn_block,
}

pub fn name_of_fn(fk: &fn_kind) -> ident {
    match *fk {
      fk_item_fn(name, _, _, _) | fk_method(name, _, _) => {
          name
      }
      fk_anon(*) | fk_fn_block(*) => parse::token::special_idents::anon,
    }
}

pub fn generics_of_fn(fk: &fn_kind) -> Generics {
    match *fk {
        fk_item_fn(_, generics, _, _) |
        fk_method(_, generics, _) => {
            (*generics).clone()
        }
        fk_anon(*) | fk_fn_block(*) => {
            Generics {
                lifetimes: opt_vec::Empty,
                ty_params: opt_vec::Empty,
            }
        }
    }
}

pub struct Visitor<E> {
    visit_mod: @fn(&_mod, span, NodeId, (E, vt<E>)),
    visit_view_item: @fn(&view_item, (E, vt<E>)),
    visit_foreign_item: @fn(@foreign_item, (E, vt<E>)),
    visit_item: @fn(@item, (E, vt<E>)),
    visit_local: @fn(@Local, (E, vt<E>)),
    visit_block: @fn(&Block, (E, vt<E>)),
    visit_stmt: @fn(@stmt, (E, vt<E>)),
    visit_arm: @fn(&arm, (E, vt<E>)),
    visit_pat: @fn(@pat, (E, vt<E>)),
    visit_decl: @fn(@decl, (E, vt<E>)),
    visit_expr: @fn(@expr, (E, vt<E>)),
    visit_expr_post: @fn(@expr, (E, vt<E>)),
    visit_ty: @fn(&Ty, (E, vt<E>)),
    visit_generics: @fn(&Generics, (E, vt<E>)),
    visit_fn: @fn(&fn_kind, &fn_decl, &Block, span, NodeId, (E, vt<E>)),
    visit_ty_method: @fn(&TypeMethod, (E, vt<E>)),
    visit_trait_method: @fn(&trait_method, (E, vt<E>)),
    visit_struct_def: @fn(@struct_def, ident, &Generics, NodeId, (E, vt<E>)),
    visit_struct_field: @fn(@struct_field, (E, vt<E>)),
}

pub type visitor<E> = @Visitor<E>;

pub fn default_visitor<E:Clone>() -> visitor<E> {
    return @Visitor {
        visit_mod: |a,b,c,d|visit_mod::<E>(a, b, c, d),
        visit_view_item: |a,b|visit_view_item::<E>(a, b),
        visit_foreign_item: |a,b|visit_foreign_item::<E>(a, b),
        visit_item: |a,b|visit_item::<E>(a, b),
        visit_local: |a,b|visit_local::<E>(a, b),
        visit_block: |a,b|visit_block::<E>(a, b),
        visit_stmt: |a,b|visit_stmt::<E>(a, b),
        visit_arm: |a,b|visit_arm::<E>(a, b),
        visit_pat: |a,b|visit_pat::<E>(a, b),
        visit_decl: |a,b|visit_decl::<E>(a, b),
        visit_expr: |a,b|visit_expr::<E>(a, b),
        visit_expr_post: |_a,_b| (),
        visit_ty: |a,b|skip_ty::<E>(a, b),
        visit_generics: |a,b|visit_generics::<E>(a, b),
        visit_fn: |a,b,c,d,e,f|visit_fn::<E>(a, b, c, d, e, f),
        visit_ty_method: |a,b|visit_ty_method::<E>(a, b),
        visit_trait_method: |a,b|visit_trait_method::<E>(a, b),
        visit_struct_def: |a,b,c,d,e|visit_struct_def::<E>(a, b, c, d, e),
        visit_struct_field: |a,b|visit_struct_field::<E>(a, b),
    };
}

pub fn visit_crate<E:Clone>(c: &Crate, (e, v): (E, vt<E>)) {
    (v.visit_mod)(&c.module, c.span, CRATE_NODE_ID, (e, v));
}

pub fn visit_mod<E:Clone>(m: &_mod,
                          _sp: span,
                          _id: NodeId,
                          (e, v): (E, vt<E>)) {
    for vi in m.view_items.iter() {
        (v.visit_view_item)(vi, (e.clone(), v));
    }
    for i in m.items.iter() {
        (v.visit_item)(*i, (e.clone(), v));
    }
}

pub fn visit_view_item<E>(_vi: &view_item, (_e, _v): (E, vt<E>)) { }

pub fn visit_local<E:Clone>(loc: &Local, (e, v): (E, vt<E>)) {
    (v.visit_pat)(loc.pat, (e.clone(), v));
    (v.visit_ty)(&loc.ty, (e.clone(), v));
    match loc.init {
      None => (),
      Some(ex) => (v.visit_expr)(ex, (e, v))
    }
}

fn visit_trait_ref<E:Clone>(tref: &ast::trait_ref, (e, v): (E, vt<E>)) {
    visit_path(&tref.path, (e, v));
}

pub fn visit_item<E:Clone>(i: &item, (e, v): (E, vt<E>)) {
    match i.node {
        item_static(ref t, _, ex) => {
            (v.visit_ty)(t, (e.clone(), v));
            (v.visit_expr)(ex, (e.clone(), v));
        }
        item_fn(ref decl, purity, abi, ref generics, ref body) => {
            (v.visit_fn)(
                &fk_item_fn(
                    i.ident,
                    generics,
                    purity,
                    abi
                ),
                decl,
                body,
                i.span,
                i.id,
                (e,
                 v)
            );
        }
        item_mod(ref m) => (v.visit_mod)(m, i.span, i.id, (e, v)),
        item_foreign_mod(ref nm) => {
            for vi in nm.view_items.iter() {
                (v.visit_view_item)(vi, (e.clone(), v));
            }
            for ni in nm.items.iter() {
                (v.visit_foreign_item)(*ni, (e.clone(), v));
            }
        }
        item_ty(ref t, ref tps) => {
            (v.visit_ty)(t, (e.clone(), v));
            (v.visit_generics)(tps, (e, v));
        }
        item_enum(ref enum_definition, ref tps) => {
            (v.visit_generics)(tps, (e.clone(), v));
            visit_enum_def(
                enum_definition,
                tps,
                (e, v)
            );
        }
        item_impl(ref tps, ref traits, ref ty, ref methods) => {
            (v.visit_generics)(tps, (e.clone(), v));
            for p in traits.iter() {
                visit_trait_ref(p, (e.clone(), v));
            }
            (v.visit_ty)(ty, (e.clone(), v));
            for m in methods.iter() {
                visit_method_helper(*m, (e.clone(), v))
            }
        }
        item_struct(struct_def, ref generics) => {
            (v.visit_generics)(generics, (e.clone(), v));
            (v.visit_struct_def)(struct_def, i.ident, generics, i.id, (e, v));
        }
        item_trait(ref generics, ref traits, ref methods) => {
            (v.visit_generics)(generics, (e.clone(), v));
            for p in traits.iter() {
                visit_path(&p.path, (e.clone(), v));
            }
            for m in methods.iter() {
                (v.visit_trait_method)(m, (e.clone(), v));
            }
        }
        item_mac(ref m) => visit_mac(m, (e, v))
    }
}

pub fn visit_enum_def<E:Clone>(enum_definition: &ast::enum_def,
                               tps: &Generics,
                               (e, v): (E, vt<E>)) {
    for vr in enum_definition.variants.iter() {
        match vr.node.kind {
            tuple_variant_kind(ref variant_args) => {
                for va in variant_args.iter() {
                    (v.visit_ty)(&va.ty, (e.clone(), v));
                }
            }
            struct_variant_kind(struct_def) => {
                (v.visit_struct_def)(struct_def, vr.node.name, tps,
                                     vr.node.id, (e.clone(), v));
            }
        }
        // Visit the disr expr if it exists
        for ex in vr.node.disr_expr.iter() {
            (v.visit_expr)(*ex, (e.clone(), v))
        }
    }
}

pub fn skip_ty<E>(_t: &Ty, (_e,_v): (E, vt<E>)) {}

pub fn visit_ty<E:Clone>(t: &Ty, (e, v): (E, vt<E>)) {
    match t.node {
        ty_box(ref mt) | ty_uniq(ref mt) |
        ty_vec(ref mt) | ty_ptr(ref mt) | ty_rptr(_, ref mt) => {
            (v.visit_ty)(mt.ty, (e, v));
        },
        ty_tup(ref ts) => {
            for tt in ts.iter() {
                (v.visit_ty)(tt, (e.clone(), v));
            }
        },
        ty_closure(ref f) => {
            for a in f.decl.inputs.iter() {
                (v.visit_ty)(&a.ty, (e.clone(), v));
            }
            (v.visit_ty)(&f.decl.output, (e.clone(), v));
            do f.bounds.map |bounds| {
                visit_ty_param_bounds(bounds, (e.clone(), v));
            };
        },
        ty_bare_fn(ref f) => {
            for a in f.decl.inputs.iter() {
                (v.visit_ty)(&a.ty, (e.clone(), v));
            }
            (v.visit_ty)(&f.decl.output, (e, v));
        },
        ty_path(ref p, ref bounds, _) => {
            visit_path(p, (e.clone(), v));
            do bounds.map |bounds| {
                visit_ty_param_bounds(bounds, (e.clone(), v));
            };
        },
        ty_fixed_length_vec(ref mt, ex) => {
            (v.visit_ty)(mt.ty, (e.clone(), v));
            (v.visit_expr)(ex, (e.clone(), v));
        },
        ty_nil | ty_bot | ty_mac(_) | ty_infer => ()
    }
}

pub fn visit_path<E:Clone>(p: &Path, (e, v): (E, vt<E>)) {
    for segment in p.segments.iter() {
        for typ in segment.types.iter() {
            (v.visit_ty)(typ, (e.clone(), v))
        }
    }
}

pub fn visit_pat<E:Clone>(p: &pat, (e, v): (E, vt<E>)) {
    match p.node {
        pat_enum(ref path, ref children) => {
            visit_path(path, (e.clone(), v));
            for children in children.iter() {
                for child in children.iter() {
                    (v.visit_pat)(*child, (e.clone(), v));
                }
            }
        }
        pat_struct(ref path, ref fields, _) => {
            visit_path(path, (e.clone(), v));
            for f in fields.iter() {
                (v.visit_pat)(f.pat, (e.clone(), v));
            }
        }
        pat_tup(ref elts) => {
            for elt in elts.iter() {
                (v.visit_pat)(*elt, (e.clone(), v))
            }
        },
        pat_box(inner) | pat_uniq(inner) | pat_region(inner) => {
            (v.visit_pat)(inner, (e, v))
        },
        pat_ident(_, ref path, ref inner) => {
            visit_path(path, (e.clone(), v));
            for subpat in inner.iter() {
                (v.visit_pat)(*subpat, (e.clone(), v))
            }
        }
        pat_lit(ex) => (v.visit_expr)(ex, (e, v)),
        pat_range(e1, e2) => {
            (v.visit_expr)(e1, (e.clone(), v));
            (v.visit_expr)(e2, (e, v));
        }
        pat_wild => (),
        pat_vec(ref before, ref slice, ref after) => {
            for elt in before.iter() {
                (v.visit_pat)(*elt, (e.clone(), v));
            }
            for elt in slice.iter() {
                (v.visit_pat)(*elt, (e.clone(), v));
            }
            for tail in after.iter() {
                (v.visit_pat)(*tail, (e.clone(), v));
            }
        }
    }
}

pub fn visit_foreign_item<E:Clone>(ni: &foreign_item, (e, v): (E, vt<E>)) {
    match ni.node {
        foreign_item_fn(ref fd, ref generics) => {
            visit_fn_decl(fd, (e.clone(), v));
            (v.visit_generics)(generics, (e, v));
        }
        foreign_item_static(ref t, _) => {
            (v.visit_ty)(t, (e, v));
        }
    }
}

pub fn visit_ty_param_bounds<E:Clone>(bounds: &OptVec<TyParamBound>,
                                      (e, v): (E, vt<E>)) {
    for bound in bounds.iter() {
        match *bound {
            TraitTyParamBound(ref ty) => visit_trait_ref(ty, (e.clone(), v)),
            RegionTyParamBound => {}
        }
    }
}

pub fn visit_generics<E:Clone>(generics: &Generics, (e, v): (E, vt<E>)) {
    for tp in generics.ty_params.iter() {
        visit_ty_param_bounds(&tp.bounds, (e.clone(), v));
    }
}

pub fn visit_fn_decl<E:Clone>(fd: &fn_decl, (e, v): (E, vt<E>)) {
    for a in fd.inputs.iter() {
        (v.visit_pat)(a.pat, (e.clone(), v));
        (v.visit_ty)(&a.ty, (e.clone(), v));
    }
    (v.visit_ty)(&fd.output, (e, v));
}

// Note: there is no visit_method() method in the visitor, instead override
// visit_fn() and check for fk_method().  I named this visit_method_helper()
// because it is not a default impl of any method, though I doubt that really
// clarifies anything. - Niko
pub fn visit_method_helper<E:Clone>(m: &method, (e, v): (E, vt<E>)) {
    (v.visit_fn)(&fk_method(m.ident, &m.generics, m),
                 &m.decl,
                 &m.body,
                 m.span,
                 m.id,
                 (e, v));
}

pub fn visit_fn<E:Clone>(fk: &fn_kind,
                         decl: &fn_decl,
                         body: &Block,
                         _sp: span,
                         _id: NodeId,
                         (e, v): (E, vt<E>)) {
    visit_fn_decl(decl, (e.clone(), v));
    let generics = generics_of_fn(fk);
    (v.visit_generics)(&generics, (e.clone(), v));
    (v.visit_block)(body, (e, v));
}

pub fn visit_ty_method<E:Clone>(m: &TypeMethod, (e, v): (E, vt<E>)) {
    for a in m.decl.inputs.iter() {
        (v.visit_ty)(&a.ty, (e.clone(), v));
    }
    (v.visit_generics)(&m.generics, (e.clone(), v));
    (v.visit_ty)(&m.decl.output, (e, v));
}

pub fn visit_trait_method<E:Clone>(m: &trait_method, (e, v): (E, vt<E>)) {
    match *m {
      required(ref ty_m) => (v.visit_ty_method)(ty_m, (e, v)),
      provided(m) => visit_method_helper(m, (e, v))
    }
}

pub fn visit_struct_def<E:Clone>(
    sd: @struct_def,
    _nm: ast::ident,
    _generics: &Generics,
    _id: NodeId,
    (e, v): (E, vt<E>)
) {
    for f in sd.fields.iter() {
        (v.visit_struct_field)(*f, (e.clone(), v));
    }
}

pub fn visit_struct_field<E:Clone>(sf: &struct_field, (e, v): (E, vt<E>)) {
    (v.visit_ty)(&sf.node.ty, (e, v));
}

pub fn visit_block<E:Clone>(b: &Block, (e, v): (E, vt<E>)) {
    for vi in b.view_items.iter() {
        (v.visit_view_item)(vi, (e.clone(), v));
    }
    for s in b.stmts.iter() {
        (v.visit_stmt)(*s, (e.clone(), v));
    }
    visit_expr_opt(b.expr, (e, v));
}

pub fn visit_stmt<E>(s: &stmt, (e, v): (E, vt<E>)) {
    match s.node {
      stmt_decl(d, _) => (v.visit_decl)(d, (e, v)),
      stmt_expr(ex, _) => (v.visit_expr)(ex, (e, v)),
      stmt_semi(ex, _) => (v.visit_expr)(ex, (e, v)),
      stmt_mac(ref mac, _) => visit_mac(mac, (e, v))
    }
}

pub fn visit_decl<E:Clone>(d: &decl, (e, v): (E, vt<E>)) {
    match d.node {
        decl_local(ref loc) => (v.visit_local)(*loc, (e, v)),
        decl_item(it) => (v.visit_item)(it, (e, v))
    }
}

pub fn visit_expr_opt<E>(eo: Option<@expr>, (e, v): (E, vt<E>)) {
    match eo { None => (), Some(ex) => (v.visit_expr)(ex, (e, v)) }
}

pub fn visit_exprs<E:Clone>(exprs: &[@expr], (e, v): (E, vt<E>)) {
    for ex in exprs.iter() { (v.visit_expr)(*ex, (e.clone(), v)); }
}

pub fn visit_mac<E>(_m: &mac, (_e, _v): (E, vt<E>)) {
    /* no user-serviceable parts inside */
}

pub fn visit_expr<E:Clone>(ex: @expr, (e, v): (E, vt<E>)) {
    match ex.node {
        expr_vstore(x, _) => (v.visit_expr)(x, (e.clone(), v)),
        expr_vec(ref es, _) => visit_exprs(*es, (e.clone(), v)),
        expr_repeat(element, count, _) => {
            (v.visit_expr)(element, (e.clone(), v));
            (v.visit_expr)(count, (e.clone(), v));
        }
        expr_struct(ref p, ref flds, base) => {
            visit_path(p, (e.clone(), v));
            for f in flds.iter() {
                (v.visit_expr)(f.expr, (e.clone(), v));
            }
            visit_expr_opt(base, (e.clone(), v));
        }
        expr_tup(ref elts) => {
            for el in elts.iter() { (v.visit_expr)(*el, (e.clone(), v)) }
        }
        expr_call(callee, ref args, _) => {
            visit_exprs(*args, (e.clone(), v));
            (v.visit_expr)(callee, (e.clone(), v));
        }
        expr_method_call(_, callee, _, ref tys, ref args, _) => {
            visit_exprs(*args, (e.clone(), v));
            for tp in tys.iter() {
                (v.visit_ty)(tp, (e.clone(), v));
            }
            (v.visit_expr)(callee, (e.clone(), v));
        }
        expr_binary(_, _, a, b) => {
            (v.visit_expr)(a, (e.clone(), v));
            (v.visit_expr)(b, (e.clone(), v));
        }
        expr_addr_of(_, x) | expr_unary(_, _, x) |
        expr_do_body(x) => (v.visit_expr)(x, (e.clone(), v)),
        expr_lit(_) => (),
        expr_cast(x, ref t) => {
            (v.visit_expr)(x, (e.clone(), v));
            (v.visit_ty)(t, (e.clone(), v));
        }
        expr_if(x, ref b, eo) => {
            (v.visit_expr)(x, (e.clone(), v));
            (v.visit_block)(b, (e.clone(), v));
            visit_expr_opt(eo, (e.clone(), v));
        }
        expr_while(x, ref b) => {
            (v.visit_expr)(x, (e.clone(), v));
            (v.visit_block)(b, (e.clone(), v));
        }
        expr_for_loop(pattern, subexpression, ref block) => {
            (v.visit_pat)(pattern, (e.clone(), v));
            (v.visit_expr)(subexpression, (e.clone(), v));
            (v.visit_block)(block, (e.clone(), v))
        }
        expr_loop(ref b, _) => (v.visit_block)(b, (e.clone(), v)),
        expr_match(x, ref arms) => {
            (v.visit_expr)(x, (e.clone(), v));
            for a in arms.iter() { (v.visit_arm)(a, (e.clone(), v)); }
        }
        expr_fn_block(ref decl, ref body) => {
            (v.visit_fn)(
                &fk_fn_block,
                decl,
                body,
                ex.span,
                ex.id,
                (e.clone(), v)
            );
        }
        expr_block(ref b) => (v.visit_block)(b, (e.clone(), v)),
        expr_assign(a, b) => {
            (v.visit_expr)(b, (e.clone(), v));
            (v.visit_expr)(a, (e.clone(), v));
        }
        expr_assign_op(_, _, a, b) => {
            (v.visit_expr)(b, (e.clone(), v));
            (v.visit_expr)(a, (e.clone(), v));
        }
        expr_field(x, _, ref tys) => {
            (v.visit_expr)(x, (e.clone(), v));
            for tp in tys.iter() {
                (v.visit_ty)(tp, (e.clone(), v));
            }
        }
        expr_index(_, a, b) => {
            (v.visit_expr)(a, (e.clone(), v));
            (v.visit_expr)(b, (e.clone(), v));
        }
        expr_path(ref p) => visit_path(p, (e.clone(), v)),
        expr_self => (),
        expr_break(_) => (),
        expr_again(_) => (),
        expr_ret(eo) => visit_expr_opt(eo, (e.clone(), v)),
        expr_log(lv, x) => {
            (v.visit_expr)(lv, (e.clone(), v));
            (v.visit_expr)(x, (e.clone(), v));
        }
        expr_mac(ref mac) => visit_mac(mac, (e.clone(), v)),
        expr_paren(x) => (v.visit_expr)(x, (e.clone(), v)),
        expr_inline_asm(ref a) => {
            for &(_, input) in a.inputs.iter() {
                (v.visit_expr)(input, (e.clone(), v));
            }
            for &(_, out) in a.outputs.iter() {
                (v.visit_expr)(out, (e.clone(), v));
            }
        }
    }
    (v.visit_expr_post)(ex, (e, v));
}

pub fn visit_arm<E:Clone>(a: &arm, (e, v): (E, vt<E>)) {
    for p in a.pats.iter() { (v.visit_pat)(*p, (e.clone(), v)); }
    visit_expr_opt(a.guard, (e.clone(), v));
    (v.visit_block)(&a.body, (e.clone(), v));
}

// Simpler, non-context passing interface. Always walks the whole tree, simply
// calls the given functions on the nodes.

pub struct SimpleVisitor {
    visit_mod: @fn(&_mod, span, NodeId),
    visit_view_item: @fn(&view_item),
    visit_foreign_item: @fn(@foreign_item),
    visit_item: @fn(@item),
    visit_local: @fn(@Local),
    visit_block: @fn(&Block),
    visit_stmt: @fn(@stmt),
    visit_arm: @fn(&arm),
    visit_pat: @fn(@pat),
    visit_decl: @fn(@decl),
    visit_expr: @fn(@expr),
    visit_expr_post: @fn(@expr),
    visit_ty: @fn(&Ty),
    visit_generics: @fn(&Generics),
    visit_fn: @fn(&fn_kind, &fn_decl, &Block, span, NodeId),
    visit_ty_method: @fn(&TypeMethod),
    visit_trait_method: @fn(&trait_method),
    visit_struct_def: @fn(@struct_def, ident, &Generics, NodeId),
    visit_struct_field: @fn(@struct_field),
    visit_struct_method: @fn(@method)
}

pub type simple_visitor = @SimpleVisitor;

pub fn simple_ignore_ty(_t: &Ty) {}

pub fn default_simple_visitor() -> @SimpleVisitor {
    @SimpleVisitor {
        visit_mod: |_m, _sp, _id| { },
        visit_view_item: |_vi| { },
        visit_foreign_item: |_ni| { },
        visit_item: |_i| { },
        visit_local: |_l| { },
        visit_block: |_b| { },
        visit_stmt: |_s| { },
        visit_arm: |_a| { },
        visit_pat: |_p| { },
        visit_decl: |_d| { },
        visit_expr: |_e| { },
        visit_expr_post: |_e| { },
        visit_ty: simple_ignore_ty,
        visit_generics: |_| {},
        visit_fn: |_, _, _, _, _| {},
        visit_ty_method: |_| {},
        visit_trait_method: |_| {},
        visit_struct_def: |_, _, _, _| {},
        visit_struct_field: |_| {},
        visit_struct_method: |_| {},
    }
}

pub fn mk_simple_visitor(v: simple_visitor) -> vt<()> {
    fn v_mod(
        f: @fn(&_mod, span, NodeId),
        m: &_mod,
        sp: span,
        id: NodeId,
        (e, v): ((), vt<()>)
    ) {
        f(m, sp, id);
        visit_mod(m, sp, id, (e, v));
    }
    fn v_view_item(f: @fn(&view_item), vi: &view_item, (e, v): ((), vt<()>)) {
        f(vi);
        visit_view_item(vi, (e, v));
    }
    fn v_foreign_item(f: @fn(@foreign_item), ni: @foreign_item, (e, v): ((), vt<()>)) {
        f(ni);
        visit_foreign_item(ni, (e, v));
    }
    fn v_item(f: @fn(@item), i: @item, (e, v): ((), vt<()>)) {
        f(i);
        visit_item(i, (e, v));
    }
    fn v_local(f: @fn(@Local), l: @Local, (e, v): ((), vt<()>)) {
        f(l);
        visit_local(l, (e, v));
    }
    fn v_block(f: @fn(&ast::Block), bl: &ast::Block, (e, v): ((), vt<()>)) {
        f(bl);
        visit_block(bl, (e, v));
    }
    fn v_stmt(f: @fn(@stmt), st: @stmt, (e, v): ((), vt<()>)) {
        f(st);
        visit_stmt(st, (e, v));
    }
    fn v_arm(f: @fn(&arm), a: &arm, (e, v): ((), vt<()>)) {
        f(a);
        visit_arm(a, (e, v));
    }
    fn v_pat(f: @fn(@pat), p: @pat, (e, v): ((), vt<()>)) {
        f(p);
        visit_pat(p, (e, v));
    }
    fn v_decl(f: @fn(@decl), d: @decl, (e, v): ((), vt<()>)) {
        f(d);
        visit_decl(d, (e, v));
    }
    fn v_expr(f: @fn(@expr), ex: @expr, (e, v): ((), vt<()>)) {
        f(ex);
        visit_expr(ex, (e, v));
    }
    fn v_expr_post(f: @fn(@expr), ex: @expr, (_e, _v): ((), vt<()>)) {
        f(ex);
    }
    fn v_ty(f: @fn(&Ty), ty: &Ty, (e, v): ((), vt<()>)) {
        f(ty);
        visit_ty(ty, (e, v));
    }
    fn v_ty_method(f: @fn(&TypeMethod), ty: &TypeMethod, (e, v): ((), vt<()>)) {
        f(ty);
        visit_ty_method(ty, (e, v));
    }
    fn v_trait_method(f: @fn(&trait_method),
                      m: &trait_method,
                      (e, v): ((), vt<()>)) {
        f(m);
        visit_trait_method(m, (e, v));
    }
    fn v_struct_def(
        f: @fn(@struct_def, ident, &Generics, NodeId),
        sd: @struct_def,
        nm: ident,
        generics: &Generics,
        id: NodeId,
        (e, v): ((), vt<()>)
    ) {
        f(sd, nm, generics, id);
        visit_struct_def(sd, nm, generics, id, (e, v));
    }
    fn v_generics(
        f: @fn(&Generics),
        ps: &Generics,
        (e, v): ((), vt<()>)
    ) {
        f(ps);
        visit_generics(ps, (e, v));
    }
    fn v_fn(
        f: @fn(&fn_kind, &fn_decl, &Block, span, NodeId),
        fk: &fn_kind,
        decl: &fn_decl,
        body: &Block,
        sp: span,
        id: NodeId,
        (e, v): ((), vt<()>)
    ) {
        f(fk, decl, body, sp, id);
        visit_fn(fk, decl, body, sp, id, (e, v));
    }
    let visit_ty: @fn(&Ty, ((), vt<()>)) =
        |a,b| v_ty(v.visit_ty, a, b);
    fn v_struct_field(f: @fn(@struct_field), sf: @struct_field, (e, v): ((), vt<()>)) {
        f(sf);
        visit_struct_field(sf, (e, v));
    }
    return mk_vt(@Visitor {
        visit_mod: |a,b,c,d|v_mod(v.visit_mod, a, b, c, d),
        visit_view_item: |a,b| v_view_item(v.visit_view_item, a, b),
        visit_foreign_item:
            |a,b|v_foreign_item(v.visit_foreign_item, a, b),
        visit_item: |a,b|v_item(v.visit_item, a, b),
        visit_local: |a,b|v_local(v.visit_local, a, b),
        visit_block: |a,b|v_block(v.visit_block, a, b),
        visit_stmt: |a,b|v_stmt(v.visit_stmt, a, b),
        visit_arm: |a,b|v_arm(v.visit_arm, a, b),
        visit_pat: |a,b|v_pat(v.visit_pat, a, b),
        visit_decl: |a,b|v_decl(v.visit_decl, a, b),
        visit_expr: |a,b|v_expr(v.visit_expr, a, b),
        visit_expr_post: |a,b| v_expr_post(v.visit_expr_post, a, b),
        visit_ty: visit_ty,
        visit_generics: |a,b|
            v_generics(v.visit_generics, a, b),
        visit_fn: |a,b,c,d,e,f|
            v_fn(v.visit_fn, a, b, c, d, e, f),
        visit_ty_method: |a,b|
            v_ty_method(v.visit_ty_method, a, b),
        visit_trait_method: |a,b|
            v_trait_method(v.visit_trait_method, a, b),
        visit_struct_def: |a,b,c,d,e|
            v_struct_def(v.visit_struct_def, a, b, c, d, e),
        visit_struct_field: |a,b|
            v_struct_field(v.visit_struct_field, a, b),
    });
}
