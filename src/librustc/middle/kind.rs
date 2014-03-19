// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


use middle::freevars::freevar_entry;
use middle::freevars;
use middle::ty;
use middle::typeck;
use util::ppaux::{Repr, ty_to_str};
use util::ppaux::UserString;

use std::vec_ng::Vec;
use syntax::ast::*;
use syntax::attr;
use syntax::codemap::Span;
use syntax::opt_vec;
use syntax::print::pprust::expr_to_str;
use syntax::{visit,ast_util};
use syntax::visit::Visitor;

// Kind analysis pass.
//
// There are several kinds defined by various operations. The most restrictive
// kind is noncopyable. The noncopyable kind can be extended with any number
// of the following attributes.
//
//  send: Things that can be sent on channels or included in spawned closures.
//  freeze: Things thare are deeply immutable. They are guaranteed never to
//    change, and can be safely shared without copying between tasks.
//  'static: Things that do not contain references.
//
// Send includes scalar types as well as classes and unique types containing
// only sendable types.
//
// Freeze include scalar types, things without non-const fields, and pointers
// to freezable things.
//
// This pass ensures that type parameters are only instantiated with types
// whose kinds are equal or less general than the way the type parameter was
// annotated (with the `Send` or `Freeze` bound).
//
// It also verifies that noncopyable kinds are not copied. Sendability is not
// applied, since none of our language primitives send. Instead, the sending
// primitives in the stdlib are explicitly annotated to only take sendable
// types.

#[deriving(Clone)]
pub struct Context<'a> {
    tcx: &'a ty::ctxt,
    method_map: typeck::MethodMap,
}

impl<'a> Visitor<()> for Context<'a> {

    fn visit_expr(&mut self, ex: &Expr, _: ()) {
        check_expr(self, ex);
    }

    fn visit_fn(&mut self, fk: &visit::FnKind, fd: &FnDecl,
                b: &Block, s: Span, n: NodeId, _: ()) {
        check_fn(self, fk, fd, b, s, n);
    }

    fn visit_ty(&mut self, t: &Ty, _: ()) {
        check_ty(self, t);
    }
    fn visit_item(&mut self, i: &Item, _: ()) {
        check_item(self, i);
    }
}

pub fn check_crate(tcx: &ty::ctxt,
                   method_map: typeck::MethodMap,
                   krate: &Crate) {
    let mut ctx = Context {
        tcx: tcx,
        method_map: method_map,
    };
    visit::walk_crate(&mut ctx, krate, ());
    tcx.sess.abort_if_errors();
}

fn check_struct_safe_for_destructor(cx: &mut Context,
                                    span: Span,
                                    struct_did: DefId) {
    let struct_tpt = ty::lookup_item_type(cx.tcx, struct_did);
    if !struct_tpt.generics.has_type_params() {
        let struct_ty = ty::mk_struct(cx.tcx, struct_did, ty::substs {
            regions: ty::NonerasedRegions(opt_vec::Empty),
            self_ty: None,
            tps: Vec::new()
        });
        if !ty::type_is_sendable(cx.tcx, struct_ty) {
            cx.tcx.sess.span_err(span,
                                 "cannot implement a destructor on a \
                                  structure that does not satisfy Send");
            cx.tcx.sess.span_note(span,
                                  "use \"#[unsafe_destructor]\" on the \
                                   implementation to force the compiler to \
                                   allow this");
        }
    } else {
        cx.tcx.sess.span_err(span,
                             "cannot implement a destructor on a structure \
                              with type parameters");
        cx.tcx.sess.span_note(span,
                              "use \"#[unsafe_destructor]\" on the \
                               implementation to force the compiler to \
                               allow this");
    }
}

fn check_impl_of_trait(cx: &mut Context, it: &Item, trait_ref: &TraitRef, self_type: &Ty) {
    let def_map = cx.tcx.def_map.borrow();
    let ast_trait_def = def_map.get()
                               .find(&trait_ref.ref_id)
                               .expect("trait ref not in def map!");
    let trait_def_id = ast_util::def_id_of_def(*ast_trait_def);
    let trait_def;
    {
        let trait_defs = cx.tcx.trait_defs.borrow();
        trait_def = *trait_defs.get()
                               .find(&trait_def_id)
                               .expect("trait def not in trait-defs map!");
    }

    // If this trait has builtin-kind supertraits, meet them.
    let self_ty: ty::t = ty::node_id_to_type(cx.tcx, it.id);
    debug!("checking impl with self type {:?}", ty::get(self_ty).sty);
    check_builtin_bounds(cx, self_ty, trait_def.bounds, |missing| {
        cx.tcx.sess.span_err(self_type.span,
            format!("the type `{}', which does not fulfill `{}`, cannot implement this \
                  trait", ty_to_str(cx.tcx, self_ty), missing.user_string(cx.tcx)));
        cx.tcx.sess.span_note(self_type.span,
            format!("types implementing this trait must fulfill `{}`",
                 trait_def.bounds.user_string(cx.tcx)));
    });

    // If this is a destructor, check kinds.
    if cx.tcx.lang_items.drop_trait() == Some(trait_def_id) {
        match self_type.node {
            TyPath(_, ref bounds, path_node_id) => {
                assert!(bounds.is_none());
                let struct_def = def_map.get().get_copy(&path_node_id);
                let struct_did = ast_util::def_id_of_def(struct_def);
                check_struct_safe_for_destructor(cx, self_type.span, struct_did);
            }
            _ => {
                cx.tcx.sess.span_bug(self_type.span,
                    "the self type for the Drop trait impl is not a path");
            }
        }
    }
}

fn check_item(cx: &mut Context, item: &Item) {
    if !attr::contains_name(item.attrs.as_slice(), "unsafe_destructor") {
        match item.node {
            ItemImpl(_, Some(ref trait_ref), self_type, _) => {
                check_impl_of_trait(cx, item, trait_ref, self_type);
            }
            _ => {}
        }
    }

    visit::walk_item(cx, item, ());
}

// Yields the appropriate function to check the kind of closed over
// variables. `id` is the NodeId for some expression that creates the
// closure.
fn with_appropriate_checker(cx: &Context,
                            id: NodeId,
                            b: |checker: |&Context, @freevar_entry||) {
    fn check_for_uniq(cx: &Context, fv: &freevar_entry, bounds: ty::BuiltinBounds) {
        // all captured data must be owned, regardless of whether it is
        // moved in or copied in.
        let id = ast_util::def_id_of_def(fv.def).node;
        let var_t = ty::node_id_to_type(cx.tcx, id);

        check_freevar_bounds(cx, fv.span, var_t, bounds, None);
    }

    fn check_for_block(cx: &Context, fv: &freevar_entry,
                       bounds: ty::BuiltinBounds, region: ty::Region) {
        let id = ast_util::def_id_of_def(fv.def).node;
        let var_t = ty::node_id_to_type(cx.tcx, id);
        // FIXME(#3569): Figure out whether the implicit borrow is actually
        // mutable. Currently we assume all upvars are referenced mutably.
        let implicit_borrowed_type = ty::mk_mut_rptr(cx.tcx, region, var_t);
        check_freevar_bounds(cx, fv.span, implicit_borrowed_type,
                             bounds, Some(var_t));
    }

    fn check_for_bare(cx: &Context, fv: @freevar_entry) {
        cx.tcx.sess.span_err(
            fv.span,
            "can't capture dynamic environment in a fn item; \
            use the || { ... } closure form instead");
    } // same check is done in resolve.rs, but shouldn't be done

    let fty = ty::node_id_to_type(cx.tcx, id);
    match ty::get(fty).sty {
        ty::ty_closure(~ty::ClosureTy {
            sigil: OwnedSigil,
            bounds: bounds,
            ..
        }) => {
            b(|cx, fv| check_for_uniq(cx, fv, bounds))
        }
        ty::ty_closure(~ty::ClosureTy {
            sigil: ManagedSigil,
            ..
        }) => {
            // can't happen
            fail!("internal error: saw closure with managed sigil (@fn)");
        }
        ty::ty_closure(~ty::ClosureTy {
            sigil: BorrowedSigil,
            bounds: bounds,
            region: region,
            ..
        }) => {
            b(|cx, fv| check_for_block(cx, fv, bounds, region))
        }
        ty::ty_bare_fn(_) => {
            b(check_for_bare)
        }
        ref s => {
            cx.tcx.sess.bug(
                format!("expect fn type in kind checker, not {:?}", s));
        }
    }
}

// Check that the free variables used in a shared/sendable closure conform
// to the copy/move kind bounds. Then recursively check the function body.
fn check_fn(
    cx: &mut Context,
    fk: &visit::FnKind,
    decl: &FnDecl,
    body: &Block,
    sp: Span,
    fn_id: NodeId) {

    // Check kinds on free variables:
    with_appropriate_checker(cx, fn_id, |chk| {
        let r = freevars::get_freevars(cx.tcx, fn_id);
        for fv in r.iter() {
            chk(cx, *fv);
        }
    });

    visit::walk_fn(cx, fk, decl, body, sp, fn_id, ());
}

pub fn check_expr(cx: &mut Context, e: &Expr) {
    debug!("kind::check_expr({})", expr_to_str(e));

    // Handle any kind bounds on type parameters
    {
        let method_map = cx.method_map.borrow();
        let method = method_map.get().find(&typeck::MethodCall::expr(e.id));
        let node_type_substs = cx.tcx.node_type_substs.borrow();
        let r = match method {
            Some(method) => Some(&method.substs.tps),
            None => node_type_substs.get().find(&e.id)
        };
        for ts in r.iter() {
            let def_map = cx.tcx.def_map.borrow();
            let type_param_defs = match e.node {
              ExprPath(_) => {
                let did = ast_util::def_id_of_def(def_map.get()
                                                         .get_copy(&e.id));
                ty::lookup_item_type(cx.tcx, did).generics.type_param_defs.clone()
              }
              _ => {
                // Type substitutions should only occur on paths and
                // method calls, so this needs to be a method call.

                // Even though the callee_id may have been the id with
                // node_type_substs, e.id is correct here.
                match method {
                    Some(method) => {
                        ty::method_call_type_param_defs(cx.tcx, method.origin)
                    }
                    None => {
                        cx.tcx.sess.span_bug(e.span,
                            "non path/method call expr has type substs??");
                    }
                }
              }
            };
            let type_param_defs = type_param_defs.deref();
            if ts.len() != type_param_defs.len() {
                // Fail earlier to make debugging easier
                fail!("internal error: in kind::check_expr, length \
                      mismatch between actual and declared bounds: actual = \
                      {}, declared = {}",
                      ts.repr(cx.tcx),
                      type_param_defs.repr(cx.tcx));
            }
            for (&ty, type_param_def) in ts.iter().zip(type_param_defs.iter()) {
                check_typaram_bounds(cx, e.span, ty, type_param_def)
            }
        }
    }

    match e.node {
        ExprUnary(UnBox, interior) => {
            let interior_type = ty::expr_ty(cx.tcx, interior);
            let _ = check_static(cx.tcx, interior_type, interior.span);
        }
        ExprCast(source, _) => {
            let source_ty = ty::expr_ty(cx.tcx, source);
            let target_ty = ty::expr_ty(cx.tcx, e);
            check_trait_cast(cx, source_ty, target_ty, source.span);
        }
        ExprRepeat(element, count_expr, _) => {
            let count = ty::eval_repeat_count(cx.tcx, count_expr);
            if count > 1 {
                let element_ty = ty::expr_ty(cx.tcx, element);
                check_copy(cx, element_ty, element.span,
                           "repeated element will be copied");
            }
        }
        _ => {}
    }

    // Search for auto-adjustments to find trait coercions.
    let adjustments = cx.tcx.adjustments.borrow();
    match adjustments.get().find(&e.id) {
        Some(adjustment) => {
            match **adjustment {
                ty::AutoObject(..) => {
                    let source_ty = ty::expr_ty(cx.tcx, e);
                    let target_ty = ty::expr_ty_adjusted(cx.tcx, e,
                                                         cx.method_map.borrow().get());
                    check_trait_cast(cx, source_ty, target_ty, e.span);
                }
                ty::AutoAddEnv(..) |
                ty::AutoDerefRef(..) => {}
            }
        }
        None => {}
    }

    visit::walk_expr(cx, e, ());
}

fn check_trait_cast(cx: &mut Context, source_ty: ty::t, target_ty: ty::t, span: Span) {
    check_cast_for_escaping_regions(cx, source_ty, target_ty, span);
    match ty::get(target_ty).sty {
        ty::ty_trait(~ty::TyTrait { bounds, .. }) => {
            check_trait_cast_bounds(cx, span, source_ty, bounds);
        }
        _ => {}
    }
}

fn check_ty(cx: &mut Context, aty: &Ty) {
    match aty.node {
        TyPath(_, _, id) => {
            let node_type_substs = cx.tcx.node_type_substs.borrow();
            let r = node_type_substs.get().find(&id);
            for ts in r.iter() {
                let def_map = cx.tcx.def_map.borrow();
                let did = ast_util::def_id_of_def(def_map.get().get_copy(&id));
                let generics = ty::lookup_item_type(cx.tcx, did).generics;
                let type_param_defs = generics.type_param_defs();
                for (&ty, type_param_def) in ts.iter().zip(type_param_defs.iter()) {
                    check_typaram_bounds(cx, aty.span, ty, type_param_def)
                }
            }
        }
        _ => {}
    }
    visit::walk_ty(cx, aty, ());
}

// Calls "any_missing" if any bounds were missing.
pub fn check_builtin_bounds(cx: &Context,
                            ty: ty::t,
                            bounds: ty::BuiltinBounds,
                            any_missing: |ty::BuiltinBounds|) {
    let kind = ty::type_contents(cx.tcx, ty);
    let mut missing = ty::EmptyBuiltinBounds();
    for bound in bounds.iter() {
        if !kind.meets_bound(cx.tcx, bound) {
            missing.add(bound);
        }
    }
    if !missing.is_empty() {
        any_missing(missing);
    }
}

pub fn check_typaram_bounds(cx: &Context,
                            sp: Span,
                            ty: ty::t,
                            type_param_def: &ty::TypeParameterDef) {
    check_builtin_bounds(cx,
                         ty,
                         type_param_def.bounds.builtin_bounds,
                         |missing| {
        cx.tcx.sess.span_err(
            sp,
            format!("instantiating a type parameter with an incompatible type \
                  `{}`, which does not fulfill `{}`",
                 ty_to_str(cx.tcx, ty),
                 missing.user_string(cx.tcx)));
    });
}

pub fn check_freevar_bounds(cx: &Context, sp: Span, ty: ty::t,
                            bounds: ty::BuiltinBounds, referenced_ty: Option<ty::t>)
{
    check_builtin_bounds(cx, ty, bounds, |missing| {
        // Will be Some if the freevar is implicitly borrowed (stack closure).
        // Emit a less mysterious error message in this case.
        match referenced_ty {
            Some(rty) => cx.tcx.sess.span_err(sp,
                format!("cannot implicitly borrow variable of type `{}` in a bounded \
                      stack closure (implicit reference does not fulfill `{}`)",
                     ty_to_str(cx.tcx, rty), missing.user_string(cx.tcx))),
            None => cx.tcx.sess.span_err(sp,
                format!("cannot capture variable of type `{}`, which does \
                      not fulfill `{}`, in a bounded closure",
                     ty_to_str(cx.tcx, ty), missing.user_string(cx.tcx))),
        }
        cx.tcx.sess.span_note(
            sp,
            format!("this closure's environment must satisfy `{}`",
                 bounds.user_string(cx.tcx)));
    });
}

pub fn check_trait_cast_bounds(cx: &Context, sp: Span, ty: ty::t,
                               bounds: ty::BuiltinBounds) {
    check_builtin_bounds(cx, ty, bounds, |missing| {
        cx.tcx.sess.span_err(sp,
            format!("cannot pack type `{}`, which does not fulfill \
                  `{}`, as a trait bounded by {}",
                 ty_to_str(cx.tcx, ty), missing.user_string(cx.tcx),
                 bounds.user_string(cx.tcx)));
    });
}

fn check_copy(cx: &Context, ty: ty::t, sp: Span, reason: &str) {
    debug!("type_contents({})={}",
           ty_to_str(cx.tcx, ty),
           ty::type_contents(cx.tcx, ty).to_str());
    if ty::type_moves_by_default(cx.tcx, ty) {
        cx.tcx.sess.span_err(
            sp, format!("copying a value of non-copyable type `{}`",
                     ty_to_str(cx.tcx, ty)));
        cx.tcx.sess.span_note(sp, format!("{}", reason));
    }
}

pub fn check_send(cx: &Context, ty: ty::t, sp: Span) -> bool {
    if !ty::type_is_sendable(cx.tcx, ty) {
        cx.tcx.sess.span_err(
            sp, format!("value has non-sendable type `{}`",
                     ty_to_str(cx.tcx, ty)));
        false
    } else {
        true
    }
}

pub fn check_static(tcx: &ty::ctxt, ty: ty::t, sp: Span) -> bool {
    if !ty::type_is_static(tcx, ty) {
        match ty::get(ty).sty {
          ty::ty_param(..) => {
            tcx.sess.span_err(sp,
                format!("value may contain references; \
                         add `'static` bound to `{}`", ty_to_str(tcx, ty)));
          }
          _ => {
            tcx.sess.span_err(sp, "value may contain references");
          }
        }
        false
    } else {
        true
    }
}

/// This is rather subtle.  When we are casting a value to an instantiated
/// trait like `a as trait<'r>`, regionck already ensures that any references
/// that appear in the type of `a` are bounded by `'r` (ed.: rem
/// FIXME(#5723)).  However, it is possible that there are *type parameters*
/// in the type of `a`, and those *type parameters* may have references
/// within them.  We have to guarantee that the regions which appear in those
/// type parameters are not obscured.
///
/// Therefore, we ensure that one of three conditions holds:
///
/// (1) The trait instance cannot escape the current fn.  This is
/// guaranteed if the region bound `&r` is some scope within the fn
/// itself.  This case is safe because whatever references are
/// found within the type parameter, they must enclose the fn body
/// itself.
///
/// (2) The type parameter appears in the type of the trait.  For
/// example, if the type parameter is `T` and the trait type is
/// `deque<T>`, then whatever references may appear in `T` also
/// appear in `deque<T>`.
///
/// (3) The type parameter is sendable (and therefore does not contain
/// references).
///
/// FIXME(#5723)---This code should probably move into regionck.
pub fn check_cast_for_escaping_regions(
    cx: &Context,
    source_ty: ty::t,
    target_ty: ty::t,
    source_span: Span)
{
    // Determine what type we are casting to; if it is not a trait, then no
    // worries.
    match ty::get(target_ty).sty {
        ty::ty_trait(..) => {}
        _ => { return; }
    }

    // Collect up the regions that appear in the target type.  We want to
    // ensure that these lifetimes are shorter than all lifetimes that are in
    // the source type.  See test `src/test/compile-fail/regions-trait-2.rs`
    let mut target_regions = Vec::new();
    ty::walk_regions_and_ty(
        cx.tcx,
        target_ty,
        |r| {
            if !r.is_bound() {
                target_regions.push(r);
            }
        },
        |_| ());

    // Check, based on the region associated with the trait, whether it can
    // possibly escape the enclosing fn item (note that all type parameters
    // must have been declared on the enclosing fn item).
    if target_regions.iter().any(|r| is_ReScope(*r)) {
        return; /* case (1) */
    }

    // Assuming the trait instance can escape, then ensure that each parameter
    // either appears in the trait type or is sendable.
    let target_params = ty::param_tys_in_type(target_ty);
    ty::walk_regions_and_ty(
        cx.tcx,
        source_ty,

        |_r| {
            // FIXME(#5723) --- turn this check on once &Objects are usable
            //
            // if !target_regions.iter().any(|t_r| is_subregion_of(cx, *t_r, r)) {
            //     cx.tcx.sess.span_err(
            //         source_span,
            //         format!("source contains reference with lifetime \
            //               not found in the target type `{}`",
            //              ty_to_str(cx.tcx, target_ty)));
            //     note_and_explain_region(
            //         cx.tcx, "source data is only valid for ", r, "");
            // }
        },

        |ty| {
            match ty::get(ty).sty {
                ty::ty_param(source_param) => {
                    if target_params.iter().any(|x| x == &source_param) {
                        /* case (2) */
                    } else {
                        check_static(cx.tcx, ty, source_span); /* case (3) */
                    }
                }
                _ => {}
            }
        });

    fn is_ReScope(r: ty::Region) -> bool {
        match r {
            ty::ReScope(..) => true,
            _ => false
        }
    }
}
