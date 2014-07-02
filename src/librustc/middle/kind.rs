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
use middle::subst;
use middle::ty;
use middle::typeck::{MethodCall, NoAdjustment};
use middle::typeck;
use util::ppaux::{Repr, ty_to_str};
use util::ppaux::UserString;

use syntax::ast::*;
use syntax::attr;
use syntax::codemap::Span;
use syntax::print::pprust::{expr_to_str, ident_to_str};
use syntax::{visit};
use syntax::visit::Visitor;

// Kind analysis pass.
//
// There are several kinds defined by various operations. The most restrictive
// kind is noncopyable. The noncopyable kind can be extended with any number
// of the following attributes.
//
//  Send: Things that can be sent on channels or included in spawned closures. It
//  includes scalar types as well as classes and unique types containing only
//  sendable types.
//  'static: Things that do not contain references.
//
// This pass ensures that type parameters are only instantiated with types
// whose kinds are equal or less general than the way the type parameter was
// annotated (with the `Send` bound).
//
// It also verifies that noncopyable kinds are not copied. Sendability is not
// applied, since none of our language primitives send. Instead, the sending
// primitives in the stdlib are explicitly annotated to only take sendable
// types.

#[deriving(Clone)]
pub struct Context<'a> {
    tcx: &'a ty::ctxt,
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

    fn visit_pat(&mut self, p: &Pat, _: ()) {
        check_pat(self, p);
    }
}

pub fn check_crate(tcx: &ty::ctxt,
                   krate: &Crate) {
    let mut ctx = Context {
        tcx: tcx,
    };
    visit::walk_crate(&mut ctx, krate, ());
    tcx.sess.abort_if_errors();
}

fn check_struct_safe_for_destructor(cx: &mut Context,
                                    span: Span,
                                    struct_did: DefId) {
    let struct_tpt = ty::lookup_item_type(cx.tcx, struct_did);
    if !struct_tpt.generics.has_type_params(subst::TypeSpace) {
        let struct_ty = ty::mk_struct(cx.tcx, struct_did,
                                      subst::Substs::empty());
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
    let ast_trait_def = *cx.tcx.def_map.borrow()
                              .find(&trait_ref.ref_id)
                              .expect("trait ref not in def map!");
    let trait_def_id = ast_trait_def.def_id();
    let trait_def = cx.tcx.trait_defs.borrow()
                          .find_copy(&trait_def_id)
                          .expect("trait def not in trait-defs map!");

    // If this trait has builtin-kind supertraits, meet them.
    let self_ty: ty::t = ty::node_id_to_type(cx.tcx, it.id);
    debug!("checking impl with self type {:?}", ty::get(self_ty).sty);
    check_builtin_bounds(cx, self_ty, trait_def.bounds, |missing| {
        cx.tcx.sess.span_err(self_type.span,
            format!("the type `{}', which does not fulfill `{}`, cannot implement this \
                    trait",
                    ty_to_str(cx.tcx, self_ty),
                    missing.user_string(cx.tcx)).as_slice());
        cx.tcx.sess.span_note(self_type.span,
            format!("types implementing this trait must fulfill `{}`",
                    trait_def.bounds.user_string(cx.tcx)).as_slice());
    });

    // If this is a destructor, check kinds.
    if cx.tcx.lang_items.drop_trait() == Some(trait_def_id) {
        match self_type.node {
            TyPath(_, ref bounds, path_node_id) => {
                assert!(bounds.is_none());
                let struct_def = cx.tcx.def_map.borrow().get_copy(&path_node_id);
                let struct_did = struct_def.def_id();
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
            ItemImpl(_, Some(ref trait_ref), ref self_type, _) => {
                check_impl_of_trait(cx, item, trait_ref, &**self_type);
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
                            b: |checker: |&Context, &freevar_entry||) {
    fn check_for_uniq(cx: &Context, fv: &freevar_entry, bounds: ty::BuiltinBounds) {
        // all captured data must be owned, regardless of whether it is
        // moved in or copied in.
        let id = fv.def.def_id().node;
        let var_t = ty::node_id_to_type(cx.tcx, id);

        check_freevar_bounds(cx, fv.span, var_t, bounds, None);
    }

    fn check_for_block(cx: &Context, fv: &freevar_entry,
                       bounds: ty::BuiltinBounds, region: ty::Region) {
        let id = fv.def.def_id().node;
        let var_t = ty::node_id_to_type(cx.tcx, id);
        // FIXME(#3569): Figure out whether the implicit borrow is actually
        // mutable. Currently we assume all upvars are referenced mutably.
        let implicit_borrowed_type = ty::mk_mut_rptr(cx.tcx, region, var_t);
        check_freevar_bounds(cx, fv.span, implicit_borrowed_type,
                             bounds, Some(var_t));
    }

    fn check_for_bare(cx: &Context, fv: &freevar_entry) {
        cx.tcx.sess.span_err(
            fv.span,
            "can't capture dynamic environment in a fn item; \
            use the || { ... } closure form instead");
    } // same check is done in resolve.rs, but shouldn't be done

    let fty = ty::node_id_to_type(cx.tcx, id);
    match ty::get(fty).sty {
        ty::ty_closure(box ty::ClosureTy {
            store: ty::UniqTraitStore,
            bounds: mut bounds, ..
        }) => {
            // Procs can't close over non-static references!
            bounds.add(ty::BoundStatic);

            b(|cx, fv| check_for_uniq(cx, fv, bounds))
        }

        ty::ty_closure(box ty::ClosureTy {
            store: ty::RegionTraitStore(region, _), bounds, ..
        }) => b(|cx, fv| check_for_block(cx, fv, bounds, region)),

        ty::ty_bare_fn(_) => {
            b(check_for_bare)
        }
        ref s => {
            cx.tcx.sess.bug(format!("expect fn type in kind checker, not \
                                     {:?}",
                                    s).as_slice());
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
        freevars::with_freevars(cx.tcx, fn_id, |freevars| {
            for fv in freevars.iter() {
                chk(cx, fv);
            }
        });
    });

    visit::walk_fn(cx, fk, decl, body, sp, ());
}

pub fn check_expr(cx: &mut Context, e: &Expr) {
    debug!("kind::check_expr({})", expr_to_str(e));

    // Handle any kind bounds on type parameters
    check_bounds_on_type_parameters(cx, e);

    match e.node {
        ExprBox(ref loc, ref interior) => {
            let def = ty::resolve_expr(cx.tcx, &**loc);
            if Some(def.def_id()) == cx.tcx.lang_items.managed_heap() {
                let interior_type = ty::expr_ty(cx.tcx, &**interior);
                let _ = check_static(cx.tcx, interior_type, interior.span);
            }
        }
        ExprCast(ref source, _) => {
            let source_ty = ty::expr_ty(cx.tcx, &**source);
            let target_ty = ty::expr_ty(cx.tcx, e);
            let method_call = MethodCall {
                expr_id: e.id,
                adjustment: NoAdjustment,
            };
            check_trait_cast(cx,
                             source_ty,
                             target_ty,
                             source.span,
                             method_call);
        }
        ExprRepeat(ref element, ref count_expr) => {
            let count = ty::eval_repeat_count(cx.tcx, &**count_expr);
            if count > 1 {
                let element_ty = ty::expr_ty(cx.tcx, &**element);
                check_copy(cx, element_ty, element.span,
                           "repeated element will be copied");
            }
        }
        _ => {}
    }

    // Search for auto-adjustments to find trait coercions.
    match cx.tcx.adjustments.borrow().find(&e.id) {
        Some(adjustment) => {
            match *adjustment {
                ty::AutoObject(..) => {
                    let source_ty = ty::expr_ty(cx.tcx, e);
                    let target_ty = ty::expr_ty_adjusted(cx.tcx, e);
                    let method_call = MethodCall {
                        expr_id: e.id,
                        adjustment: typeck::AutoObject,
                    };
                    check_trait_cast(cx,
                                     source_ty,
                                     target_ty,
                                     e.span,
                                     method_call);
                }
                ty::AutoAddEnv(..) |
                ty::AutoDerefRef(..) => {}
            }
        }
        None => {}
    }

    visit::walk_expr(cx, e, ());
}

fn check_bounds_on_type_parameters(cx: &mut Context, e: &Expr) {
    let method_map = cx.tcx.method_map.borrow();
    let method = method_map.find(&typeck::MethodCall::expr(e.id));

    // Find the values that were provided (if any)
    let item_substs = cx.tcx.item_substs.borrow();
    let (types, is_object_call) = match method {
        Some(method) => {
            let is_object_call = match method.origin {
                typeck::MethodObject(..) => true,
                typeck::MethodStatic(..) | typeck::MethodParam(..) => false
            };
            (&method.substs.types, is_object_call)
        }
        None => {
            match item_substs.find(&e.id) {
                None => { return; }
                Some(s) => { (&s.substs.types, false) }
            }
        }
    };

    // Find the relevant type parameter definitions
    let def_map = cx.tcx.def_map.borrow();
    let type_param_defs = match e.node {
        ExprPath(_) => {
            let did = def_map.get_copy(&e.id).def_id();
            ty::lookup_item_type(cx.tcx, did).generics.types.clone()
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

    // Check that the value provided for each definition meets the
    // kind requirements
    for type_param_def in type_param_defs.iter() {
        let ty = *types.get(type_param_def.space, type_param_def.index);

        // If this is a call to an object method (`foo.bar()` where
        // `foo` has a type like `Trait`), then the self type is
        // unknown (after all, this is a virtual call). In that case,
        // we will have put a ty_err in the substitutions, and we can
        // just skip over validating the bounds (because the bounds
        // would have been enforced when the object instance was
        // created).
        if is_object_call && type_param_def.space == subst::SelfSpace {
            assert_eq!(type_param_def.index, 0);
            assert!(ty::type_is_error(ty));
            continue;
        }

        debug!("type_param_def space={} index={} ty={}",
               type_param_def.space, type_param_def.index, ty.repr(cx.tcx));
        check_typaram_bounds(cx, e.span, ty, type_param_def)
    }
}

fn check_type_parameter_bounds_in_vtable_result(
        cx: &mut Context,
        span: Span,
        vtable_res: &typeck::vtable_res) {
    for origins in vtable_res.iter() {
        for origin in origins.iter() {
            let (type_param_defs, substs) = match *origin {
                typeck::vtable_static(def_id, ref tys, _) => {
                    let type_param_defs =
                        ty::lookup_item_type(cx.tcx, def_id).generics
                                                            .types
                                                            .clone();
                    (type_param_defs, (*tys).clone())
                }
                _ => {
                    // Nothing to do here.
                    continue
                }
            };
            for type_param_def in type_param_defs.iter() {
                let typ = substs.types.get(type_param_def.space,
                                           type_param_def.index);
                check_typaram_bounds(cx, span, *typ, type_param_def)
            }
        }
    }
}

fn check_trait_cast(cx: &mut Context,
                    source_ty: ty::t,
                    target_ty: ty::t,
                    span: Span,
                    method_call: MethodCall) {
    check_cast_for_escaping_regions(cx, source_ty, target_ty, span);
    match ty::get(target_ty).sty {
        ty::ty_uniq(ty) | ty::ty_rptr(_, ty::mt{ ty, .. }) => {
            match ty::get(ty).sty {
                ty::ty_trait(box ty::TyTrait { bounds, .. }) => {
                     match cx.tcx.vtable_map.borrow().find(&method_call) {
                        None => {
                            cx.tcx.sess.span_bug(span,
                                                 "trait cast not in vtable \
                                                  map?!")
                        }
                        Some(vtable_res) => {
                            check_type_parameter_bounds_in_vtable_result(
                                cx,
                                span,
                                vtable_res)
                        }
                    };
                    check_trait_cast_bounds(cx, span, source_ty, bounds);
                }
                _ => {}
            }
        }
        _ => {}
    }
}

fn check_ty(cx: &mut Context, aty: &Ty) {
    match aty.node {
        TyPath(_, _, id) => {
            match cx.tcx.item_substs.borrow().find(&id) {
                None => { }
                Some(ref item_substs) => {
                    let def_map = cx.tcx.def_map.borrow();
                    let did = def_map.get_copy(&id).def_id();
                    let generics = ty::lookup_item_type(cx.tcx, did).generics;
                    for def in generics.types.iter() {
                        let ty = *item_substs.substs.types.get(def.space,
                                                               def.index);
                        check_typaram_bounds(cx, aty.span, ty, def)
                    }
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
    let mut missing = ty::empty_builtin_bounds();
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
                    missing.user_string(cx.tcx)).as_slice());
    });
}

pub fn check_freevar_bounds(cx: &Context, sp: Span, ty: ty::t,
                            bounds: ty::BuiltinBounds, referenced_ty: Option<ty::t>)
{
    check_builtin_bounds(cx, ty, bounds, |missing| {
        // Will be Some if the freevar is implicitly borrowed (stack closure).
        // Emit a less mysterious error message in this case.
        match referenced_ty {
            Some(rty) => {
                cx.tcx.sess.span_err(sp,
                format!("cannot implicitly borrow variable of type `{}` in a \
                         bounded stack closure (implicit reference does not \
                         fulfill `{}`)",
                        ty_to_str(cx.tcx, rty),
                        missing.user_string(cx.tcx)).as_slice())
            }
            None => {
                cx.tcx.sess.span_err(sp,
                format!("cannot capture variable of type `{}`, which does \
                         not fulfill `{}`, in a bounded closure",
                        ty_to_str(cx.tcx, ty),
                        missing.user_string(cx.tcx)).as_slice())
            }
        }
        cx.tcx.sess.span_note(
            sp,
            format!("this closure's environment must satisfy `{}`",
                    bounds.user_string(cx.tcx)).as_slice());
    });
}

pub fn check_trait_cast_bounds(cx: &Context, sp: Span, ty: ty::t,
                               bounds: ty::BuiltinBounds) {
    check_builtin_bounds(cx, ty, bounds, |missing| {
        cx.tcx.sess.span_err(sp,
            format!("cannot pack type `{}`, which does not fulfill \
                     `{}`, as a trait bounded by {}",
                    ty_to_str(cx.tcx, ty), missing.user_string(cx.tcx),
                    bounds.user_string(cx.tcx)).as_slice());
    });
}

fn check_copy(cx: &Context, ty: ty::t, sp: Span, reason: &str) {
    debug!("type_contents({})={}",
           ty_to_str(cx.tcx, ty),
           ty::type_contents(cx.tcx, ty).to_str());
    if ty::type_moves_by_default(cx.tcx, ty) {
        cx.tcx.sess.span_err(
            sp,
            format!("copying a value of non-copyable type `{}`",
                    ty_to_str(cx.tcx, ty)).as_slice());
        cx.tcx.sess.span_note(sp, format!("{}", reason).as_slice());
    }
}

pub fn check_static(tcx: &ty::ctxt, ty: ty::t, sp: Span) -> bool {
    if !ty::type_is_static(tcx, ty) {
        match ty::get(ty).sty {
          ty::ty_param(..) => {
            tcx.sess.span_err(sp,
                format!("value may contain references; \
                         add `'static` bound to `{}`",
                        ty_to_str(tcx, ty)).as_slice());
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
    if !ty::type_is_trait(target_ty) {
        return;
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
                    if source_param.space == subst::SelfSpace {
                        // FIXME (#5723) -- there is no reason that
                        // Self should be exempt from this check,
                        // except for historical accident. Bottom
                        // line, we need proper region bounding.
                    } else if target_params.iter().any(|x| x == &source_param) {
                        /* case (2) */
                    } else {
                        check_static(cx.tcx, ty, source_span); /* case (3) */
                    }
                }
                _ => {}
            }
        });

    #[allow(non_snake_case_functions)]
    fn is_ReScope(r: ty::Region) -> bool {
        match r {
            ty::ReScope(..) => true,
            _ => false
        }
    }
}

// Ensure that `ty` has a statically known size (i.e., it has the `Sized` bound).
fn check_sized(tcx: &ty::ctxt, ty: ty::t, name: String, sp: Span) {
    if !ty::type_is_sized(tcx, ty) {
        tcx.sess.span_err(sp,
                          format!("variable `{}` has dynamically sized type \
                                   `{}`",
                                  name,
                                  ty_to_str(tcx, ty)).as_slice());
    }
}

// Check that any variables in a pattern have types with statically known size.
fn check_pat(cx: &mut Context, pat: &Pat) {
    let var_name = match pat.node {
        PatWild => Some("_".to_string()),
        PatIdent(_, ref path1, _) => Some(ident_to_str(&path1.node).to_string()),
        _ => None
    };

    match var_name {
        Some(name) => {
            let types = cx.tcx.node_types.borrow();
            let ty = types.find(&(pat.id as uint));
            match ty {
                Some(ty) => {
                    debug!("kind: checking sized-ness of variable {}: {}",
                           name, ty_to_str(cx.tcx, *ty));
                    check_sized(cx.tcx, *ty, name, pat.span);
                }
                None => {} // extern fn args
            }
        }
        None => {}
    }

    visit::walk_pat(cx, pat, ());
}
