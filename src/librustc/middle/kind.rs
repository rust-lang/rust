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
use middle::ty::ParameterEnvironment;
use middle::ty;
use middle::ty_fold::TypeFoldable;
use middle::ty_fold;
use middle::typeck::check::vtable;
use middle::typeck::{MethodCall, NoAdjustment};
use middle::typeck;
use util::ppaux::{Repr, ty_to_string};
use util::ppaux::UserString;

use std::collections::HashSet;
use syntax::ast::*;
use syntax::ast_util;
use syntax::attr;
use syntax::codemap::Span;
use syntax::print::pprust::{expr_to_string, ident_to_string};
use syntax::visit::Visitor;
use syntax::visit;

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

pub struct Context<'a, 'tcx: 'a> {
    tcx: &'a ty::ctxt<'tcx>,
    struct_and_enum_bounds_checked: HashSet<ty::t>,
    parameter_environments: Vec<ParameterEnvironment>,
}

impl<'a, 'tcx, 'v> Visitor<'v> for Context<'a, 'tcx> {
    fn visit_expr(&mut self, ex: &Expr) {
        check_expr(self, ex);
    }

    fn visit_fn(&mut self, fk: visit::FnKind, fd: &'v FnDecl,
                b: &'v Block, s: Span, n: NodeId) {
        check_fn(self, fk, fd, b, s, n);
    }

    fn visit_ty(&mut self, t: &Ty) {
        check_ty(self, t);
    }

    fn visit_item(&mut self, i: &Item) {
        check_item(self, i);
    }

    fn visit_pat(&mut self, p: &Pat) {
        check_pat(self, p);
    }

    fn visit_local(&mut self, l: &Local) {
        check_local(self, l);
    }
}

pub fn check_crate(tcx: &ty::ctxt) {
    let mut ctx = Context {
        tcx: tcx,
        struct_and_enum_bounds_checked: HashSet::new(),
        parameter_environments: Vec::new(),
    };
    visit::walk_crate(&mut ctx, tcx.map.krate());
    tcx.sess.abort_if_errors();
}

struct EmptySubstsFolder<'a, 'tcx: 'a> {
    tcx: &'a ty::ctxt<'tcx>
}
impl<'a, 'tcx> ty_fold::TypeFolder<'tcx> for EmptySubstsFolder<'a, 'tcx> {
    fn tcx<'a>(&'a self) -> &'a ty::ctxt<'tcx> {
        self.tcx
    }
    fn fold_substs(&mut self, _: &subst::Substs) -> subst::Substs {
        subst::Substs::empty()
    }
}

fn check_struct_safe_for_destructor(cx: &mut Context,
                                    span: Span,
                                    struct_did: DefId) {
    let struct_tpt = ty::lookup_item_type(cx.tcx, struct_did);
    if !struct_tpt.generics.has_type_params(subst::TypeSpace)
      && !struct_tpt.generics.has_region_params(subst::TypeSpace) {
        let mut folder = EmptySubstsFolder { tcx: cx.tcx };
        if !ty::type_is_sendable(cx.tcx, struct_tpt.ty.fold_with(&mut folder)) {
            span_err!(cx.tcx.sess, span, E0125,
                      "cannot implement a destructor on a \
                       structure or enumeration that does not satisfy Send");
            span_note!(cx.tcx.sess, span,
                       "use \"#[unsafe_destructor]\" on the implementation \
                        to force the compiler to allow this");
        }
    } else {
        span_err!(cx.tcx.sess, span, E0141,
                  "cannot implement a destructor on a structure \
                   with type parameters");
        span_note!(cx.tcx.sess, span,
                   "use \"#[unsafe_destructor]\" on the implementation \
                    to force the compiler to allow this");
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
    debug!("checking impl with self type {}", ty::get(self_ty).sty);
    check_builtin_bounds(
        cx, self_ty, trait_def.bounds.builtin_bounds,
        |missing| {
            span_err!(cx.tcx.sess, self_type.span, E0142,
                      "the type `{}', which does not fulfill `{}`, \
                       cannot implement this trait",
                      ty_to_string(cx.tcx, self_ty), missing.user_string(cx.tcx));
            span_note!(cx.tcx.sess, self_type.span,
                       "types implementing this trait must fulfill `{}`",
                       trait_def.bounds.user_string(cx.tcx));
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
            ItemImpl(_, ref trait_ref, ref self_type, _) => {
                let parameter_environment =
                    ParameterEnvironment::for_item(cx.tcx, item.id);
                cx.parameter_environments.push(parameter_environment);

                // Check bounds on the `self` type.
                check_bounds_on_structs_or_enums_in_type_if_possible(
                    cx,
                    item.span,
                    ty::node_id_to_type(cx.tcx, item.id));

                match trait_ref {
                    &Some(ref trait_ref) => {
                        check_impl_of_trait(cx, item, trait_ref, &**self_type);

                        // Check bounds on the trait ref.
                        match ty::impl_trait_ref(cx.tcx,
                                                 ast_util::local_def(item.id)) {
                            None => {}
                            Some(trait_ref) => {
                                check_bounds_on_structs_or_enums_in_trait_ref(
                                    cx,
                                    item.span,
                                    &*trait_ref);

                                let trait_def = ty::lookup_trait_def(cx.tcx, trait_ref.def_id);
                                for (ty, type_param_def) in trait_ref.substs.types
                                                                  .iter()
                                                                  .zip(trait_def.generics
                                                                                .types
                                                                                .iter()) {
                                    check_typaram_bounds(cx, item.span, *ty, type_param_def);
                                }
                            }
                        }
                    }
                    &None => {}
                }

                drop(cx.parameter_environments.pop());
            }
            ItemEnum(..) => {
                let parameter_environment =
                    ParameterEnvironment::for_item(cx.tcx, item.id);
                cx.parameter_environments.push(parameter_environment);

                let def_id = ast_util::local_def(item.id);
                for variant in ty::enum_variants(cx.tcx, def_id).iter() {
                    for arg in variant.args.iter() {
                        check_bounds_on_structs_or_enums_in_type_if_possible(
                            cx,
                            item.span,
                            *arg)
                    }
                }

                drop(cx.parameter_environments.pop());
            }
            ItemStruct(..) => {
                let parameter_environment =
                    ParameterEnvironment::for_item(cx.tcx, item.id);
                cx.parameter_environments.push(parameter_environment);

                let def_id = ast_util::local_def(item.id);
                for field in ty::lookup_struct_fields(cx.tcx, def_id).iter() {
                    check_bounds_on_structs_or_enums_in_type_if_possible(
                        cx,
                        item.span,
                        ty::node_id_to_type(cx.tcx, field.id.node))
                }

                drop(cx.parameter_environments.pop());

            }
            ItemStatic(..) => {
                let parameter_environment =
                    ParameterEnvironment::for_item(cx.tcx, item.id);
                cx.parameter_environments.push(parameter_environment);

                check_bounds_on_structs_or_enums_in_type_if_possible(
                    cx,
                    item.span,
                    ty::node_id_to_type(cx.tcx, item.id));

                drop(cx.parameter_environments.pop());
            }
            _ => {}
        }
    }

    visit::walk_item(cx, item)
}

fn check_local(cx: &mut Context, local: &Local) {
    check_bounds_on_structs_or_enums_in_type_if_possible(
        cx,
        local.span,
        ty::node_id_to_type(cx.tcx, local.id));

    visit::walk_local(cx, local)
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
        span_err!(cx.tcx.sess, fv.span, E0143,
                  "can't capture dynamic environment in a fn item; \
                   use the || {} closure form instead", "{ ... }");
    } // same check is done in resolve.rs, but shouldn't be done

    let fty = ty::node_id_to_type(cx.tcx, id);
    match ty::get(fty).sty {
        ty::ty_closure(box ty::ClosureTy {
            store: ty::UniqTraitStore,
            bounds: bounds,
            ..
        }) => {
            b(|cx, fv| check_for_uniq(cx, fv, bounds.builtin_bounds))
        }

        ty::ty_closure(box ty::ClosureTy {
            store: ty::RegionTraitStore(region, _), bounds, ..
        }) => b(|cx, fv| check_for_block(cx, fv, bounds.builtin_bounds, region)),

        ty::ty_bare_fn(_) => {
            b(check_for_bare)
        }

        ty::ty_unboxed_closure(..) => {}

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
    fk: visit::FnKind,
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

    match fk {
        visit::FkFnBlock(..) => {
            let ty = ty::node_id_to_type(cx.tcx, fn_id);
            check_bounds_on_structs_or_enums_in_type_if_possible(cx, sp, ty);

            visit::walk_fn(cx, fk, decl, body, sp)
        }
        visit::FkItemFn(..) | visit::FkMethod(..) => {
            let parameter_environment = ParameterEnvironment::for_item(cx.tcx,
                                                                       fn_id);
            cx.parameter_environments.push(parameter_environment);

            let ty = ty::node_id_to_type(cx.tcx, fn_id);
            check_bounds_on_structs_or_enums_in_type_if_possible(cx, sp, ty);

            visit::walk_fn(cx, fk, decl, body, sp);
            drop(cx.parameter_environments.pop());
        }
    }
}

pub fn check_expr(cx: &mut Context, e: &Expr) {
    debug!("kind::check_expr({})", expr_to_string(e));

    // Handle any kind bounds on type parameters
    check_bounds_on_type_parameters(cx, e);

    // Check bounds on structures or enumerations in the type of the
    // expression.
    let expression_type = ty::expr_ty(cx.tcx, e);
    check_bounds_on_structs_or_enums_in_type_if_possible(cx,
                                                         e.span,
                                                         expression_type);

    match e.node {
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
        ExprAssign(ref lhs, _) |
        ExprAssignOp(_, ref lhs, _) => {
            let lhs_ty = ty::expr_ty(cx.tcx, &**lhs);
            if !ty::type_is_sized(cx.tcx, lhs_ty) {
                cx.tcx.sess.span_err(lhs.span, "dynamically sized type on lhs of assignment");
            }
        }
        ExprStruct(..) => {
            let e_ty = ty::expr_ty(cx.tcx, e);
            if !ty::type_is_sized(cx.tcx, e_ty) {
                cx.tcx.sess.span_err(e.span, "trying to initialise a dynamically sized struct");
            }
        }
        _ => {}
    }

    // Search for auto-adjustments to find trait coercions.
    match cx.tcx.adjustments.borrow().find(&e.id) {
        Some(adjustment) => {
            match adjustment {
                adj if ty::adjust_is_object(adj) => {
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
                _ => {}
            }
        }
        None => {}
    }

    visit::walk_expr(cx, e);
}

fn check_bounds_on_type_parameters(cx: &mut Context, e: &Expr) {
    let method_map = cx.tcx.method_map.borrow();
    let method_call = typeck::MethodCall::expr(e.id);
    let method = method_map.find(&method_call);

    // Find the values that were provided (if any)
    let item_substs = cx.tcx.item_substs.borrow();
    let (types, is_object_call) = match method {
        Some(method) => {
            let is_object_call = match method.origin {
                typeck::MethodObject(..) => true,
                typeck::MethodStatic(..) |
                typeck::MethodStaticUnboxedClosure(..) |
                typeck::MethodParam(..) => false
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

    // Check the vtable.
    let vtable_map = cx.tcx.vtable_map.borrow();
    let vtable_res = match vtable_map.find(&method_call) {
        None => return,
        Some(vtable_res) => vtable_res,
    };
    check_type_parameter_bounds_in_vtable_result(cx, e.span, vtable_res);
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
                    check_trait_cast_bounds(cx, span, source_ty,
                                            bounds.builtin_bounds);
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
                None => {}
                Some(ref item_substs) => {
                    let def_map = cx.tcx.def_map.borrow();
                    let did = def_map.get_copy(&id).def_id();
                    let generics = ty::lookup_item_type(cx.tcx, did).generics;
                    for def in generics.types.iter() {
                        let ty = *item_substs.substs.types.get(def.space,
                                                               def.index);
                        check_typaram_bounds(cx, aty.span, ty, def);
                    }
                }
            }
        }
        _ => {}
    }

    visit::walk_ty(cx, aty);
}

// Calls "any_missing" if any bounds were missing.
pub fn check_builtin_bounds(cx: &Context,
                            ty: ty::t,
                            bounds: ty::BuiltinBounds,
                            any_missing: |ty::BuiltinBounds|) {
    let kind = ty::type_contents(cx.tcx, ty);
    let mut missing = ty::empty_builtin_bounds();
    for bound in bounds.iter() {
        if !kind.meets_builtin_bound(cx.tcx, bound) {
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
        span_err!(cx.tcx.sess, sp, E0144,
                  "instantiating a type parameter with an incompatible type \
                   `{}`, which does not fulfill `{}`",
                   ty_to_string(cx.tcx, ty),
                   missing.user_string(cx.tcx));
    });
}

fn check_bounds_on_structs_or_enums_in_type_if_possible(cx: &mut Context,
                                                        span: Span,
                                                        ty: ty::t) {
    // If we aren't in a function, structure, or enumeration context, we don't
    // have enough information to ensure that bounds on structures or
    // enumerations are satisfied. So we don't perform the check.
    if cx.parameter_environments.len() == 0 {
        return
    }

    // If we've already checked for this type, don't do it again. This
    // massively speeds up kind checking.
    if cx.struct_and_enum_bounds_checked.contains(&ty) {
        return
    }
    cx.struct_and_enum_bounds_checked.insert(ty);

    ty::walk_ty(ty, |ty| {
        match ty::get(ty).sty {
            ty::ty_struct(type_id, ref substs) |
            ty::ty_enum(type_id, ref substs) => {
                let polytype = ty::lookup_item_type(cx.tcx, type_id);

                // Check builtin bounds.
                for (ty, type_param_def) in substs.types
                                                  .iter()
                                                  .zip(polytype.generics
                                                               .types
                                                               .iter()) {
                    check_typaram_bounds(cx, span, *ty, type_param_def);
                }

                // Check trait bounds.
                let parameter_environment =
                    cx.parameter_environments.get(cx.parameter_environments
                                                    .len() - 1);
                debug!(
                    "check_bounds_on_structs_or_enums_in_type_if_possible(): \
                     checking {}",
                    ty.repr(cx.tcx));
                vtable::check_param_bounds(cx.tcx,
                                           span,
                                           parameter_environment,
                                           &polytype.generics.types,
                                           substs,
                                           |missing| {
                    cx.tcx
                      .sess
                      .span_err(span,
                                format!("instantiating a type parameter with \
                                         an incompatible type `{}`, which \
                                         does not fulfill `{}`",
                                        ty_to_string(cx.tcx, ty),
                                        missing.user_string(
                                            cx.tcx)).as_slice());
                })
            }
            _ => {}
        }
    });
}

fn check_bounds_on_structs_or_enums_in_trait_ref(cx: &mut Context,
                                                 span: Span,
                                                 trait_ref: &ty::TraitRef) {
    for ty in trait_ref.substs.types.iter() {
        check_bounds_on_structs_or_enums_in_type_if_possible(cx, span, *ty)
    }
}

pub fn check_freevar_bounds(cx: &Context, sp: Span, ty: ty::t,
                            bounds: ty::BuiltinBounds, referenced_ty: Option<ty::t>)
{
    check_builtin_bounds(cx, ty, bounds, |missing| {
        // Will be Some if the freevar is implicitly borrowed (stack closure).
        // Emit a less mysterious error message in this case.
        match referenced_ty {
            Some(rty) => {
                span_err!(cx.tcx.sess, sp, E0145,
                    "cannot implicitly borrow variable of type `{}` in a \
                     bounded stack closure (implicit reference does not fulfill `{}`)",
                    ty_to_string(cx.tcx, rty), missing.user_string(cx.tcx));
            }
            None => {
                span_err!(cx.tcx.sess, sp, E0146,
                    "cannot capture variable of type `{}`, which does \
                     not fulfill `{}`, in a bounded closure",
                    ty_to_string(cx.tcx, ty), missing.user_string(cx.tcx));
            }
        }
        span_note!(cx.tcx.sess, sp,
            "this closure's environment must satisfy `{}`",
            bounds.user_string(cx.tcx));
    });
}

pub fn check_trait_cast_bounds(cx: &Context, sp: Span, ty: ty::t,
                               bounds: ty::BuiltinBounds) {
    check_builtin_bounds(cx, ty, bounds, |missing| {
        span_err!(cx.tcx.sess, sp, E0147,
            "cannot pack type `{}`, which does not fulfill `{}`, as a trait bounded by {}",
            ty_to_string(cx.tcx, ty),
            missing.user_string(cx.tcx),
            bounds.user_string(cx.tcx));
    });
}

fn check_copy(cx: &Context, ty: ty::t, sp: Span, reason: &str) {
    debug!("type_contents({})={}",
           ty_to_string(cx.tcx, ty),
           ty::type_contents(cx.tcx, ty).to_string());
    if ty::type_moves_by_default(cx.tcx, ty) {
        span_err!(cx.tcx.sess, sp, E0148,
            "copying a value of non-copyable type `{}`",
            ty_to_string(cx.tcx, ty));
        span_note!(cx.tcx.sess, sp, "{}", reason.as_slice());
    }
}

// Ensure that `ty` has a statically known size (i.e., it has the `Sized` bound).
fn check_sized(tcx: &ty::ctxt, ty: ty::t, name: String, sp: Span) {
    if !ty::type_is_sized(tcx, ty) {
        span_err!(tcx.sess, sp, E0151,
            "variable `{}` has dynamically sized type `{}`",
            name, ty_to_string(tcx, ty));
    }
}

// Check that any variables in a pattern have types with statically known size.
fn check_pat(cx: &mut Context, pat: &Pat) {
    let var_name = match pat.node {
        PatWild(PatWildSingle) => Some("_".to_string()),
        PatIdent(_, ref path1, _) => Some(ident_to_string(&path1.node).to_string()),
        _ => None
    };

    match var_name {
        Some(name) => {
            let types = cx.tcx.node_types.borrow();
            let ty = types.find(&(pat.id as uint));
            match ty {
                Some(ty) => {
                    debug!("kind: checking sized-ness of variable {}: {}",
                           name, ty_to_string(cx.tcx, *ty));
                    check_sized(cx.tcx, *ty, name, pat.span);
                }
                None => {} // extern fn args
            }
        }
        None => {}
    }

    visit::walk_pat(cx, pat);
}
