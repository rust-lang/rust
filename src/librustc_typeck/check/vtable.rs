// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use check::{FnCtxt, structurally_resolved_type};
use middle::subst::{FnSpace, SelfSpace};
use middle::traits;
use middle::traits::{Obligation, ObligationCause};
use middle::traits::report_fulfillment_errors;
use middle::ty::{mod, Ty, AsPredicate};
use middle::infer;
use syntax::ast;
use syntax::codemap::Span;
use util::nodemap::FnvHashSet;
use util::ppaux::{Repr, UserString};

pub fn check_object_cast<'a, 'tcx>(fcx: &FnCtxt<'a, 'tcx>,
                                   cast_expr: &ast::Expr,
                                   source_expr: &ast::Expr,
                                   target_object_ty: Ty<'tcx>)
{
    debug!("check_object_cast(cast_expr={}, target_object_ty={})",
           cast_expr.repr(fcx.tcx()),
           target_object_ty.repr(fcx.tcx()));

    // Look up vtables for the type we're casting to,
    // passing in the source and target type.  The source
    // must be a pointer type suitable to the object sigil,
    // e.g.: `&x as &Trait` or `box x as Box<Trait>`
    let source_ty = fcx.expr_ty(source_expr);
    let source_ty = structurally_resolved_type(fcx, source_expr.span, source_ty);
    debug!("source_ty={}", source_ty.repr(fcx.tcx()));
    match (&source_ty.sty, &target_object_ty.sty) {
        (&ty::ty_uniq(referent_ty), &ty::ty_uniq(object_trait_ty)) => {
            let object_trait = object_trait(&object_trait_ty);

            // Ensure that if ~T is cast to ~Trait, then T : Trait
            push_cast_obligation(fcx, cast_expr, object_trait, referent_ty);
            check_object_safety(fcx.tcx(), object_trait, source_expr.span);
        }

        (&ty::ty_rptr(referent_region, ty::mt { ty: referent_ty,
                                                mutbl: referent_mutbl }),
         &ty::ty_rptr(target_region, ty::mt { ty: object_trait_ty,
                                              mutbl: target_mutbl })) =>
        {
            let object_trait = object_trait(&object_trait_ty);
            if !mutability_allowed(referent_mutbl, target_mutbl) {
                fcx.tcx().sess.span_err(source_expr.span,
                                        "types differ in mutability");
            } else {
                // Ensure that if &'a T is cast to &'b Trait, then T : Trait
                push_cast_obligation(fcx, cast_expr,
                                     object_trait,
                                     referent_ty);

                // Ensure that if &'a T is cast to &'b Trait, then 'b <= 'a
                infer::mk_subr(fcx.infcx(),
                               infer::RelateObjectBound(source_expr.span),
                               *target_region,
                               *referent_region);

                check_object_safety(fcx.tcx(), object_trait, source_expr.span);
            }
        }

        (_, &ty::ty_uniq(..)) => {
            fcx.ccx.tcx.sess.span_err(
                source_expr.span,
                format!("can only cast an boxed pointer \
                         to a boxed object, not a {}",
                        ty::ty_sort_string(fcx.tcx(), source_ty))[]);
        }

        (_, &ty::ty_rptr(..)) => {
            fcx.ccx.tcx.sess.span_err(
                source_expr.span,
                format!("can only cast a &-pointer \
                         to an &-object, not a {}",
                        ty::ty_sort_string(fcx.tcx(), source_ty))[]);
        }

        _ => {
            fcx.tcx().sess.span_bug(
                source_expr.span,
                "expected object type");
        }
    }

    fn object_trait<'a, 'tcx>(t: &'a Ty<'tcx>) -> &'a ty::TyTrait<'tcx> {
        match t.sty {
            ty::ty_trait(ref ty_trait) => &**ty_trait,
            _ => panic!("expected ty_trait")
        }
    }

    fn mutability_allowed(a_mutbl: ast::Mutability,
                          b_mutbl: ast::Mutability)
                          -> bool {
        a_mutbl == b_mutbl ||
            (a_mutbl == ast::MutMutable && b_mutbl == ast::MutImmutable)
    }

    fn push_cast_obligation<'a, 'tcx>(fcx: &FnCtxt<'a, 'tcx>,
                                      cast_expr: &ast::Expr,
                                      object_trait: &ty::TyTrait<'tcx>,
                                      referent_ty: Ty<'tcx>) {
        let object_trait_ref =
            register_object_cast_obligations(fcx,
                                             cast_expr.span,
                                             object_trait,
                                             referent_ty);

        // Finally record the object_trait_ref for use during trans
        // (it would prob be better not to do this, but it's just kind
        // of a pain to have to reconstruct it).
        fcx.write_object_cast(cast_expr.id, object_trait_ref);
    }
}

// Check that a trait is 'object-safe'. This should be checked whenever a trait object
// is created (by casting or coercion, etc.). A trait is object-safe if all its
// methods are object-safe. A trait method is object-safe if it does not take
// self by value, has no type parameters and does not use the `Self` type, except
// in self position.
pub fn check_object_safety<'tcx>(tcx: &ty::ctxt<'tcx>,
                                 object_trait: &ty::TyTrait<'tcx>,
                                 span: Span)
{
    // Also check that the type `object_trait` specifies all
    // associated types for all supertraits.
    let mut associated_types: FnvHashSet<(ast::DefId, ast::Name)> = FnvHashSet::new();

    let object_trait_ref =
        object_trait.principal_trait_ref_with_self_ty(tcx, tcx.types.err);
    for tr in traits::supertraits(tcx, object_trait_ref.clone()) {
        check_object_safety_inner(tcx, &tr, span);

        let trait_def = ty::lookup_trait_def(tcx, object_trait_ref.def_id());
        for &associated_type_name in trait_def.associated_type_names.iter() {
            associated_types.insert((object_trait_ref.def_id(), associated_type_name));
        }
    }

    for projection_bound in object_trait.bounds.projection_bounds.iter() {
        let pair = (projection_bound.0.projection_ty.trait_ref.def_id,
                    projection_bound.0.projection_ty.item_name);
        associated_types.remove(&pair);
    }

    for (trait_def_id, name) in associated_types.into_iter() {
        tcx.sess.span_err(
            span,
            format!("the value of the associated type `{}` (from the trait `{}`) must be specified",
                    name.user_string(tcx),
                    ty::item_path_str(tcx, trait_def_id)).as_slice());
    }
}

fn check_object_safety_inner<'tcx>(tcx: &ty::ctxt<'tcx>,
                                   object_trait: &ty::PolyTraitRef<'tcx>,
                                   span: Span) {
    let trait_items = ty::trait_items(tcx, object_trait.def_id());

    let mut errors = Vec::new();
    for item in trait_items.iter() {
        match *item {
            ty::MethodTraitItem(ref m) => {
                errors.push(check_object_safety_of_method(tcx, object_trait, &**m))
            }
            ty::TypeTraitItem(_) => {}
        }
    }

    let mut errors = errors.iter().flat_map(|x| x.iter()).peekable();
    if errors.peek().is_some() {
        let trait_name = ty::item_path_str(tcx, object_trait.def_id());
        span_err!(tcx.sess, span, E0038,
            "cannot convert to a trait object because trait `{}` is not object-safe",
            trait_name);

        for msg in errors {
            tcx.sess.note(msg[]);
        }
    }

    /// Returns a vec of error messages. If the vec is empty - no errors!
    ///
    /// There are some limitations to calling functions through an object, because (a) the self
    /// type is not known (that's the whole point of a trait instance, after all, to obscure the
    /// self type), (b) the call must go through a vtable and hence cannot be monomorphized and
    /// (c) the trait contains static methods which can't be called because we don't know the
    /// concrete type.
    fn check_object_safety_of_method<'tcx>(tcx: &ty::ctxt<'tcx>,
                                           object_trait: &ty::PolyTraitRef<'tcx>,
                                           method: &ty::Method<'tcx>)
                                           -> Vec<String> {
        let mut msgs = Vec::new();

        let method_name = method.name.repr(tcx);

        match method.explicit_self {
            ty::ByValueExplicitSelfCategory => { // reason (a) above
                msgs.push(format!("cannot call a method (`{}`) with a by-value \
                                   receiver through a trait object", method_name))
            }

            ty::StaticExplicitSelfCategory => {
                // Static methods are never object safe (reason (c)).
                msgs.push(format!("cannot call a static method (`{}`) \
                                   through a trait object",
                                  method_name));
                return msgs;
            }
            ty::ByReferenceExplicitSelfCategory(..) |
            ty::ByBoxExplicitSelfCategory => {}
        }

        // reason (a) above
        let check_for_self_ty = |ty| {
            if contains_illegal_self_type_reference(tcx, object_trait.def_id(), ty) {
                Some(format!(
                    "cannot call a method (`{}`) whose type contains \
                     a self-type (`{}`) through a trait object",
                    method_name, ty.user_string(tcx)))
            } else {
                None
            }
        };
        let ref sig = method.fty.sig;
        for &input_ty in sig.0.inputs[1..].iter() {
            if let Some(msg) = check_for_self_ty(input_ty) {
                msgs.push(msg);
            }
        }
        if let ty::FnConverging(result_type) = sig.0.output {
            if let Some(msg) = check_for_self_ty(result_type) {
                msgs.push(msg);
            }
        }

        if method.generics.has_type_params(FnSpace) {
            // reason (b) above
            msgs.push(format!("cannot call a generic method (`{}`) through a trait object",
                              method_name));
        }

        msgs
    }

    fn contains_illegal_self_type_reference<'tcx>(tcx: &ty::ctxt<'tcx>,
                                                  trait_def_id: ast::DefId,
                                                  ty: Ty<'tcx>)
                                                  -> bool
    {
        // This is somewhat subtle. In general, we want to forbid
        // references to `Self` in the argument and return types,
        // since the value of `Self` is erased. However, there is one
        // exception: it is ok to reference `Self` in order to access
        // an associated type of the current trait, since we retain
        // the value of those associated types in the object type
        // itself.
        //
        // ```rust
        // trait SuperTrait {
        //     type X;
        // }
        //
        // trait Trait : SuperTrait {
        //     type Y;
        //     fn foo(&self, x: Self) // bad
        //     fn foo(&self) -> Self // bad
        //     fn foo(&self) -> Option<Self> // bad
        //     fn foo(&self) -> Self::Y // OK, desugars to next example
        //     fn foo(&self) -> <Self as Trait>::Y // OK
        //     fn foo(&self) -> Self::X // OK, desugars to next example
        //     fn foo(&self) -> <Self as SuperTrait>::X // OK
        // }
        // ```
        //
        // However, it is not as simple as allowing `Self` in a projected
        // type, because there are illegal ways to use `Self` as well:
        //
        // ```rust
        // trait Trait : SuperTrait {
        //     ...
        //     fn foo(&self) -> <Self as SomeOtherTrait>::X;
        // }
        // ```
        //
        // Here we will not have the type of `X` recorded in the
        // object type, and we cannot resolve `Self as SomeOtherTrait`
        // without knowing what `Self` is.

        let mut supertraits: Option<Vec<ty::PolyTraitRef<'tcx>>> = None;
        let mut error = false;
        ty::maybe_walk_ty(ty, |ty| {
            match ty.sty {
                ty::ty_param(ref param_ty) => {
                    if param_ty.space == SelfSpace {
                        error = true;
                    }

                    false // no contained types to walk
                }

                ty::ty_projection(ref data) => {
                    // This is a projected type `<Foo as SomeTrait>::X`.

                    // Compute supertraits of current trait lazilly.
                    if supertraits.is_none() {
                        let trait_def = ty::lookup_trait_def(tcx, trait_def_id);
                        let trait_ref = ty::Binder(trait_def.trait_ref.clone());
                        supertraits = Some(traits::supertraits(tcx, trait_ref).collect());
                    }

                    // Determine whether the trait reference `Foo as
                    // SomeTrait` is in fact a supertrait of the
                    // current trait. In that case, this type is
                    // legal, because the type `X` will be specified
                    // in the object type.  Note that we can just use
                    // direct equality here because all of these types
                    // are part of the formal parameter listing, and
                    // hence there should be no inference variables.
                    let projection_trait_ref = ty::Binder(data.trait_ref.clone());
                    let is_supertrait_of_current_trait =
                        supertraits.as_ref().unwrap().contains(&projection_trait_ref);

                    if is_supertrait_of_current_trait {
                        false // do not walk contained types, do not report error, do collect $200
                    } else {
                        true // DO walk contained types, POSSIBLY reporting an error
                    }
                }

                _ => true, // walk contained types, if any
            }
        });

        error
    }
}

pub fn register_object_cast_obligations<'a, 'tcx>(fcx: &FnCtxt<'a, 'tcx>,
                                                  span: Span,
                                                  object_trait: &ty::TyTrait<'tcx>,
                                                  referent_ty: Ty<'tcx>)
                                                  -> ty::PolyTraitRef<'tcx>
{
    // We can only make objects from sized types.
    fcx.register_builtin_bound(
        referent_ty,
        ty::BoundSized,
        traits::ObligationCause::new(span, fcx.body_id, traits::ObjectSized));

    // This is just for better error reporting. Kinda goofy. The object type stuff
    // needs some refactoring so there is a more convenient type to pass around.
    let object_trait_ty =
        ty::mk_trait(fcx.tcx(),
                     object_trait.principal.clone(),
                     object_trait.bounds.clone());

    debug!("register_object_cast_obligations: referent_ty={} object_trait_ty={}",
           referent_ty.repr(fcx.tcx()),
           object_trait_ty.repr(fcx.tcx()));

    let cause = ObligationCause::new(span,
                                     fcx.body_id,
                                     traits::ObjectCastObligation(object_trait_ty));

    // Create the obligation for casting from T to Trait.
    let object_trait_ref =
        object_trait.principal_trait_ref_with_self_ty(fcx.tcx(), referent_ty);
    let object_obligation =
        Obligation::new(cause.clone(), object_trait_ref.as_predicate());
    fcx.register_predicate(object_obligation);

    // Create additional obligations for all the various builtin
    // bounds attached to the object cast. (In other words, if the
    // object type is Foo+Send, this would create an obligation
    // for the Send check.)
    for builtin_bound in object_trait.bounds.builtin_bounds.iter() {
        fcx.register_builtin_bound(
            referent_ty,
            builtin_bound,
            cause.clone());
    }

    // Finally, create obligations for the projection predicates.
    let projection_bounds =
        object_trait.projection_bounds_with_self_ty(fcx.tcx(), referent_ty);
    for projection_bound in projection_bounds.iter() {
        let projection_obligation =
            Obligation::new(cause.clone(), projection_bound.as_predicate());
        fcx.register_predicate(projection_obligation);
    }

    object_trait_ref
}

pub fn select_all_fcx_obligations_or_error(fcx: &FnCtxt) {
    debug!("select_all_fcx_obligations_or_error");

    let mut fulfillment_cx = fcx.inh.fulfillment_cx.borrow_mut();
    let r = fulfillment_cx.select_all_or_error(fcx.infcx(),
                                               &fcx.inh.param_env,
                                               fcx);
    match r {
        Ok(()) => { }
        Err(errors) => { report_fulfillment_errors(fcx.infcx(), &errors); }
    }
}

/// Select as many obligations as we can at present.
pub fn select_fcx_obligations_where_possible(fcx: &FnCtxt)
{
    match
        fcx.inh.fulfillment_cx
        .borrow_mut()
        .select_where_possible(fcx.infcx(), &fcx.inh.param_env, fcx)
    {
        Ok(()) => { }
        Err(errors) => { report_fulfillment_errors(fcx.infcx(), &errors); }
    }
}

/// Try to select any fcx obligation that we haven't tried yet, in an effort to improve inference.
/// You could just call `select_fcx_obligations_where_possible` except that it leads to repeated
/// work.
pub fn select_new_fcx_obligations(fcx: &FnCtxt) {
    match
        fcx.inh.fulfillment_cx
        .borrow_mut()
        .select_new_obligations(fcx.infcx(), &fcx.inh.param_env, fcx)
    {
        Ok(()) => { }
        Err(errors) => { report_fulfillment_errors(fcx.infcx(), &errors); }
    }
}

