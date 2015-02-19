// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! "Object safety" refers to the ability for a trait to be converted
//! to an object. In general, traits may only be converted to an
//! object if all of their methods meet certain criteria. In particular,
//! they must:
//!
//!   - have a suitable receiver from which we can extract a vtable;
//!   - not reference the erased type `Self` except for in this receiver;
//!   - not have generic type parameters

use super::supertraits;
use super::elaborate_predicates;

use middle::subst::{self, SelfSpace, TypeSpace};
use middle::traits;
use middle::ty::{self, Ty};
use std::rc::Rc;
use syntax::ast;
use util::ppaux::Repr;

pub enum ObjectSafetyViolation<'tcx> {
    /// Self : Sized declared on the trait
    SizedSelf,

    /// Supertrait reference references `Self` an in illegal location
    /// (e.g. `trait Foo : Bar<Self>`)
    SupertraitSelf,

    /// Method has something illegal
    Method(Rc<ty::Method<'tcx>>, MethodViolationCode),
}

/// Reasons a method might not be object-safe.
#[derive(Copy,Clone,Debug)]
pub enum MethodViolationCode {
    /// e.g., `fn(self)`
    ByValueSelf,

    /// e.g., `fn foo()`
    StaticMethod,

    /// e.g., `fn foo(&self, x: Self)` or `fn foo(&self) -> Self`
    ReferencesSelf,

    /// e.g., `fn foo<A>()`
    Generic,
}

pub fn is_object_safe<'tcx>(tcx: &ty::ctxt<'tcx>,
                            trait_ref: ty::PolyTraitRef<'tcx>)
                            -> bool
{
    // Because we query yes/no results frequently, we keep a cache:
    let cached_result =
        tcx.object_safety_cache.borrow().get(&trait_ref.def_id()).cloned();

    let result =
        cached_result.unwrap_or_else(|| {
            let result = object_safety_violations(tcx, trait_ref.clone()).is_empty();

            // Record just a yes/no result in the cache; this is what is
            // queried most frequently. Note that this may overwrite a
            // previous result, but always with the same thing.
            tcx.object_safety_cache.borrow_mut().insert(trait_ref.def_id(), result);

            result
        });

    debug!("is_object_safe({}) = {}", trait_ref.repr(tcx), result);

    result
}

pub fn object_safety_violations<'tcx>(tcx: &ty::ctxt<'tcx>,
                                      sub_trait_ref: ty::PolyTraitRef<'tcx>)
                                      -> Vec<ObjectSafetyViolation<'tcx>>
{
    supertraits(tcx, sub_trait_ref)
        .flat_map(|tr| object_safety_violations_for_trait(tcx, tr.def_id()).into_iter())
        .collect()
}

fn object_safety_violations_for_trait<'tcx>(tcx: &ty::ctxt<'tcx>,
                                            trait_def_id: ast::DefId)
                                            -> Vec<ObjectSafetyViolation<'tcx>>
{
    // Check methods for violations.
    let mut violations: Vec<_> =
        ty::trait_items(tcx, trait_def_id).iter()
        .flat_map(|item| {
            match *item {
                ty::MethodTraitItem(ref m) => {
                    object_safety_violations_for_method(tcx, trait_def_id, &**m)
                        .map(|code| ObjectSafetyViolation::Method(m.clone(), code))
                        .into_iter()
                }
                ty::TypeTraitItem(_) => {
                    None.into_iter()
                }
            }
        })
        .collect();

    // Check the trait itself.
    if trait_has_sized_self(tcx, trait_def_id) {
        violations.push(ObjectSafetyViolation::SizedSelf);
    }
    if supertraits_reference_self(tcx, trait_def_id) {
        violations.push(ObjectSafetyViolation::SupertraitSelf);
    }

    debug!("object_safety_violations_for_trait(trait_def_id={}) = {}",
           trait_def_id.repr(tcx),
           violations.repr(tcx));

    violations
}

fn supertraits_reference_self<'tcx>(tcx: &ty::ctxt<'tcx>,
                                    trait_def_id: ast::DefId)
                                    -> bool
{
    let trait_def = ty::lookup_trait_def(tcx, trait_def_id);
    let trait_ref = trait_def.trait_ref.clone();
    let predicates = ty::predicates_for_trait_ref(tcx, &ty::Binder(trait_ref));
    predicates
        .into_iter()
        .any(|predicate| {
            match predicate {
                ty::Predicate::Trait(ref data) => {
                    // In the case of a trait predicate, we can skip the "self" type.
                    Some(data.def_id()) != tcx.lang_items.phantom_fn() &&
                        data.0.trait_ref.substs.types.get_slice(TypeSpace)
                                                     .iter()
                                                     .cloned()
                                                     .any(is_self)
                }
                ty::Predicate::Projection(..) |
                ty::Predicate::TypeOutlives(..) |
                ty::Predicate::RegionOutlives(..) |
                ty::Predicate::Equate(..) => {
                    false
                }
            }
        })
}

fn trait_has_sized_self<'tcx>(tcx: &ty::ctxt<'tcx>,
                              trait_def_id: ast::DefId)
                              -> bool
{
    let sized_def_id = match tcx.lang_items.sized_trait() {
        Some(def_id) => def_id,
        None => { return false; /* No Sized trait, can't require it! */ }
    };

    // Search for a predicate like `Self : Sized` amongst the trait bounds.
    let trait_def = ty::lookup_trait_def(tcx, trait_def_id);
    let free_substs = ty::construct_free_substs(tcx, &trait_def.generics, ast::DUMMY_NODE_ID);

    let trait_predicates = ty::lookup_predicates(tcx, trait_def_id);
    let predicates = trait_predicates.instantiate(tcx, &free_substs).predicates.into_vec();

    elaborate_predicates(tcx, predicates)
        .any(|predicate| {
            match predicate {
                ty::Predicate::Trait(ref trait_pred) if trait_pred.def_id() == sized_def_id => {
                    is_self(trait_pred.0.self_ty())
                }
                ty::Predicate::Projection(..) |
                ty::Predicate::Trait(..) |
                ty::Predicate::Equate(..) |
                ty::Predicate::RegionOutlives(..) |
                ty::Predicate::TypeOutlives(..) => {
                    false
                }
            }
        })
}

fn object_safety_violations_for_method<'tcx>(tcx: &ty::ctxt<'tcx>,
                                             trait_def_id: ast::DefId,
                                             method: &ty::Method<'tcx>)
                                             -> Option<MethodViolationCode>
{
    // The method's first parameter must be something that derefs to
    // `&self`. For now, we only accept `&self` and `Box<Self>`.
    match method.explicit_self {
        ty::ByValueExplicitSelfCategory => {
            return Some(MethodViolationCode::ByValueSelf);
        }

        ty::StaticExplicitSelfCategory => {
            return Some(MethodViolationCode::StaticMethod);
        }

        ty::ByReferenceExplicitSelfCategory(..) |
        ty::ByBoxExplicitSelfCategory => {
        }
    }

    // The `Self` type is erased, so it should not appear in list of
    // arguments or return type apart from the receiver.
    let ref sig = method.fty.sig;
    for &input_ty in &sig.0.inputs[1..] {
        if contains_illegal_self_type_reference(tcx, trait_def_id, input_ty) {
            return Some(MethodViolationCode::ReferencesSelf);
        }
    }
    if let ty::FnConverging(result_type) = sig.0.output {
        if contains_illegal_self_type_reference(tcx, trait_def_id, result_type) {
            return Some(MethodViolationCode::ReferencesSelf);
        }
    }

    // We can't monomorphize things like `fn foo<A>(...)`.
    if !method.generics.types.is_empty_in(subst::FnSpace) {
        return Some(MethodViolationCode::Generic);
    }

    None
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

                // Compute supertraits of current trait lazily.
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

impl<'tcx> Repr<'tcx> for ObjectSafetyViolation<'tcx> {
    fn repr(&self, tcx: &ty::ctxt<'tcx>) -> String {
        match *self {
            ObjectSafetyViolation::SizedSelf =>
                format!("SizedSelf"),
            ObjectSafetyViolation::SupertraitSelf =>
                format!("SupertraitSelf"),
            ObjectSafetyViolation::Method(ref m, code) =>
                format!("Method({},{:?})", m.repr(tcx), code),
        }
    }
}

fn is_self<'tcx>(ty: Ty<'tcx>) -> bool {
    match ty.sty {
        ty::ty_param(ref data) => data.space == subst::SelfSpace,
        _ => false,
    }
}
