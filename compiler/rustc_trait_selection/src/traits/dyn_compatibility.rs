//! "Dyn-compatibility"[^1] refers to the ability for a trait to be converted
//! to a trait object. In general, traits may only be converted to a trait
//! object if certain criteria are met.
//!
//! [^1]: Formerly known as "object safety".

use std::ops::ControlFlow;

use rustc_errors::FatalError;
use rustc_hir::def_id::DefId;
use rustc_hir::{self as hir, LangItem};
use rustc_middle::query::Providers;
use rustc_middle::ty::{
    self, EarlyBinder, GenericArgs, Ty, TyCtxt, TypeFoldable, TypeFolder, TypeSuperFoldable,
    TypeSuperVisitable, TypeVisitable, TypeVisitableExt, TypeVisitor, TypingMode, Upcast,
    elaborate,
};
use rustc_span::{DUMMY_SP, Span};
use smallvec::SmallVec;
use tracing::{debug, instrument};

use super::elaborate;
use crate::infer::TyCtxtInferExt;
pub use crate::traits::DynCompatibilityViolation;
use crate::traits::query::evaluate_obligation::InferCtxtExt;
use crate::traits::{
    MethodViolationCode, Obligation, ObligationCause, normalize_param_env_or_error, util,
};

/// Returns the dyn-compatibility violations that affect HIR ty lowering.
///
/// Currently that is `Self` in supertraits. This is needed
/// because `dyn_compatibility_violations` can't be used during
/// type collection, as type collection is needed for `dyn_compatiblity_violations` itself.
#[instrument(level = "debug", skip(tcx), ret)]
pub fn hir_ty_lowering_dyn_compatibility_violations(
    tcx: TyCtxt<'_>,
    trait_def_id: DefId,
) -> Vec<DynCompatibilityViolation> {
    debug_assert!(tcx.generics_of(trait_def_id).has_self);
    elaborate::supertrait_def_ids(tcx, trait_def_id)
        .map(|def_id| predicates_reference_self(tcx, def_id, true))
        .filter(|spans| !spans.is_empty())
        .map(DynCompatibilityViolation::SupertraitSelf)
        .collect()
}

fn dyn_compatibility_violations(
    tcx: TyCtxt<'_>,
    trait_def_id: DefId,
) -> &'_ [DynCompatibilityViolation] {
    debug_assert!(tcx.generics_of(trait_def_id).has_self);
    debug!("dyn_compatibility_violations: {:?}", trait_def_id);
    tcx.arena.alloc_from_iter(
        elaborate::supertrait_def_ids(tcx, trait_def_id)
            .flat_map(|def_id| dyn_compatibility_violations_for_trait(tcx, def_id)),
    )
}

fn is_dyn_compatible(tcx: TyCtxt<'_>, trait_def_id: DefId) -> bool {
    tcx.dyn_compatibility_violations(trait_def_id).is_empty()
}

/// We say a method is *vtable safe* if it can be invoked on a trait
/// object. Note that dyn-compatible traits can have some
/// non-vtable-safe methods, so long as they require `Self: Sized` or
/// otherwise ensure that they cannot be used when `Self = Trait`.
pub fn is_vtable_safe_method(tcx: TyCtxt<'_>, trait_def_id: DefId, method: ty::AssocItem) -> bool {
    debug_assert!(tcx.generics_of(trait_def_id).has_self);
    debug!("is_vtable_safe_method({:?}, {:?})", trait_def_id, method);
    // Any method that has a `Self: Sized` bound cannot be called.
    if tcx.generics_require_sized_self(method.def_id) {
        return false;
    }

    virtual_call_violations_for_method(tcx, trait_def_id, method).is_empty()
}

#[instrument(level = "debug", skip(tcx), ret)]
fn dyn_compatibility_violations_for_trait(
    tcx: TyCtxt<'_>,
    trait_def_id: DefId,
) -> Vec<DynCompatibilityViolation> {
    // Check assoc items for violations.
    let mut violations: Vec<_> = tcx
        .associated_items(trait_def_id)
        .in_definition_order()
        .flat_map(|&item| dyn_compatibility_violations_for_assoc_item(tcx, trait_def_id, item))
        .collect();

    // Check the trait itself.
    if trait_has_sized_self(tcx, trait_def_id) {
        // We don't want to include the requirement from `Sized` itself to be `Sized` in the list.
        let spans = get_sized_bounds(tcx, trait_def_id);
        violations.push(DynCompatibilityViolation::SizedSelf(spans));
    }
    let spans = predicates_reference_self(tcx, trait_def_id, false);
    if !spans.is_empty() {
        violations.push(DynCompatibilityViolation::SupertraitSelf(spans));
    }
    let spans = bounds_reference_self(tcx, trait_def_id);
    if !spans.is_empty() {
        violations.push(DynCompatibilityViolation::SupertraitSelf(spans));
    }
    let spans = super_predicates_have_non_lifetime_binders(tcx, trait_def_id);
    if !spans.is_empty() {
        violations.push(DynCompatibilityViolation::SupertraitNonLifetimeBinder(spans));
    }

    violations
}

fn sized_trait_bound_spans<'tcx>(
    tcx: TyCtxt<'tcx>,
    bounds: hir::GenericBounds<'tcx>,
) -> impl 'tcx + Iterator<Item = Span> {
    bounds.iter().filter_map(move |b| match b {
        hir::GenericBound::Trait(trait_ref)
            if trait_has_sized_self(
                tcx,
                trait_ref.trait_ref.trait_def_id().unwrap_or_else(|| FatalError.raise()),
            ) =>
        {
            // Fetch spans for supertraits that are `Sized`: `trait T: Super`
            Some(trait_ref.span)
        }
        _ => None,
    })
}

fn get_sized_bounds(tcx: TyCtxt<'_>, trait_def_id: DefId) -> SmallVec<[Span; 1]> {
    tcx.hir_get_if_local(trait_def_id)
        .and_then(|node| match node {
            hir::Node::Item(hir::Item {
                kind: hir::ItemKind::Trait(.., generics, bounds, _),
                ..
            }) => Some(
                generics
                    .predicates
                    .iter()
                    .filter_map(|pred| {
                        match pred.kind {
                            hir::WherePredicateKind::BoundPredicate(pred)
                                if pred.bounded_ty.hir_id.owner.to_def_id() == trait_def_id =>
                            {
                                // Fetch spans for trait bounds that are Sized:
                                // `trait T where Self: Pred`
                                Some(sized_trait_bound_spans(tcx, pred.bounds))
                            }
                            _ => None,
                        }
                    })
                    .flatten()
                    // Fetch spans for supertraits that are `Sized`: `trait T: Super`.
                    .chain(sized_trait_bound_spans(tcx, bounds))
                    .collect::<SmallVec<[Span; 1]>>(),
            ),
            _ => None,
        })
        .unwrap_or_else(SmallVec::new)
}

fn predicates_reference_self(
    tcx: TyCtxt<'_>,
    trait_def_id: DefId,
    supertraits_only: bool,
) -> SmallVec<[Span; 1]> {
    let trait_ref = ty::Binder::dummy(ty::TraitRef::identity(tcx, trait_def_id));
    let predicates = if supertraits_only {
        tcx.explicit_super_predicates_of(trait_def_id).skip_binder()
    } else {
        tcx.predicates_of(trait_def_id).predicates
    };
    predicates
        .iter()
        .map(|&(predicate, sp)| (predicate.instantiate_supertrait(tcx, trait_ref), sp))
        .filter_map(|(clause, sp)| {
            // Super predicates cannot allow self projections, since they're
            // impossible to make into existential bounds without eager resolution
            // or something.
            // e.g. `trait A: B<Item = Self::Assoc>`.
            predicate_references_self(tcx, trait_def_id, clause, sp, AllowSelfProjections::No)
        })
        .collect()
}

fn bounds_reference_self(tcx: TyCtxt<'_>, trait_def_id: DefId) -> SmallVec<[Span; 1]> {
    tcx.associated_items(trait_def_id)
        .in_definition_order()
        // We're only looking at associated type bounds
        .filter(|item| item.is_type())
        // Ignore GATs with `Self: Sized`
        .filter(|item| !tcx.generics_require_sized_self(item.def_id))
        .flat_map(|item| tcx.explicit_item_bounds(item.def_id).iter_identity_copied())
        .filter_map(|(clause, sp)| {
            // Item bounds *can* have self projections, since they never get
            // their self type erased.
            predicate_references_self(tcx, trait_def_id, clause, sp, AllowSelfProjections::Yes)
        })
        .collect()
}

fn predicate_references_self<'tcx>(
    tcx: TyCtxt<'tcx>,
    trait_def_id: DefId,
    predicate: ty::Clause<'tcx>,
    sp: Span,
    allow_self_projections: AllowSelfProjections,
) -> Option<Span> {
    match predicate.kind().skip_binder() {
        ty::ClauseKind::Trait(ref data) => {
            // In the case of a trait predicate, we can skip the "self" type.
            data.trait_ref.args[1..].iter().any(|&arg| contains_illegal_self_type_reference(tcx, trait_def_id, arg, allow_self_projections)).then_some(sp)
        }
        ty::ClauseKind::Projection(ref data) => {
            // And similarly for projections. This should be redundant with
            // the previous check because any projection should have a
            // matching `Trait` predicate with the same inputs, but we do
            // the check to be safe.
            //
            // It's also won't be redundant if we allow type-generic associated
            // types for trait objects.
            //
            // Note that we *do* allow projection *outputs* to contain
            // `self` (i.e., `trait Foo: Bar<Output=Self::Result> { type Result; }`),
            // we just require the user to specify *both* outputs
            // in the object type (i.e., `dyn Foo<Output=(), Result=()>`).
            //
            // This is ALT2 in issue #56288, see that for discussion of the
            // possible alternatives.
            data.projection_term.args[1..].iter().any(|&arg| contains_illegal_self_type_reference(tcx, trait_def_id, arg, allow_self_projections)).then_some(sp)
        }
        ty::ClauseKind::ConstArgHasType(_ct, ty) => contains_illegal_self_type_reference(tcx, trait_def_id, ty, allow_self_projections).then_some(sp),

        ty::ClauseKind::WellFormed(..)
        | ty::ClauseKind::TypeOutlives(..)
        | ty::ClauseKind::RegionOutlives(..)
        // FIXME(generic_const_exprs): this can mention `Self`
        | ty::ClauseKind::ConstEvaluatable(..)
        | ty::ClauseKind::HostEffect(..)
         => None,
    }
}

fn super_predicates_have_non_lifetime_binders(
    tcx: TyCtxt<'_>,
    trait_def_id: DefId,
) -> SmallVec<[Span; 1]> {
    // If non_lifetime_binders is disabled, then exit early
    if !tcx.features().non_lifetime_binders() {
        return SmallVec::new();
    }
    tcx.explicit_super_predicates_of(trait_def_id)
        .iter_identity_copied()
        .filter_map(|(pred, span)| pred.has_non_region_bound_vars().then_some(span))
        .collect()
}

fn trait_has_sized_self(tcx: TyCtxt<'_>, trait_def_id: DefId) -> bool {
    tcx.generics_require_sized_self(trait_def_id)
}

fn generics_require_sized_self(tcx: TyCtxt<'_>, def_id: DefId) -> bool {
    let Some(sized_def_id) = tcx.lang_items().sized_trait() else {
        return false; /* No Sized trait, can't require it! */
    };

    // Search for a predicate like `Self : Sized` amongst the trait bounds.
    let predicates = tcx.predicates_of(def_id);
    let predicates = predicates.instantiate_identity(tcx).predicates;
    elaborate(tcx, predicates).any(|pred| match pred.kind().skip_binder() {
        ty::ClauseKind::Trait(ref trait_pred) => {
            trait_pred.def_id() == sized_def_id && trait_pred.self_ty().is_param(0)
        }
        ty::ClauseKind::RegionOutlives(_)
        | ty::ClauseKind::TypeOutlives(_)
        | ty::ClauseKind::Projection(_)
        | ty::ClauseKind::ConstArgHasType(_, _)
        | ty::ClauseKind::WellFormed(_)
        | ty::ClauseKind::ConstEvaluatable(_)
        | ty::ClauseKind::HostEffect(..) => false,
    })
}

/// Returns `Some(_)` if this item makes the containing trait dyn-incompatible.
#[instrument(level = "debug", skip(tcx), ret)]
pub fn dyn_compatibility_violations_for_assoc_item(
    tcx: TyCtxt<'_>,
    trait_def_id: DefId,
    item: ty::AssocItem,
) -> Vec<DynCompatibilityViolation> {
    // Any item that has a `Self : Sized` requisite is otherwise
    // exempt from the regulations.
    if tcx.generics_require_sized_self(item.def_id) {
        return Vec::new();
    }

    match item.kind {
        // Associated consts are never dyn-compatible, as they can't have `where` bounds yet at all,
        // and associated const bounds in trait objects aren't a thing yet either.
        ty::AssocKind::Const { name } => {
            vec![DynCompatibilityViolation::AssocConst(name, item.ident(tcx).span)]
        }
        ty::AssocKind::Fn { name, .. } => {
            virtual_call_violations_for_method(tcx, trait_def_id, item)
                .into_iter()
                .map(|v| {
                    let node = tcx.hir_get_if_local(item.def_id);
                    // Get an accurate span depending on the violation.
                    let span = match (&v, node) {
                        (MethodViolationCode::ReferencesSelfInput(Some(span)), _) => *span,
                        (MethodViolationCode::UndispatchableReceiver(Some(span)), _) => *span,
                        (MethodViolationCode::ReferencesImplTraitInTrait(span), _) => *span,
                        (MethodViolationCode::ReferencesSelfOutput, Some(node)) => {
                            node.fn_decl().map_or(item.ident(tcx).span, |decl| decl.output.span())
                        }
                        _ => item.ident(tcx).span,
                    };

                    DynCompatibilityViolation::Method(name, v, span)
                })
                .collect()
        }
        // Associated types can only be dyn-compatible if they have `Self: Sized` bounds.
        ty::AssocKind::Type { .. } => {
            if !tcx.generics_of(item.def_id).is_own_empty() && !item.is_impl_trait_in_trait() {
                vec![DynCompatibilityViolation::GAT(item.name(), item.ident(tcx).span)]
            } else {
                // We will permit associated types if they are explicitly mentioned in the trait object.
                // We can't check this here, as here we only check if it is guaranteed to not be possible.
                Vec::new()
            }
        }
    }
}

/// Returns `Some(_)` if this method cannot be called on a trait
/// object; this does not necessarily imply that the enclosing trait
/// is dyn-incompatible, because the method might have a where clause
/// `Self: Sized`.
fn virtual_call_violations_for_method<'tcx>(
    tcx: TyCtxt<'tcx>,
    trait_def_id: DefId,
    method: ty::AssocItem,
) -> Vec<MethodViolationCode> {
    let sig = tcx.fn_sig(method.def_id).instantiate_identity();

    // The method's first parameter must be named `self`
    if !method.is_method() {
        let sugg = if let Some(hir::Node::TraitItem(hir::TraitItem {
            generics,
            kind: hir::TraitItemKind::Fn(sig, _),
            ..
        })) = tcx.hir_get_if_local(method.def_id).as_ref()
        {
            let sm = tcx.sess.source_map();
            Some((
                (
                    format!("&self{}", if sig.decl.inputs.is_empty() { "" } else { ", " }),
                    sm.span_through_char(sig.span, '(').shrink_to_hi(),
                ),
                (
                    format!("{} Self: Sized", generics.add_where_or_trailing_comma()),
                    generics.tail_span_for_predicate_suggestion(),
                ),
            ))
        } else {
            None
        };

        // Not having `self` parameter messes up the later checks,
        // so we need to return instead of pushing
        return vec![MethodViolationCode::StaticMethod(sugg)];
    }

    let mut errors = Vec::new();

    for (i, &input_ty) in sig.skip_binder().inputs().iter().enumerate().skip(1) {
        if contains_illegal_self_type_reference(
            tcx,
            trait_def_id,
            sig.rebind(input_ty),
            AllowSelfProjections::Yes,
        ) {
            let span = if let Some(hir::Node::TraitItem(hir::TraitItem {
                kind: hir::TraitItemKind::Fn(sig, _),
                ..
            })) = tcx.hir_get_if_local(method.def_id).as_ref()
            {
                Some(sig.decl.inputs[i].span)
            } else {
                None
            };
            errors.push(MethodViolationCode::ReferencesSelfInput(span));
        }
    }
    if contains_illegal_self_type_reference(
        tcx,
        trait_def_id,
        sig.output(),
        AllowSelfProjections::Yes,
    ) {
        errors.push(MethodViolationCode::ReferencesSelfOutput);
    }
    if let Some(code) = contains_illegal_impl_trait_in_trait(tcx, method.def_id, sig.output()) {
        errors.push(code);
    }

    // We can't monomorphize things like `fn foo<A>(...)`.
    let own_counts = tcx.generics_of(method.def_id).own_counts();
    if own_counts.types > 0 || own_counts.consts > 0 {
        errors.push(MethodViolationCode::Generic);
    }

    let receiver_ty = tcx.liberate_late_bound_regions(method.def_id, sig.input(0));

    // `self: Self` can't be dispatched on.
    // However, this is considered dyn compatible. We allow it as a special case here.
    // FIXME(mikeyhew) get rid of this `if` statement once `receiver_is_dispatchable` allows
    // `Receiver: Unsize<Receiver[Self => dyn Trait]>`.
    if receiver_ty != tcx.types.self_param {
        if !receiver_is_dispatchable(tcx, method, receiver_ty) {
            let span = if let Some(hir::Node::TraitItem(hir::TraitItem {
                kind: hir::TraitItemKind::Fn(sig, _),
                ..
            })) = tcx.hir_get_if_local(method.def_id).as_ref()
            {
                Some(sig.decl.inputs[0].span)
            } else {
                None
            };
            errors.push(MethodViolationCode::UndispatchableReceiver(span));
        } else {
            // We confirm that the `receiver_is_dispatchable` is accurate later,
            // see `check_receiver_correct`. It should be kept in sync with this code.
        }
    }

    // NOTE: This check happens last, because it results in a lint, and not a
    // hard error.
    if tcx.predicates_of(method.def_id).predicates.iter().any(|&(pred, _span)| {
        // dyn Trait is okay:
        //
        //     trait Trait {
        //         fn f(&self) where Self: 'static;
        //     }
        //
        // because a trait object can't claim to live longer than the concrete
        // type. If the lifetime bound holds on dyn Trait then it's guaranteed
        // to hold as well on the concrete type.
        if pred.as_type_outlives_clause().is_some() {
            return false;
        }

        // dyn Trait is okay:
        //
        //     auto trait AutoTrait {}
        //
        //     trait Trait {
        //         fn f(&self) where Self: AutoTrait;
        //     }
        //
        // because `impl AutoTrait for dyn Trait` is disallowed by coherence.
        // Traits with a default impl are implemented for a trait object if and
        // only if the autotrait is one of the trait object's trait bounds, like
        // in `dyn Trait + AutoTrait`. This guarantees that trait objects only
        // implement auto traits if the underlying type does as well.
        if let ty::ClauseKind::Trait(ty::TraitPredicate {
            trait_ref: pred_trait_ref,
            polarity: ty::PredicatePolarity::Positive,
        }) = pred.kind().skip_binder()
            && pred_trait_ref.self_ty() == tcx.types.self_param
            && tcx.trait_is_auto(pred_trait_ref.def_id)
        {
            // Consider bounds like `Self: Bound<Self>`. Auto traits are not
            // allowed to have generic parameters so `auto trait Bound<T> {}`
            // would already have reported an error at the definition of the
            // auto trait.
            if pred_trait_ref.args.len() != 1 {
                assert!(
                    tcx.dcx().has_errors().is_some(),
                    "auto traits cannot have generic parameters"
                );
            }
            return false;
        }

        contains_illegal_self_type_reference(tcx, trait_def_id, pred, AllowSelfProjections::Yes)
    }) {
        errors.push(MethodViolationCode::WhereClauseReferencesSelf);
    }

    errors
}

/// Performs a type instantiation to produce the version of `receiver_ty` when `Self = self_ty`.
/// For example, for `receiver_ty = Rc<Self>` and `self_ty = Foo`, returns `Rc<Foo>`.
fn receiver_for_self_ty<'tcx>(
    tcx: TyCtxt<'tcx>,
    receiver_ty: Ty<'tcx>,
    self_ty: Ty<'tcx>,
    method_def_id: DefId,
) -> Ty<'tcx> {
    debug!("receiver_for_self_ty({:?}, {:?}, {:?})", receiver_ty, self_ty, method_def_id);
    let args = GenericArgs::for_item(tcx, method_def_id, |param, _| {
        if param.index == 0 { self_ty.into() } else { tcx.mk_param_from_def(param) }
    });

    let result = EarlyBinder::bind(receiver_ty).instantiate(tcx, args);
    debug!(
        "receiver_for_self_ty({:?}, {:?}, {:?}) = {:?}",
        receiver_ty, self_ty, method_def_id, result
    );
    result
}

/// Checks the method's receiver (the `self` argument) can be dispatched on when `Self` is a
/// trait object. We require that `DispatchableFromDyn` be implemented for the receiver type
/// in the following way:
/// - let `Receiver` be the type of the `self` argument, i.e `Self`, `&Self`, `Rc<Self>`,
/// - require the following bound:
///
///   ```ignore (not-rust)
///   Receiver[Self => T]: DispatchFromDyn<Receiver[Self => dyn Trait]>
///   ```
///
///   where `Foo[X => Y]` means "the same type as `Foo`, but with `X` replaced with `Y`"
///   (instantiation notation).
///
/// Some examples of receiver types and their required obligation:
/// - `&'a mut self` requires `&'a mut Self: DispatchFromDyn<&'a mut dyn Trait>`,
/// - `self: Rc<Self>` requires `Rc<Self>: DispatchFromDyn<Rc<dyn Trait>>`,
/// - `self: Pin<Box<Self>>` requires `Pin<Box<Self>>: DispatchFromDyn<Pin<Box<dyn Trait>>>`.
///
/// The only case where the receiver is not dispatchable, but is still a valid receiver
/// type (just not dyn compatible), is when there is more than one level of pointer indirection.
/// E.g., `self: &&Self`, `self: &Rc<Self>`, `self: Box<Box<Self>>`. In these cases, there
/// is no way, or at least no inexpensive way, to coerce the receiver from the version where
/// `Self = dyn Trait` to the version where `Self = T`, where `T` is the unknown erased type
/// contained by the trait object, because the object that needs to be coerced is behind
/// a pointer.
///
/// In practice, we cannot use `dyn Trait` explicitly in the obligation because it would result in
/// a new check that `Trait` is dyn-compatible, creating a cycle.
/// Instead, we emulate a placeholder by introducing a new type parameter `U` such that
/// `Self: Unsize<U>` and `U: Trait + MetaSized`, and use `U` in place of `dyn Trait`.
///
/// Written as a chalk-style query:
/// ```ignore (not-rust)
/// forall (U: Trait + MetaSized) {
///     if (Self: Unsize<U>) {
///         Receiver: DispatchFromDyn<Receiver[Self => U]>
///     }
/// }
/// ```
/// for `self: &'a mut Self`, this means `&'a mut Self: DispatchFromDyn<&'a mut U>`
/// for `self: Rc<Self>`, this means `Rc<Self>: DispatchFromDyn<Rc<U>>`
/// for `self: Pin<Box<Self>>`, this means `Pin<Box<Self>>: DispatchFromDyn<Pin<Box<U>>>`
//
// FIXME(mikeyhew) when unsized receivers are implemented as part of unsized rvalues, add this
// fallback query: `Receiver: Unsize<Receiver[Self => U]>` to support receivers like
// `self: Wrapper<Self>`.
fn receiver_is_dispatchable<'tcx>(
    tcx: TyCtxt<'tcx>,
    method: ty::AssocItem,
    receiver_ty: Ty<'tcx>,
) -> bool {
    debug!("receiver_is_dispatchable: method = {:?}, receiver_ty = {:?}", method, receiver_ty);

    let (Some(unsize_did), Some(dispatch_from_dyn_did)) =
        (tcx.lang_items().unsize_trait(), tcx.lang_items().dispatch_from_dyn_trait())
    else {
        debug!("receiver_is_dispatchable: Missing `Unsize` or `DispatchFromDyn` traits");
        return false;
    };

    // the type `U` in the query
    // use a bogus type parameter to mimic a forall(U) query using u32::MAX for now.
    let unsized_self_ty: Ty<'tcx> =
        Ty::new_param(tcx, u32::MAX, rustc_span::sym::RustaceansAreAwesome);

    // `Receiver[Self => U]`
    let unsized_receiver_ty =
        receiver_for_self_ty(tcx, receiver_ty, unsized_self_ty, method.def_id);

    // create a modified param env, with `Self: Unsize<U>` and `U: Trait` (and all of
    // its supertraits) added to caller bounds. `U: MetaSized` is already implied here.
    let param_env = {
        // N.B. We generally want to emulate the construction of the `unnormalized_param_env`
        // in the param-env query here. The fact that we don't just start with the clauses
        // in the param-env of the method is because those are already normalized, and mixing
        // normalized and unnormalized copies of predicates in `normalize_param_env_or_error`
        // will cause ambiguity that the user can't really avoid.
        //
        // We leave out certain complexities of the param-env query here. Specifically, we:
        // 1. Do not add `~const` bounds since there are no `dyn const Trait`s.
        // 2. Do not add RPITIT self projection bounds for defaulted methods, since we
        //    are not constructing a param-env for "inside" of the body of the defaulted
        //    method, so we don't really care about projecting to a specific RPIT type,
        //    and because RPITITs are not dyn compatible (yet).
        let mut predicates = tcx.predicates_of(method.def_id).instantiate_identity(tcx).predicates;

        // Self: Unsize<U>
        let unsize_predicate =
            ty::TraitRef::new(tcx, unsize_did, [tcx.types.self_param, unsized_self_ty]);
        predicates.push(unsize_predicate.upcast(tcx));

        // U: Trait<Arg1, ..., ArgN>
        let trait_def_id = method.trait_container(tcx).unwrap();
        let args = GenericArgs::for_item(tcx, trait_def_id, |param, _| {
            if param.index == 0 { unsized_self_ty.into() } else { tcx.mk_param_from_def(param) }
        });
        let trait_predicate = ty::TraitRef::new_from_args(tcx, trait_def_id, args);
        predicates.push(trait_predicate.upcast(tcx));

        let meta_sized_predicate = {
            let meta_sized_did = tcx.require_lang_item(LangItem::MetaSized, DUMMY_SP);
            ty::TraitRef::new(tcx, meta_sized_did, [unsized_self_ty]).upcast(tcx)
        };
        predicates.push(meta_sized_predicate);

        normalize_param_env_or_error(
            tcx,
            ty::ParamEnv::new(tcx.mk_clauses(&predicates)),
            ObligationCause::dummy_with_span(tcx.def_span(method.def_id)),
        )
    };

    // Receiver: DispatchFromDyn<Receiver[Self => U]>
    let obligation = {
        let predicate =
            ty::TraitRef::new(tcx, dispatch_from_dyn_did, [receiver_ty, unsized_receiver_ty]);

        Obligation::new(tcx, ObligationCause::dummy(), param_env, predicate)
    };

    let infcx = tcx.infer_ctxt().build(TypingMode::non_body_analysis());
    // the receiver is dispatchable iff the obligation holds
    infcx.predicate_must_hold_modulo_regions(&obligation)
}

#[derive(Copy, Clone)]
enum AllowSelfProjections {
    Yes,
    No,
}

/// This is somewhat subtle. In general, we want to forbid
/// references to `Self` in the argument and return types,
/// since the value of `Self` is erased. However, there is one
/// exception: it is ok to reference `Self` in order to access
/// an associated type of the current trait, since we retain
/// the value of those associated types in the object type
/// itself.
///
/// ```rust,ignore (example)
/// trait SuperTrait {
///     type X;
/// }
///
/// trait Trait : SuperTrait {
///     type Y;
///     fn foo(&self, x: Self) // bad
///     fn foo(&self) -> Self // bad
///     fn foo(&self) -> Option<Self> // bad
///     fn foo(&self) -> Self::Y // OK, desugars to next example
///     fn foo(&self) -> <Self as Trait>::Y // OK
///     fn foo(&self) -> Self::X // OK, desugars to next example
///     fn foo(&self) -> <Self as SuperTrait>::X // OK
/// }
/// ```
///
/// However, it is not as simple as allowing `Self` in a projected
/// type, because there are illegal ways to use `Self` as well:
///
/// ```rust,ignore (example)
/// trait Trait : SuperTrait {
///     ...
///     fn foo(&self) -> <Self as SomeOtherTrait>::X;
/// }
/// ```
///
/// Here we will not have the type of `X` recorded in the
/// object type, and we cannot resolve `Self as SomeOtherTrait`
/// without knowing what `Self` is.
fn contains_illegal_self_type_reference<'tcx, T: TypeVisitable<TyCtxt<'tcx>>>(
    tcx: TyCtxt<'tcx>,
    trait_def_id: DefId,
    value: T,
    allow_self_projections: AllowSelfProjections,
) -> bool {
    value
        .visit_with(&mut IllegalSelfTypeVisitor {
            tcx,
            trait_def_id,
            supertraits: None,
            allow_self_projections,
        })
        .is_break()
}

struct IllegalSelfTypeVisitor<'tcx> {
    tcx: TyCtxt<'tcx>,
    trait_def_id: DefId,
    supertraits: Option<Vec<ty::TraitRef<'tcx>>>,
    allow_self_projections: AllowSelfProjections,
}

impl<'tcx> TypeVisitor<TyCtxt<'tcx>> for IllegalSelfTypeVisitor<'tcx> {
    type Result = ControlFlow<()>;

    fn visit_ty(&mut self, t: Ty<'tcx>) -> Self::Result {
        match t.kind() {
            ty::Param(_) => {
                if t == self.tcx.types.self_param {
                    ControlFlow::Break(())
                } else {
                    ControlFlow::Continue(())
                }
            }
            ty::Alias(ty::Projection, data) if self.tcx.is_impl_trait_in_trait(data.def_id) => {
                // We'll deny these later in their own pass
                ControlFlow::Continue(())
            }
            ty::Alias(ty::Projection, data) => {
                match self.allow_self_projections {
                    AllowSelfProjections::Yes => {
                        // This is a projected type `<Foo as SomeTrait>::X`.

                        // Compute supertraits of current trait lazily.
                        if self.supertraits.is_none() {
                            self.supertraits = Some(
                                util::supertraits(
                                    self.tcx,
                                    ty::Binder::dummy(ty::TraitRef::identity(
                                        self.tcx,
                                        self.trait_def_id,
                                    )),
                                )
                                .map(|trait_ref| {
                                    self.tcx.erase_regions(
                                        self.tcx.instantiate_bound_regions_with_erased(trait_ref),
                                    )
                                })
                                .collect(),
                            );
                        }

                        // Determine whether the trait reference `Foo as
                        // SomeTrait` is in fact a supertrait of the
                        // current trait. In that case, this type is
                        // legal, because the type `X` will be specified
                        // in the object type. Note that we can just use
                        // direct equality here because all of these types
                        // are part of the formal parameter listing, and
                        // hence there should be no inference variables.
                        let is_supertrait_of_current_trait =
                            self.supertraits.as_ref().unwrap().contains(
                                &data.trait_ref(self.tcx).fold_with(
                                    &mut EraseEscapingBoundRegions {
                                        tcx: self.tcx,
                                        binder: ty::INNERMOST,
                                    },
                                ),
                            );

                        // only walk contained types if it's not a super trait
                        if is_supertrait_of_current_trait {
                            ControlFlow::Continue(())
                        } else {
                            t.super_visit_with(self) // POSSIBLY reporting an error
                        }
                    }
                    AllowSelfProjections::No => t.super_visit_with(self),
                }
            }
            _ => t.super_visit_with(self),
        }
    }

    fn visit_const(&mut self, ct: ty::Const<'tcx>) -> Self::Result {
        // Constants can only influence dyn-compatibility if they are generic and reference `Self`.
        // This is only possible for unevaluated constants, so we walk these here.
        self.tcx.expand_abstract_consts(ct).super_visit_with(self)
    }
}

struct EraseEscapingBoundRegions<'tcx> {
    tcx: TyCtxt<'tcx>,
    binder: ty::DebruijnIndex,
}

impl<'tcx> TypeFolder<TyCtxt<'tcx>> for EraseEscapingBoundRegions<'tcx> {
    fn cx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn fold_binder<T>(&mut self, t: ty::Binder<'tcx, T>) -> ty::Binder<'tcx, T>
    where
        T: TypeFoldable<TyCtxt<'tcx>>,
    {
        self.binder.shift_in(1);
        let result = t.super_fold_with(self);
        self.binder.shift_out(1);
        result
    }

    fn fold_region(&mut self, r: ty::Region<'tcx>) -> ty::Region<'tcx> {
        if let ty::ReBound(debruijn, _) = r.kind()
            && debruijn < self.binder
        {
            r
        } else {
            self.tcx.lifetimes.re_erased
        }
    }
}

fn contains_illegal_impl_trait_in_trait<'tcx>(
    tcx: TyCtxt<'tcx>,
    fn_def_id: DefId,
    ty: ty::Binder<'tcx, Ty<'tcx>>,
) -> Option<MethodViolationCode> {
    let ty = tcx.liberate_late_bound_regions(fn_def_id, ty);

    if tcx.asyncness(fn_def_id).is_async() {
        // Rendering the error as a separate `async-specific` message is better.
        Some(MethodViolationCode::AsyncFn)
    } else {
        ty.visit_with(&mut IllegalRpititVisitor { tcx, allowed: None }).break_value()
    }
}

struct IllegalRpititVisitor<'tcx> {
    tcx: TyCtxt<'tcx>,
    allowed: Option<ty::AliasTy<'tcx>>,
}

impl<'tcx> TypeVisitor<TyCtxt<'tcx>> for IllegalRpititVisitor<'tcx> {
    type Result = ControlFlow<MethodViolationCode>;

    fn visit_ty(&mut self, ty: Ty<'tcx>) -> Self::Result {
        if let ty::Alias(ty::Projection, proj) = *ty.kind()
            && Some(proj) != self.allowed
            && self.tcx.is_impl_trait_in_trait(proj.def_id)
        {
            ControlFlow::Break(MethodViolationCode::ReferencesImplTraitInTrait(
                self.tcx.def_span(proj.def_id),
            ))
        } else {
            ty.super_visit_with(self)
        }
    }
}

pub(crate) fn provide(providers: &mut Providers) {
    *providers = Providers {
        dyn_compatibility_violations,
        is_dyn_compatible,
        generics_require_sized_self,
        ..*providers
    };
}
