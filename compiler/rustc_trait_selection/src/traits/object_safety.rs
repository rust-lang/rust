//! "Object safety" refers to the ability for a trait to be converted
//! to an object. In general, traits may only be converted to an
//! object if all of their methods meet certain criteria. In particular,
//! they must:
//!
//!   - have a suitable receiver from which we can extract a vtable and coerce to a "thin" version
//!     that doesn't contain the vtable;
//!   - not reference the erased type `Self` except for in this receiver;
//!   - not have generic type parameters.

use super::{elaborate_predicates, elaborate_trait_ref};

use crate::infer::TyCtxtInferExt;
use crate::traits::query::evaluate_obligation::InferCtxtExt;
use crate::traits::{self, Obligation, ObligationCause};
use hir::def::DefKind;
use rustc_errors::{DelayDm, FatalError, MultiSpan};
use rustc_hir as hir;
use rustc_hir::def_id::DefId;
use rustc_middle::ty::subst::{GenericArg, InternalSubsts};
use rustc_middle::ty::{
    self, EarlyBinder, Ty, TyCtxt, TypeSuperVisitable, TypeVisitable, TypeVisitor,
};
use rustc_middle::ty::{Predicate, ToPredicate};
use rustc_session::lint::builtin::WHERE_CLAUSES_OBJECT_SAFETY;
use rustc_span::symbol::Symbol;
use rustc_span::Span;
use smallvec::SmallVec;

use std::iter;
use std::ops::ControlFlow;

pub use crate::traits::{MethodViolationCode, ObjectSafetyViolation};

/// Returns the object safety violations that affect
/// astconv -- currently, `Self` in supertraits. This is needed
/// because `object_safety_violations` can't be used during
/// type collection.
pub fn astconv_object_safety_violations(
    tcx: TyCtxt<'_>,
    trait_def_id: DefId,
) -> Vec<ObjectSafetyViolation> {
    debug_assert!(tcx.generics_of(trait_def_id).has_self);
    let violations = traits::supertrait_def_ids(tcx, trait_def_id)
        .map(|def_id| predicates_reference_self(tcx, def_id, true))
        .filter(|spans| !spans.is_empty())
        .map(ObjectSafetyViolation::SupertraitSelf)
        .collect();

    debug!("astconv_object_safety_violations(trait_def_id={:?}) = {:?}", trait_def_id, violations);

    violations
}

fn object_safety_violations(tcx: TyCtxt<'_>, trait_def_id: DefId) -> &'_ [ObjectSafetyViolation] {
    debug_assert!(tcx.generics_of(trait_def_id).has_self);
    debug!("object_safety_violations: {:?}", trait_def_id);

    tcx.arena.alloc_from_iter(
        traits::supertrait_def_ids(tcx, trait_def_id)
            .flat_map(|def_id| object_safety_violations_for_trait(tcx, def_id)),
    )
}

fn check_is_object_safe(tcx: TyCtxt<'_>, trait_def_id: DefId) -> bool {
    let violations = tcx.object_safety_violations(trait_def_id);

    if violations.is_empty() {
        return true;
    }

    // If the trait contains any other violations, then let the error reporting path
    // report it instead of emitting a warning here.
    if violations.iter().all(|violation| {
        matches!(
            violation,
            ObjectSafetyViolation::Method(_, MethodViolationCode::WhereClauseReferencesSelf, _)
        )
    }) {
        for violation in violations {
            if let ObjectSafetyViolation::Method(
                _,
                MethodViolationCode::WhereClauseReferencesSelf,
                span,
            ) = violation
            {
                lint_object_unsafe_trait(tcx, *span, trait_def_id, &violation);
            }
        }
        return true;
    }

    false
}

/// We say a method is *vtable safe* if it can be invoked on a trait
/// object. Note that object-safe traits can have some
/// non-vtable-safe methods, so long as they require `Self: Sized` or
/// otherwise ensure that they cannot be used when `Self = Trait`.
pub fn is_vtable_safe_method(tcx: TyCtxt<'_>, trait_def_id: DefId, method: &ty::AssocItem) -> bool {
    debug_assert!(tcx.generics_of(trait_def_id).has_self);
    debug!("is_vtable_safe_method({:?}, {:?})", trait_def_id, method);
    // Any method that has a `Self: Sized` bound cannot be called.
    if generics_require_sized_self(tcx, method.def_id) {
        return false;
    }

    match virtual_call_violation_for_method(tcx, trait_def_id, method) {
        None | Some(MethodViolationCode::WhereClauseReferencesSelf) => true,
        Some(_) => false,
    }
}

fn object_safety_violations_for_trait(
    tcx: TyCtxt<'_>,
    trait_def_id: DefId,
) -> Vec<ObjectSafetyViolation> {
    // Check methods for violations.
    let mut violations: Vec<_> = tcx
        .associated_items(trait_def_id)
        .in_definition_order()
        .filter(|item| item.kind == ty::AssocKind::Fn)
        .filter_map(|item| {
            object_safety_violation_for_method(tcx, trait_def_id, &item)
                .map(|(code, span)| ObjectSafetyViolation::Method(item.name, code, span))
        })
        .collect();

    // Check the trait itself.
    if trait_has_sized_self(tcx, trait_def_id) {
        // We don't want to include the requirement from `Sized` itself to be `Sized` in the list.
        let spans = get_sized_bounds(tcx, trait_def_id);
        violations.push(ObjectSafetyViolation::SizedSelf(spans));
    }
    let spans = predicates_reference_self(tcx, trait_def_id, false);
    if !spans.is_empty() {
        violations.push(ObjectSafetyViolation::SupertraitSelf(spans));
    }
    let spans = bounds_reference_self(tcx, trait_def_id);
    if !spans.is_empty() {
        violations.push(ObjectSafetyViolation::SupertraitSelf(spans));
    }

    violations.extend(
        tcx.associated_items(trait_def_id)
            .in_definition_order()
            .filter(|item| item.kind == ty::AssocKind::Const)
            .map(|item| {
                let ident = item.ident(tcx);
                ObjectSafetyViolation::AssocConst(ident.name, ident.span)
            }),
    );

    if !tcx.features().generic_associated_types_extended {
        violations.extend(
            tcx.associated_items(trait_def_id)
                .in_definition_order()
                .filter(|item| item.kind == ty::AssocKind::Type)
                .filter(|item| !tcx.generics_of(item.def_id).params.is_empty())
                .map(|item| {
                    let ident = item.ident(tcx);
                    ObjectSafetyViolation::GAT(ident.name, ident.span)
                }),
        );
    }

    debug!(
        "object_safety_violations_for_trait(trait_def_id={:?}) = {:?}",
        trait_def_id, violations
    );

    violations
}

/// Lint object-unsafe trait.
fn lint_object_unsafe_trait(
    tcx: TyCtxt<'_>,
    span: Span,
    trait_def_id: DefId,
    violation: &ObjectSafetyViolation,
) {
    // Using `CRATE_NODE_ID` is wrong, but it's hard to get a more precise id.
    // It's also hard to get a use site span, so we use the method definition span.
    tcx.struct_span_lint_hir(
        WHERE_CLAUSES_OBJECT_SAFETY,
        hir::CRATE_HIR_ID,
        span,
        DelayDm(|| format!("the trait `{}` cannot be made into an object", tcx.def_path_str(trait_def_id))),
        |err| {
            let node = tcx.hir().get_if_local(trait_def_id);
            let mut spans = MultiSpan::from_span(span);
            if let Some(hir::Node::Item(item)) = node {
                spans.push_span_label(
                    item.ident.span,
                    "this trait cannot be made into an object...",
                );
                spans.push_span_label(span, format!("...because {}", violation.error_msg()));
            } else {
                spans.push_span_label(
                    span,
                    format!(
                        "the trait cannot be made into an object because {}",
                        violation.error_msg()
                    ),
                );
            };
            err.span_note(
                spans,
                "for a trait to be \"object safe\" it needs to allow building a vtable to allow the \
                call to be resolvable dynamically; for more information visit \
                <https://doc.rust-lang.org/reference/items/traits.html#object-safety>",
            );
            if node.is_some() {
                // Only provide the help if its a local trait, otherwise it's not
                violation.solution(err);
            }
            err
        },
    );
}

fn sized_trait_bound_spans<'tcx>(
    tcx: TyCtxt<'tcx>,
    bounds: hir::GenericBounds<'tcx>,
) -> impl 'tcx + Iterator<Item = Span> {
    bounds.iter().filter_map(move |b| match b {
        hir::GenericBound::Trait(trait_ref, hir::TraitBoundModifier::None)
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
    tcx.hir()
        .get_if_local(trait_def_id)
        .and_then(|node| match node {
            hir::Node::Item(hir::Item {
                kind: hir::ItemKind::Trait(.., generics, bounds, _),
                ..
            }) => Some(
                generics
                    .predicates
                    .iter()
                    .filter_map(|pred| {
                        match pred {
                            hir::WherePredicate::BoundPredicate(pred)
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
    let trait_ref = ty::TraitRef::identity(tcx, trait_def_id);
    let predicates = if supertraits_only {
        tcx.super_predicates_of(trait_def_id)
    } else {
        tcx.predicates_of(trait_def_id)
    };
    predicates
        .predicates
        .iter()
        .map(|&(predicate, sp)| (predicate.subst_supertrait(tcx, &trait_ref), sp))
        .filter_map(|predicate| predicate_references_self(tcx, predicate))
        .collect()
}

fn bounds_reference_self(tcx: TyCtxt<'_>, trait_def_id: DefId) -> SmallVec<[Span; 1]> {
    tcx.associated_items(trait_def_id)
        .in_definition_order()
        .filter(|item| item.kind == ty::AssocKind::Type)
        .flat_map(|item| tcx.explicit_item_bounds(item.def_id))
        .filter_map(|pred_span| predicate_references_self(tcx, *pred_span))
        .collect()
}

fn predicate_references_self<'tcx>(
    tcx: TyCtxt<'tcx>,
    (predicate, sp): (ty::Predicate<'tcx>, Span),
) -> Option<Span> {
    let self_ty = tcx.types.self_param;
    let has_self_ty = |arg: &GenericArg<'tcx>| arg.walk().any(|arg| arg == self_ty.into());
    match predicate.kind().skip_binder() {
        ty::PredicateKind::Clause(ty::Clause::Trait(ref data)) => {
            // In the case of a trait predicate, we can skip the "self" type.
            if data.trait_ref.substs[1..].iter().any(has_self_ty) { Some(sp) } else { None }
        }
        ty::PredicateKind::Clause(ty::Clause::Projection(ref data)) => {
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
            if data.projection_ty.substs[1..].iter().any(has_self_ty) { Some(sp) } else { None }
        }
        ty::PredicateKind::AliasEq(..) => bug!("`AliasEq` not allowed as assumption"),

        ty::PredicateKind::WellFormed(..)
        | ty::PredicateKind::ObjectSafe(..)
        | ty::PredicateKind::Clause(ty::Clause::TypeOutlives(..))
        | ty::PredicateKind::Clause(ty::Clause::RegionOutlives(..))
        | ty::PredicateKind::ClosureKind(..)
        | ty::PredicateKind::Subtype(..)
        | ty::PredicateKind::Coerce(..)
        // FIXME(generic_const_exprs): this can mention `Self`
        | ty::PredicateKind::ConstEvaluatable(..)
        | ty::PredicateKind::ConstEquate(..)
        | ty::PredicateKind::Ambiguous
        | ty::PredicateKind::TypeWellFormedFromEnv(..) => None,
    }
}

fn trait_has_sized_self(tcx: TyCtxt<'_>, trait_def_id: DefId) -> bool {
    generics_require_sized_self(tcx, trait_def_id)
}

fn generics_require_sized_self(tcx: TyCtxt<'_>, def_id: DefId) -> bool {
    let Some(sized_def_id) = tcx.lang_items().sized_trait() else {
        return false; /* No Sized trait, can't require it! */
    };

    // Search for a predicate like `Self : Sized` amongst the trait bounds.
    let predicates = tcx.predicates_of(def_id);
    let predicates = predicates.instantiate_identity(tcx).predicates;
    elaborate_predicates(tcx, predicates.into_iter()).any(|obligation| {
        match obligation.predicate.kind().skip_binder() {
            ty::PredicateKind::Clause(ty::Clause::Trait(ref trait_pred)) => {
                trait_pred.def_id() == sized_def_id && trait_pred.self_ty().is_param(0)
            }
            ty::PredicateKind::Clause(ty::Clause::Projection(..))
            | ty::PredicateKind::Subtype(..)
            | ty::PredicateKind::Coerce(..)
            | ty::PredicateKind::Clause(ty::Clause::RegionOutlives(..))
            | ty::PredicateKind::WellFormed(..)
            | ty::PredicateKind::ObjectSafe(..)
            | ty::PredicateKind::ClosureKind(..)
            | ty::PredicateKind::Clause(ty::Clause::TypeOutlives(..))
            | ty::PredicateKind::ConstEvaluatable(..)
            | ty::PredicateKind::ConstEquate(..)
            | ty::PredicateKind::AliasEq(..)
            | ty::PredicateKind::Ambiguous
            | ty::PredicateKind::TypeWellFormedFromEnv(..) => false,
        }
    })
}

/// Returns `Some(_)` if this method makes the containing trait not object safe.
fn object_safety_violation_for_method(
    tcx: TyCtxt<'_>,
    trait_def_id: DefId,
    method: &ty::AssocItem,
) -> Option<(MethodViolationCode, Span)> {
    debug!("object_safety_violation_for_method({:?}, {:?})", trait_def_id, method);
    // Any method that has a `Self : Sized` requisite is otherwise
    // exempt from the regulations.
    if generics_require_sized_self(tcx, method.def_id) {
        return None;
    }

    let violation = virtual_call_violation_for_method(tcx, trait_def_id, method);
    // Get an accurate span depending on the violation.
    violation.map(|v| {
        let node = tcx.hir().get_if_local(method.def_id);
        let span = match (&v, node) {
            (MethodViolationCode::ReferencesSelfInput(Some(span)), _) => *span,
            (MethodViolationCode::UndispatchableReceiver(Some(span)), _) => *span,
            (MethodViolationCode::ReferencesImplTraitInTrait(span), _) => *span,
            (MethodViolationCode::ReferencesSelfOutput, Some(node)) => {
                node.fn_decl().map_or(method.ident(tcx).span, |decl| decl.output.span())
            }
            _ => method.ident(tcx).span,
        };
        (v, span)
    })
}

/// Returns `Some(_)` if this method cannot be called on a trait
/// object; this does not necessarily imply that the enclosing trait
/// is not object safe, because the method might have a where clause
/// `Self:Sized`.
fn virtual_call_violation_for_method<'tcx>(
    tcx: TyCtxt<'tcx>,
    trait_def_id: DefId,
    method: &ty::AssocItem,
) -> Option<MethodViolationCode> {
    let sig = tcx.fn_sig(method.def_id).subst_identity();

    // The method's first parameter must be named `self`
    if !method.fn_has_self_parameter {
        let sugg = if let Some(hir::Node::TraitItem(hir::TraitItem {
            generics,
            kind: hir::TraitItemKind::Fn(sig, _),
            ..
        })) = tcx.hir().get_if_local(method.def_id).as_ref()
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
        return Some(MethodViolationCode::StaticMethod(sugg));
    }

    for (i, &input_ty) in sig.skip_binder().inputs().iter().enumerate().skip(1) {
        if contains_illegal_self_type_reference(tcx, trait_def_id, sig.rebind(input_ty)) {
            let span = if let Some(hir::Node::TraitItem(hir::TraitItem {
                kind: hir::TraitItemKind::Fn(sig, _),
                ..
            })) = tcx.hir().get_if_local(method.def_id).as_ref()
            {
                Some(sig.decl.inputs[i].span)
            } else {
                None
            };
            return Some(MethodViolationCode::ReferencesSelfInput(span));
        }
    }
    if contains_illegal_self_type_reference(tcx, trait_def_id, sig.output()) {
        return Some(MethodViolationCode::ReferencesSelfOutput);
    }
    if let Some(code) = contains_illegal_impl_trait_in_trait(tcx, method.def_id, sig.output()) {
        return Some(code);
    }

    // We can't monomorphize things like `fn foo<A>(...)`.
    let own_counts = tcx.generics_of(method.def_id).own_counts();
    if own_counts.types + own_counts.consts != 0 {
        return Some(MethodViolationCode::Generic);
    }

    let receiver_ty = tcx.liberate_late_bound_regions(method.def_id, sig.input(0));

    // Until `unsized_locals` is fully implemented, `self: Self` can't be dispatched on.
    // However, this is already considered object-safe. We allow it as a special case here.
    // FIXME(mikeyhew) get rid of this `if` statement once `receiver_is_dispatchable` allows
    // `Receiver: Unsize<Receiver[Self => dyn Trait]>`.
    if receiver_ty != tcx.types.self_param {
        if !receiver_is_dispatchable(tcx, method, receiver_ty) {
            let span = if let Some(hir::Node::TraitItem(hir::TraitItem {
                kind: hir::TraitItemKind::Fn(sig, _),
                ..
            })) = tcx.hir().get_if_local(method.def_id).as_ref()
            {
                Some(sig.decl.inputs[0].span)
            } else {
                None
            };
            return Some(MethodViolationCode::UndispatchableReceiver(span));
        } else {
            // Do sanity check to make sure the receiver actually has the layout of a pointer.

            use rustc_target::abi::Abi;

            let param_env = tcx.param_env(method.def_id);

            let abi_of_ty = |ty: Ty<'tcx>| -> Option<Abi> {
                match tcx.layout_of(param_env.and(ty)) {
                    Ok(layout) => Some(layout.abi),
                    Err(err) => {
                        // #78372
                        tcx.sess.delay_span_bug(
                            tcx.def_span(method.def_id),
                            &format!("error: {}\n while computing layout for type {:?}", err, ty),
                        );
                        None
                    }
                }
            };

            // e.g., `Rc<()>`
            let unit_receiver_ty =
                receiver_for_self_ty(tcx, receiver_ty, tcx.mk_unit(), method.def_id);

            match abi_of_ty(unit_receiver_ty) {
                Some(Abi::Scalar(..)) => (),
                abi => {
                    tcx.sess.delay_span_bug(
                        tcx.def_span(method.def_id),
                        &format!(
                            "receiver when `Self = ()` should have a Scalar ABI; found {:?}",
                            abi
                        ),
                    );
                }
            }

            let trait_object_ty =
                object_ty_for_trait(tcx, trait_def_id, tcx.mk_region(ty::ReStatic));

            // e.g., `Rc<dyn Trait>`
            let trait_object_receiver =
                receiver_for_self_ty(tcx, receiver_ty, trait_object_ty, method.def_id);

            match abi_of_ty(trait_object_receiver) {
                Some(Abi::ScalarPair(..)) => (),
                abi => {
                    tcx.sess.delay_span_bug(
                        tcx.def_span(method.def_id),
                        &format!(
                            "receiver when `Self = {}` should have a ScalarPair ABI; found {:?}",
                            trait_object_ty, abi
                        ),
                    );
                }
            }
        }
    }

    // NOTE: This check happens last, because it results in a lint, and not a
    // hard error.
    if tcx.predicates_of(method.def_id).predicates.iter().any(|&(pred, span)| {
        // dyn Trait is okay:
        //
        //     trait Trait {
        //         fn f(&self) where Self: 'static;
        //     }
        //
        // because a trait object can't claim to live longer than the concrete
        // type. If the lifetime bound holds on dyn Trait then it's guaranteed
        // to hold as well on the concrete type.
        if pred.to_opt_type_outlives().is_some() {
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
        if let ty::PredicateKind::Clause(ty::Clause::Trait(ty::TraitPredicate {
            trait_ref: pred_trait_ref,
            constness: ty::BoundConstness::NotConst,
            polarity: ty::ImplPolarity::Positive,
        })) = pred.kind().skip_binder()
            && pred_trait_ref.self_ty() == tcx.types.self_param
            && tcx.trait_is_auto(pred_trait_ref.def_id)
        {
            // Consider bounds like `Self: Bound<Self>`. Auto traits are not
            // allowed to have generic parameters so `auto trait Bound<T> {}`
            // would already have reported an error at the definition of the
            // auto trait.
            if pred_trait_ref.substs.len() != 1 {
                tcx.sess.diagnostic().delay_span_bug(
                    span,
                    "auto traits cannot have generic parameters",
                );
            }
            return false;
        }

        contains_illegal_self_type_reference(tcx, trait_def_id, pred.clone())
    }) {
        return Some(MethodViolationCode::WhereClauseReferencesSelf);
    }

    None
}

/// Performs a type substitution to produce the version of `receiver_ty` when `Self = self_ty`.
/// For example, for `receiver_ty = Rc<Self>` and `self_ty = Foo`, returns `Rc<Foo>`.
fn receiver_for_self_ty<'tcx>(
    tcx: TyCtxt<'tcx>,
    receiver_ty: Ty<'tcx>,
    self_ty: Ty<'tcx>,
    method_def_id: DefId,
) -> Ty<'tcx> {
    debug!("receiver_for_self_ty({:?}, {:?}, {:?})", receiver_ty, self_ty, method_def_id);
    let substs = InternalSubsts::for_item(tcx, method_def_id, |param, _| {
        if param.index == 0 { self_ty.into() } else { tcx.mk_param_from_def(param) }
    });

    let result = EarlyBinder(receiver_ty).subst(tcx, substs);
    debug!(
        "receiver_for_self_ty({:?}, {:?}, {:?}) = {:?}",
        receiver_ty, self_ty, method_def_id, result
    );
    result
}

/// Creates the object type for the current trait. For example,
/// if the current trait is `Deref`, then this will be
/// `dyn Deref<Target = Self::Target> + 'static`.
#[instrument(level = "trace", skip(tcx), ret)]
fn object_ty_for_trait<'tcx>(
    tcx: TyCtxt<'tcx>,
    trait_def_id: DefId,
    lifetime: ty::Region<'tcx>,
) -> Ty<'tcx> {
    let trait_ref = ty::TraitRef::identity(tcx, trait_def_id);
    debug!(?trait_ref);

    let trait_predicate = trait_ref.map_bound(|trait_ref| {
        ty::ExistentialPredicate::Trait(ty::ExistentialTraitRef::erase_self_ty(tcx, trait_ref))
    });
    debug!(?trait_predicate);

    let mut elaborated_predicates: Vec<_> = elaborate_trait_ref(tcx, trait_ref)
        .filter_map(|obligation| {
            debug!(?obligation);
            let pred = obligation.predicate.to_opt_poly_projection_pred()?;
            Some(pred.map_bound(|p| {
                ty::ExistentialPredicate::Projection(ty::ExistentialProjection::erase_self_ty(
                    tcx, p,
                ))
            }))
        })
        .collect();
    // NOTE: Since #37965, the existential predicates list has depended on the
    // list of predicates to be sorted. This is mostly to enforce that the primary
    // predicate comes first.
    elaborated_predicates.sort_by(|a, b| a.skip_binder().stable_cmp(tcx, &b.skip_binder()));
    elaborated_predicates.dedup();

    let existential_predicates = tcx
        .mk_poly_existential_predicates(iter::once(trait_predicate).chain(elaborated_predicates));
    debug!(?existential_predicates);

    tcx.mk_dynamic(existential_predicates, lifetime, ty::Dyn)
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
///   (substitution notation).
///
/// Some examples of receiver types and their required obligation:
/// - `&'a mut self` requires `&'a mut Self: DispatchFromDyn<&'a mut dyn Trait>`,
/// - `self: Rc<Self>` requires `Rc<Self>: DispatchFromDyn<Rc<dyn Trait>>`,
/// - `self: Pin<Box<Self>>` requires `Pin<Box<Self>>: DispatchFromDyn<Pin<Box<dyn Trait>>>`.
///
/// The only case where the receiver is not dispatchable, but is still a valid receiver
/// type (just not object-safe), is when there is more than one level of pointer indirection.
/// E.g., `self: &&Self`, `self: &Rc<Self>`, `self: Box<Box<Self>>`. In these cases, there
/// is no way, or at least no inexpensive way, to coerce the receiver from the version where
/// `Self = dyn Trait` to the version where `Self = T`, where `T` is the unknown erased type
/// contained by the trait object, because the object that needs to be coerced is behind
/// a pointer.
///
/// In practice, we cannot use `dyn Trait` explicitly in the obligation because it would result
/// in a new check that `Trait` is object safe, creating a cycle (until object_safe_for_dispatch
/// is stabilized, see tracking issue <https://github.com/rust-lang/rust/issues/43561>).
/// Instead, we fudge a little by introducing a new type parameter `U` such that
/// `Self: Unsize<U>` and `U: Trait + ?Sized`, and use `U` in place of `dyn Trait`.
/// Written as a chalk-style query:
/// ```ignore (not-rust)
/// forall (U: Trait + ?Sized) {
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
#[allow(dead_code)]
fn receiver_is_dispatchable<'tcx>(
    tcx: TyCtxt<'tcx>,
    method: &ty::AssocItem,
    receiver_ty: Ty<'tcx>,
) -> bool {
    debug!("receiver_is_dispatchable: method = {:?}, receiver_ty = {:?}", method, receiver_ty);

    let traits = (tcx.lang_items().unsize_trait(), tcx.lang_items().dispatch_from_dyn_trait());
    let (Some(unsize_did), Some(dispatch_from_dyn_did)) = traits else {
        debug!("receiver_is_dispatchable: Missing Unsize or DispatchFromDyn traits");
        return false;
    };

    // the type `U` in the query
    // use a bogus type parameter to mimic a forall(U) query using u32::MAX for now.
    // FIXME(mikeyhew) this is a total hack. Once object_safe_for_dispatch is stabilized, we can
    // replace this with `dyn Trait`
    let unsized_self_ty: Ty<'tcx> =
        tcx.mk_ty_param(u32::MAX, Symbol::intern("RustaceansAreAwesome"));

    // `Receiver[Self => U]`
    let unsized_receiver_ty =
        receiver_for_self_ty(tcx, receiver_ty, unsized_self_ty, method.def_id);

    // create a modified param env, with `Self: Unsize<U>` and `U: Trait` added to caller bounds
    // `U: ?Sized` is already implied here
    let param_env = {
        let param_env = tcx.param_env(method.def_id);

        // Self: Unsize<U>
        let unsize_predicate = ty::Binder::dummy(
            tcx.mk_trait_ref(unsize_did, [tcx.types.self_param, unsized_self_ty]),
        )
        .without_const()
        .to_predicate(tcx);

        // U: Trait<Arg1, ..., ArgN>
        let trait_predicate = {
            let trait_def_id = method.trait_container(tcx).unwrap();
            let substs = InternalSubsts::for_item(tcx, trait_def_id, |param, _| {
                if param.index == 0 { unsized_self_ty.into() } else { tcx.mk_param_from_def(param) }
            });

            ty::Binder::dummy(tcx.mk_trait_ref(trait_def_id, substs)).to_predicate(tcx)
        };

        let caller_bounds: Vec<Predicate<'tcx>> =
            param_env.caller_bounds().iter().chain([unsize_predicate, trait_predicate]).collect();

        ty::ParamEnv::new(
            tcx.intern_predicates(&caller_bounds),
            param_env.reveal(),
            param_env.constness(),
        )
    };

    // Receiver: DispatchFromDyn<Receiver[Self => U]>
    let obligation = {
        let predicate = ty::Binder::dummy(
            tcx.mk_trait_ref(dispatch_from_dyn_did, [receiver_ty, unsized_receiver_ty]),
        );

        Obligation::new(tcx, ObligationCause::dummy(), param_env, predicate)
    };

    let infcx = tcx.infer_ctxt().build();
    // the receiver is dispatchable iff the obligation holds
    infcx.predicate_must_hold_modulo_regions(&obligation)
}

fn contains_illegal_self_type_reference<'tcx, T: TypeVisitable<'tcx>>(
    tcx: TyCtxt<'tcx>,
    trait_def_id: DefId,
    value: T,
) -> bool {
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

    struct IllegalSelfTypeVisitor<'tcx> {
        tcx: TyCtxt<'tcx>,
        trait_def_id: DefId,
        supertraits: Option<Vec<DefId>>,
    }

    impl<'tcx> TypeVisitor<'tcx> for IllegalSelfTypeVisitor<'tcx> {
        type BreakTy = ();

        fn visit_ty(&mut self, t: Ty<'tcx>) -> ControlFlow<Self::BreakTy> {
            match t.kind() {
                ty::Param(_) => {
                    if t == self.tcx.types.self_param {
                        ControlFlow::Break(())
                    } else {
                        ControlFlow::Continue(())
                    }
                }
                ty::Alias(ty::Projection, ref data)
                    if self.tcx.def_kind(data.def_id) == DefKind::ImplTraitPlaceholder =>
                {
                    // We'll deny these later in their own pass
                    ControlFlow::Continue(())
                }
                ty::Alias(ty::Projection, ref data) => {
                    // This is a projected type `<Foo as SomeTrait>::X`.

                    // Compute supertraits of current trait lazily.
                    if self.supertraits.is_none() {
                        let trait_ref = ty::TraitRef::identity(self.tcx, self.trait_def_id);
                        self.supertraits = Some(
                            traits::supertraits(self.tcx, trait_ref).map(|t| t.def_id()).collect(),
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
                    let is_supertrait_of_current_trait = self
                        .supertraits
                        .as_ref()
                        .unwrap()
                        .contains(&data.trait_ref(self.tcx).def_id);

                    if is_supertrait_of_current_trait {
                        ControlFlow::Continue(()) // do not walk contained types, do not report error, do collect $200
                    } else {
                        t.super_visit_with(self) // DO walk contained types, POSSIBLY reporting an error
                    }
                }
                _ => t.super_visit_with(self), // walk contained types, if any
            }
        }

        fn visit_const(&mut self, ct: ty::Const<'tcx>) -> ControlFlow<Self::BreakTy> {
            // Constants can only influence object safety if they are generic and reference `Self`.
            // This is only possible for unevaluated constants, so we walk these here.
            self.tcx.expand_abstract_consts(ct).super_visit_with(self)
        }
    }

    value
        .visit_with(&mut IllegalSelfTypeVisitor { tcx, trait_def_id, supertraits: None })
        .is_break()
}

pub fn contains_illegal_impl_trait_in_trait<'tcx>(
    tcx: TyCtxt<'tcx>,
    fn_def_id: DefId,
    ty: ty::Binder<'tcx, Ty<'tcx>>,
) -> Option<MethodViolationCode> {
    // This would be caught below, but rendering the error as a separate
    // `async-specific` message is better.
    if tcx.asyncness(fn_def_id).is_async() {
        return Some(MethodViolationCode::AsyncFn);
    }

    // FIXME(RPITIT): Perhaps we should use a visitor here?
    ty.skip_binder().walk().find_map(|arg| {
        if let ty::GenericArgKind::Type(ty) = arg.unpack()
            && let ty::Alias(ty::Projection, proj) = ty.kind()
            && tcx.def_kind(proj.def_id) == DefKind::ImplTraitPlaceholder
        {
            Some(MethodViolationCode::ReferencesImplTraitInTrait(tcx.def_span(proj.def_id)))
        } else {
            None
        }
    })
}

pub fn provide(providers: &mut ty::query::Providers) {
    *providers =
        ty::query::Providers { object_safety_violations, check_is_object_safe, ..*providers };
}
