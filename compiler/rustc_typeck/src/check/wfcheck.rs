use crate::check::{FnCtxt, Inherited};
use crate::constrained_generic_params::{identify_constrained_generic_params, Parameter};

use rustc_ast as ast;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_errors::{struct_span_err, Applicability, DiagnosticBuilder};
use rustc_hir as hir;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_hir::intravisit as hir_visit;
use rustc_hir::intravisit::Visitor;
use rustc_hir::itemlikevisit::ParItemLikeVisitor;
use rustc_hir::lang_items::LangItem;
use rustc_hir::ItemKind;
use rustc_middle::hir::map as hir_map;
use rustc_middle::ty::subst::{GenericArgKind, InternalSubsts, Subst};
use rustc_middle::ty::trait_def::TraitSpecializationKind;
use rustc_middle::ty::{
    self, AdtKind, GenericParamDefKind, ToPredicate, Ty, TyCtxt, TypeFoldable, WithConstness,
};
use rustc_session::parse::feature_err;
use rustc_span::symbol::{sym, Ident, Symbol};
use rustc_span::Span;
use rustc_trait_selection::opaque_types::may_define_opaque_type;
use rustc_trait_selection::traits::query::evaluate_obligation::InferCtxtExt;
use rustc_trait_selection::traits::{self, ObligationCause, ObligationCauseCode};

use std::ops::ControlFlow;

/// Helper type of a temporary returned by `.for_item(...)`.
/// This is necessary because we can't write the following bound:
///
/// ```rust
/// F: for<'b, 'tcx> where 'tcx FnOnce(FnCtxt<'b, 'tcx>)
/// ```
struct CheckWfFcxBuilder<'tcx> {
    inherited: super::InheritedBuilder<'tcx>,
    id: hir::HirId,
    span: Span,
    param_env: ty::ParamEnv<'tcx>,
}

impl<'tcx> CheckWfFcxBuilder<'tcx> {
    fn with_fcx<F>(&mut self, f: F)
    where
        F: for<'b> FnOnce(&FnCtxt<'b, 'tcx>, TyCtxt<'tcx>) -> Vec<Ty<'tcx>>,
    {
        let id = self.id;
        let span = self.span;
        let param_env = self.param_env;
        self.inherited.enter(|inh| {
            let fcx = FnCtxt::new(&inh, param_env, id);
            if !inh.tcx.features().trivial_bounds {
                // As predicates are cached rather than obligations, this
                // needsto be called first so that they are checked with an
                // empty `param_env`.
                check_false_global_bounds(&fcx, span, id);
            }
            let wf_tys = f(&fcx, fcx.tcx);
            fcx.select_all_obligations_or_error();
            fcx.regionck_item(id, span, &wf_tys);
        });
    }
}

/// Checks that the field types (in a struct def'n) or argument types (in an enum def'n) are
/// well-formed, meaning that they do not require any constraints not declared in the struct
/// definition itself. For example, this definition would be illegal:
///
/// ```rust
/// struct Ref<'a, T> { x: &'a T }
/// ```
///
/// because the type did not declare that `T:'a`.
///
/// We do this check as a pre-pass before checking fn bodies because if these constraints are
/// not included it frequently leads to confusing errors in fn bodies. So it's better to check
/// the types first.
pub fn check_item_well_formed(tcx: TyCtxt<'_>, def_id: LocalDefId) {
    let hir_id = tcx.hir().local_def_id_to_hir_id(def_id);
    let item = tcx.hir().expect_item(hir_id);

    debug!(
        "check_item_well_formed(it.hir_id={:?}, it.name={})",
        item.hir_id,
        tcx.def_path_str(def_id.to_def_id())
    );

    match item.kind {
        // Right now we check that every default trait implementation
        // has an implementation of itself. Basically, a case like:
        //
        //     impl Trait for T {}
        //
        // has a requirement of `T: Trait` which was required for default
        // method implementations. Although this could be improved now that
        // there's a better infrastructure in place for this, it's being left
        // for a follow-up work.
        //
        // Since there's such a requirement, we need to check *just* positive
        // implementations, otherwise things like:
        //
        //     impl !Send for T {}
        //
        // won't be allowed unless there's an *explicit* implementation of `Send`
        // for `T`
        hir::ItemKind::Impl {
            defaultness,
            defaultness_span,
            polarity,
            ref of_trait,
            ref self_ty,
            ..
        } => {
            let is_auto = tcx
                .impl_trait_ref(tcx.hir().local_def_id(item.hir_id))
                .map_or(false, |trait_ref| tcx.trait_is_auto(trait_ref.def_id));
            if let (hir::Defaultness::Default { .. }, true) = (defaultness, is_auto) {
                let sp = of_trait.as_ref().map(|t| t.path.span).unwrap_or(item.span);
                let mut err =
                    tcx.sess.struct_span_err(sp, "impls of auto traits cannot be default");
                err.span_labels(defaultness_span, "default because of this");
                err.span_label(sp, "auto trait");
                err.emit();
            }
            // We match on both `ty::ImplPolarity` and `ast::ImplPolarity` just to get the `!` span.
            match (tcx.impl_polarity(def_id), polarity) {
                (ty::ImplPolarity::Positive, _) => {
                    check_impl(tcx, item, self_ty, of_trait);
                }
                (ty::ImplPolarity::Negative, ast::ImplPolarity::Negative(span)) => {
                    // FIXME(#27579): what amount of WF checking do we need for neg impls?
                    if let hir::Defaultness::Default { .. } = defaultness {
                        let mut spans = vec![span];
                        spans.extend(defaultness_span);
                        struct_span_err!(
                            tcx.sess,
                            spans,
                            E0750,
                            "negative impls cannot be default impls"
                        )
                        .emit();
                    }
                }
                (ty::ImplPolarity::Reservation, _) => {
                    // FIXME: what amount of WF checking do we need for reservation impls?
                }
                _ => unreachable!(),
            }
        }
        hir::ItemKind::Fn(ref sig, ..) => {
            check_item_fn(tcx, item.hir_id, item.ident, item.span, sig.decl);
        }
        hir::ItemKind::Static(ref ty, ..) => {
            check_item_type(tcx, item.hir_id, ty.span, false);
        }
        hir::ItemKind::Const(ref ty, ..) => {
            check_item_type(tcx, item.hir_id, ty.span, false);
        }
        hir::ItemKind::ForeignMod(ref module) => {
            for it in module.items.iter() {
                match it.kind {
                    hir::ForeignItemKind::Fn(ref decl, ..) => {
                        check_item_fn(tcx, it.hir_id, it.ident, it.span, decl)
                    }
                    hir::ForeignItemKind::Static(ref ty, ..) => {
                        check_item_type(tcx, it.hir_id, ty.span, true)
                    }
                    hir::ForeignItemKind::Type => (),
                }
            }
        }
        hir::ItemKind::Struct(ref struct_def, ref ast_generics) => {
            check_type_defn(tcx, item, false, |fcx| vec![fcx.non_enum_variant(struct_def)]);

            check_variances_for_type_defn(tcx, item, ast_generics);
        }
        hir::ItemKind::Union(ref struct_def, ref ast_generics) => {
            check_type_defn(tcx, item, true, |fcx| vec![fcx.non_enum_variant(struct_def)]);

            check_variances_for_type_defn(tcx, item, ast_generics);
        }
        hir::ItemKind::Enum(ref enum_def, ref ast_generics) => {
            check_type_defn(tcx, item, true, |fcx| fcx.enum_variants(enum_def));

            check_variances_for_type_defn(tcx, item, ast_generics);
        }
        hir::ItemKind::Trait(..) => {
            check_trait(tcx, item);
        }
        hir::ItemKind::TraitAlias(..) => {
            check_trait(tcx, item);
        }
        _ => {}
    }
}

pub fn check_trait_item(tcx: TyCtxt<'_>, def_id: LocalDefId) {
    let hir_id = tcx.hir().local_def_id_to_hir_id(def_id);
    let trait_item = tcx.hir().expect_trait_item(hir_id);

    let method_sig = match trait_item.kind {
        hir::TraitItemKind::Fn(ref sig, _) => Some(sig),
        _ => None,
    };
    check_object_unsafe_self_trait_by_name(tcx, &trait_item);
    check_associated_item(tcx, trait_item.hir_id, trait_item.span, method_sig);
}

fn could_be_self(trait_def_id: LocalDefId, ty: &hir::Ty<'_>) -> bool {
    match ty.kind {
        hir::TyKind::TraitObject([trait_ref], ..) => match trait_ref.trait_ref.path.segments {
            [s] => s.res.and_then(|r| r.opt_def_id()) == Some(trait_def_id.to_def_id()),
            _ => false,
        },
        _ => false,
    }
}

/// Detect when an object unsafe trait is referring to itself in one of its associated items.
/// When this is done, suggest using `Self` instead.
fn check_object_unsafe_self_trait_by_name(tcx: TyCtxt<'_>, item: &hir::TraitItem<'_>) {
    let (trait_name, trait_def_id) = match tcx.hir().get(tcx.hir().get_parent_item(item.hir_id)) {
        hir::Node::Item(item) => match item.kind {
            hir::ItemKind::Trait(..) => (item.ident, tcx.hir().local_def_id(item.hir_id)),
            _ => return,
        },
        _ => return,
    };
    let mut trait_should_be_self = vec![];
    match &item.kind {
        hir::TraitItemKind::Const(ty, _) | hir::TraitItemKind::Type(_, Some(ty))
            if could_be_self(trait_def_id, ty) =>
        {
            trait_should_be_self.push(ty.span)
        }
        hir::TraitItemKind::Fn(sig, _) => {
            for ty in sig.decl.inputs {
                if could_be_self(trait_def_id, ty) {
                    trait_should_be_self.push(ty.span);
                }
            }
            match sig.decl.output {
                hir::FnRetTy::Return(ty) if could_be_self(trait_def_id, ty) => {
                    trait_should_be_self.push(ty.span);
                }
                _ => {}
            }
        }
        _ => {}
    }
    if !trait_should_be_self.is_empty() {
        if tcx.object_safety_violations(trait_def_id).is_empty() {
            return;
        }
        let sugg = trait_should_be_self.iter().map(|span| (*span, "Self".to_string())).collect();
        tcx.sess
            .struct_span_err(
                trait_should_be_self,
                "associated item referring to unboxed trait object for its own trait",
            )
            .span_label(trait_name.span, "in this trait")
            .multipart_suggestion(
                "you might have meant to use `Self` to refer to the implementing type",
                sugg,
                Applicability::MachineApplicable,
            )
            .emit();
    }
}

pub fn check_impl_item(tcx: TyCtxt<'_>, def_id: LocalDefId) {
    let hir_id = tcx.hir().local_def_id_to_hir_id(def_id);
    let impl_item = tcx.hir().expect_impl_item(hir_id);

    let method_sig = match impl_item.kind {
        hir::ImplItemKind::Fn(ref sig, _) => Some(sig),
        _ => None,
    };

    check_associated_item(tcx, impl_item.hir_id, impl_item.span, method_sig);
}

fn check_param_wf(tcx: TyCtxt<'_>, param: &hir::GenericParam<'_>) {
    match param.kind {
        // We currently only check wf of const params here.
        hir::GenericParamKind::Lifetime { .. } | hir::GenericParamKind::Type { .. } => (),

        // Const parameters are well formed if their
        // type is structural match.
        hir::GenericParamKind::Const { ty: hir_ty } => {
            let ty = tcx.type_of(tcx.hir().local_def_id(param.hir_id));

            let err_ty_str;
            let mut is_ptr = true;
            let err = if tcx.features().min_const_generics {
                match ty.kind() {
                    ty::Bool | ty::Char | ty::Int(_) | ty::Uint(_) | ty::Error(_) => None,
                    ty::FnPtr(_) => Some("function pointers"),
                    ty::RawPtr(_) => Some("raw pointers"),
                    _ => {
                        is_ptr = false;
                        err_ty_str = format!("`{}`", ty);
                        Some(err_ty_str.as_str())
                    }
                }
            } else {
                match ty.peel_refs().kind() {
                    ty::FnPtr(_) => Some("function pointers"),
                    ty::RawPtr(_) => Some("raw pointers"),
                    _ => None,
                }
            };
            if let Some(unsupported_type) = err {
                if is_ptr {
                    tcx.sess.span_err(
                        hir_ty.span,
                        &format!(
                            "using {} as const generic parameters is forbidden",
                            unsupported_type
                        ),
                    )
                } else {
                    tcx.sess
                        .struct_span_err(
                            hir_ty.span,
                            &format!(
                                "{} is forbidden as the type of a const generic parameter",
                                unsupported_type
                            ),
                        )
                        .note("the only supported types are integers, `bool` and `char`")
                        .note("more complex types are supported with `#[feature(const_generics)]`")
                        .emit()
                }
            };

            if traits::search_for_structural_match_violation(param.hir_id, param.span, tcx, ty)
                .is_some()
            {
                // We use the same error code in both branches, because this is really the same
                // issue: we just special-case the message for type parameters to make it
                // clearer.
                if let ty::Param(_) = ty.peel_refs().kind() {
                    // Const parameters may not have type parameters as their types,
                    // because we cannot be sure that the type parameter derives `PartialEq`
                    // and `Eq` (just implementing them is not enough for `structural_match`).
                    struct_span_err!(
                        tcx.sess,
                        hir_ty.span,
                        E0741,
                        "`{}` is not guaranteed to `#[derive(PartialEq, Eq)]`, so may not be \
                            used as the type of a const parameter",
                        ty,
                    )
                    .span_label(
                        hir_ty.span,
                        format!("`{}` may not derive both `PartialEq` and `Eq`", ty),
                    )
                    .note(
                        "it is not currently possible to use a type parameter as the type of a \
                            const parameter",
                    )
                    .emit();
                } else {
                    struct_span_err!(
                        tcx.sess,
                        hir_ty.span,
                        E0741,
                        "`{}` must be annotated with `#[derive(PartialEq, Eq)]` to be used as \
                            the type of a const parameter",
                        ty,
                    )
                    .span_label(
                        hir_ty.span,
                        format!("`{}` doesn't derive both `PartialEq` and `Eq`", ty),
                    )
                    .emit();
                }
            }
        }
    }
}

fn check_associated_item(
    tcx: TyCtxt<'_>,
    item_id: hir::HirId,
    span: Span,
    sig_if_method: Option<&hir::FnSig<'_>>,
) {
    debug!("check_associated_item: {:?}", item_id);

    let code = ObligationCauseCode::MiscObligation;
    for_id(tcx, item_id, span).with_fcx(|fcx, tcx| {
        let item = fcx.tcx.associated_item(fcx.tcx.hir().local_def_id(item_id));

        let (mut implied_bounds, self_ty) = match item.container {
            ty::TraitContainer(_) => (vec![], fcx.tcx.types.self_param),
            ty::ImplContainer(def_id) => {
                (fcx.impl_implied_bounds(def_id, span), fcx.tcx.type_of(def_id))
            }
        };

        match item.kind {
            ty::AssocKind::Const => {
                let ty = fcx.tcx.type_of(item.def_id);
                let ty = fcx.normalize_associated_types_in(span, &ty);
                fcx.register_wf_obligation(ty.into(), span, code.clone());
            }
            ty::AssocKind::Fn => {
                let sig = fcx.tcx.fn_sig(item.def_id);
                let sig = fcx.normalize_associated_types_in(span, &sig);
                let hir_sig = sig_if_method.expect("bad signature for method");
                check_fn_or_method(
                    tcx,
                    fcx,
                    item.ident.span,
                    sig,
                    hir_sig.decl,
                    item.def_id,
                    &mut implied_bounds,
                );
                check_method_receiver(fcx, hir_sig, &item, self_ty);
            }
            ty::AssocKind::Type => {
                if let ty::AssocItemContainer::TraitContainer(_) = item.container {
                    check_associated_type_bounds(fcx, item, span)
                }
                if item.defaultness.has_value() {
                    let ty = fcx.tcx.type_of(item.def_id);
                    let ty = fcx.normalize_associated_types_in(span, &ty);
                    fcx.register_wf_obligation(ty.into(), span, code.clone());
                }
            }
        }

        implied_bounds
    })
}

fn for_item<'tcx>(tcx: TyCtxt<'tcx>, item: &hir::Item<'_>) -> CheckWfFcxBuilder<'tcx> {
    for_id(tcx, item.hir_id, item.span)
}

fn for_id(tcx: TyCtxt<'_>, id: hir::HirId, span: Span) -> CheckWfFcxBuilder<'_> {
    let def_id = tcx.hir().local_def_id(id);
    CheckWfFcxBuilder {
        inherited: Inherited::build(tcx, def_id),
        id,
        span,
        param_env: tcx.param_env(def_id),
    }
}

fn item_adt_kind(kind: &ItemKind<'_>) -> Option<AdtKind> {
    match kind {
        ItemKind::Struct(..) => Some(AdtKind::Struct),
        ItemKind::Union(..) => Some(AdtKind::Union),
        ItemKind::Enum(..) => Some(AdtKind::Enum),
        _ => None,
    }
}

/// In a type definition, we check that to ensure that the types of the fields are well-formed.
fn check_type_defn<'tcx, F>(
    tcx: TyCtxt<'tcx>,
    item: &hir::Item<'tcx>,
    all_sized: bool,
    mut lookup_fields: F,
) where
    F: for<'fcx> FnMut(&FnCtxt<'fcx, 'tcx>) -> Vec<AdtVariant<'tcx>>,
{
    for_item(tcx, item).with_fcx(|fcx, fcx_tcx| {
        let variants = lookup_fields(fcx);
        let def_id = fcx.tcx.hir().local_def_id(item.hir_id);
        let packed = fcx.tcx.adt_def(def_id).repr.packed();

        for variant in &variants {
            // For DST, or when drop needs to copy things around, all
            // intermediate types must be sized.
            let needs_drop_copy = || {
                packed && {
                    let ty = variant.fields.last().unwrap().ty;
                    let ty = fcx.tcx.erase_regions(&ty);
                    if ty.needs_infer() {
                        fcx_tcx
                            .sess
                            .delay_span_bug(item.span, &format!("inference variables in {:?}", ty));
                        // Just treat unresolved type expression as if it needs drop.
                        true
                    } else {
                        ty.needs_drop(fcx_tcx, fcx_tcx.param_env(def_id))
                    }
                }
            };
            let all_sized = all_sized || variant.fields.is_empty() || needs_drop_copy();
            let unsized_len = if all_sized { 0 } else { 1 };
            for (idx, field) in
                variant.fields[..variant.fields.len() - unsized_len].iter().enumerate()
            {
                let last = idx == variant.fields.len() - 1;
                fcx.register_bound(
                    field.ty,
                    fcx.tcx.require_lang_item(LangItem::Sized, None),
                    traits::ObligationCause::new(
                        field.span,
                        fcx.body_id,
                        traits::FieldSized {
                            adt_kind: match item_adt_kind(&item.kind) {
                                Some(i) => i,
                                None => bug!(),
                            },
                            span: field.span,
                            last,
                        },
                    ),
                );
            }

            // All field types must be well-formed.
            for field in &variant.fields {
                fcx.register_wf_obligation(
                    field.ty.into(),
                    field.span,
                    ObligationCauseCode::MiscObligation,
                )
            }

            // Explicit `enum` discriminant values must const-evaluate successfully.
            if let Some(discr_def_id) = variant.explicit_discr {
                let discr_substs =
                    InternalSubsts::identity_for_item(fcx.tcx, discr_def_id.to_def_id());

                let cause = traits::ObligationCause::new(
                    fcx.tcx.def_span(discr_def_id),
                    fcx.body_id,
                    traits::MiscObligation,
                );
                fcx.register_predicate(traits::Obligation::new(
                    cause,
                    fcx.param_env,
                    ty::PredicateAtom::ConstEvaluatable(
                        ty::WithOptConstParam::unknown(discr_def_id.to_def_id()),
                        discr_substs,
                    )
                    .to_predicate(fcx.tcx),
                ));
            }
        }

        check_where_clauses(tcx, fcx, item.span, def_id.to_def_id(), None);

        // No implied bounds in a struct definition.
        vec![]
    });
}

fn check_trait(tcx: TyCtxt<'_>, item: &hir::Item<'_>) {
    debug!("check_trait: {:?}", item.hir_id);

    let trait_def_id = tcx.hir().local_def_id(item.hir_id);

    let trait_def = tcx.trait_def(trait_def_id);
    if trait_def.is_marker
        || matches!(trait_def.specialization_kind, TraitSpecializationKind::Marker)
    {
        for associated_def_id in &*tcx.associated_item_def_ids(trait_def_id) {
            struct_span_err!(
                tcx.sess,
                tcx.def_span(*associated_def_id),
                E0714,
                "marker traits cannot have associated items",
            )
            .emit();
        }
    }

    for_item(tcx, item).with_fcx(|fcx, _| {
        check_where_clauses(tcx, fcx, item.span, trait_def_id.to_def_id(), None);

        vec![]
    });
}

/// Checks all associated type defaults of trait `trait_def_id`.
///
/// Assuming the defaults are used, check that all predicates (bounds on the
/// assoc type and where clauses on the trait) hold.
fn check_associated_type_bounds(fcx: &FnCtxt<'_, '_>, item: &ty::AssocItem, span: Span) {
    let tcx = fcx.tcx;

    let bounds = tcx.explicit_item_bounds(item.def_id);

    debug!("check_associated_type_bounds: bounds={:?}", bounds);
    let wf_obligations = bounds.iter().flat_map(|&(bound, bound_span)| {
        let normalized_bound = fcx.normalize_associated_types_in(span, &bound);
        traits::wf::predicate_obligations(
            fcx,
            fcx.param_env,
            fcx.body_id,
            normalized_bound,
            bound_span,
        )
    });

    for obligation in wf_obligations {
        debug!("next obligation cause: {:?}", obligation.cause);
        fcx.register_predicate(obligation);
    }
}

fn check_item_fn(
    tcx: TyCtxt<'_>,
    item_id: hir::HirId,
    ident: Ident,
    span: Span,
    decl: &hir::FnDecl<'_>,
) {
    for_id(tcx, item_id, span).with_fcx(|fcx, tcx| {
        let def_id = fcx.tcx.hir().local_def_id(item_id);
        let sig = fcx.tcx.fn_sig(def_id);
        let sig = fcx.normalize_associated_types_in(span, &sig);
        let mut implied_bounds = vec![];
        check_fn_or_method(
            tcx,
            fcx,
            ident.span,
            sig,
            decl,
            def_id.to_def_id(),
            &mut implied_bounds,
        );
        implied_bounds
    })
}

fn check_item_type(tcx: TyCtxt<'_>, item_id: hir::HirId, ty_span: Span, allow_foreign_ty: bool) {
    debug!("check_item_type: {:?}", item_id);

    for_id(tcx, item_id, ty_span).with_fcx(|fcx, tcx| {
        let ty = tcx.type_of(tcx.hir().local_def_id(item_id));
        let item_ty = fcx.normalize_associated_types_in(ty_span, &ty);

        let mut forbid_unsized = true;
        if allow_foreign_ty {
            let tail = fcx.tcx.struct_tail_erasing_lifetimes(item_ty, fcx.param_env);
            if let ty::Foreign(_) = tail.kind() {
                forbid_unsized = false;
            }
        }

        fcx.register_wf_obligation(item_ty.into(), ty_span, ObligationCauseCode::MiscObligation);
        if forbid_unsized {
            fcx.register_bound(
                item_ty,
                fcx.tcx.require_lang_item(LangItem::Sized, None),
                traits::ObligationCause::new(ty_span, fcx.body_id, traits::MiscObligation),
            );
        }

        // No implied bounds in a const, etc.
        vec![]
    });
}

fn check_impl<'tcx>(
    tcx: TyCtxt<'tcx>,
    item: &'tcx hir::Item<'tcx>,
    ast_self_ty: &hir::Ty<'_>,
    ast_trait_ref: &Option<hir::TraitRef<'_>>,
) {
    debug!("check_impl: {:?}", item);

    for_item(tcx, item).with_fcx(|fcx, tcx| {
        let item_def_id = fcx.tcx.hir().local_def_id(item.hir_id);

        match *ast_trait_ref {
            Some(ref ast_trait_ref) => {
                // `#[rustc_reservation_impl]` impls are not real impls and
                // therefore don't need to be WF (the trait's `Self: Trait` predicate
                // won't hold).
                let trait_ref = fcx.tcx.impl_trait_ref(item_def_id).unwrap();
                let trait_ref =
                    fcx.normalize_associated_types_in(ast_trait_ref.path.span, &trait_ref);
                let obligations = traits::wf::trait_obligations(
                    fcx,
                    fcx.param_env,
                    fcx.body_id,
                    &trait_ref,
                    ast_trait_ref.path.span,
                    Some(item),
                );
                for obligation in obligations {
                    fcx.register_predicate(obligation);
                }
            }
            None => {
                let self_ty = fcx.tcx.type_of(item_def_id);
                let self_ty = fcx.normalize_associated_types_in(item.span, &self_ty);
                fcx.register_wf_obligation(
                    self_ty.into(),
                    ast_self_ty.span,
                    ObligationCauseCode::MiscObligation,
                );
            }
        }

        check_where_clauses(tcx, fcx, item.span, item_def_id.to_def_id(), None);

        fcx.impl_implied_bounds(item_def_id.to_def_id(), item.span)
    });
}

/// Checks where-clauses and inline bounds that are declared on `def_id`.
fn check_where_clauses<'tcx, 'fcx>(
    tcx: TyCtxt<'tcx>,
    fcx: &FnCtxt<'fcx, 'tcx>,
    span: Span,
    def_id: DefId,
    return_ty: Option<(Ty<'tcx>, Span)>,
) {
    debug!("check_where_clauses(def_id={:?}, return_ty={:?})", def_id, return_ty);

    let predicates = fcx.tcx.predicates_of(def_id);
    let generics = tcx.generics_of(def_id);

    let is_our_default = |def: &ty::GenericParamDef| match def.kind {
        GenericParamDefKind::Type { has_default, .. } => {
            has_default && def.index >= generics.parent_count as u32
        }
        _ => unreachable!(),
    };

    // Check that concrete defaults are well-formed. See test `type-check-defaults.rs`.
    // For example, this forbids the declaration:
    //
    //     struct Foo<T = Vec<[u32]>> { .. }
    //
    // Here, the default `Vec<[u32]>` is not WF because `[u32]: Sized` does not hold.
    for param in &generics.params {
        if let GenericParamDefKind::Type { .. } = param.kind {
            if is_our_default(&param) {
                let ty = fcx.tcx.type_of(param.def_id);
                // Ignore dependent defaults -- that is, where the default of one type
                // parameter includes another (e.g., `<T, U = T>`). In those cases, we can't
                // be sure if it will error or not as user might always specify the other.
                if !ty.needs_subst() {
                    fcx.register_wf_obligation(
                        ty.into(),
                        fcx.tcx.def_span(param.def_id),
                        ObligationCauseCode::MiscObligation,
                    );
                }
            }
        }
    }

    // Check that trait predicates are WF when params are substituted by their defaults.
    // We don't want to overly constrain the predicates that may be written but we want to
    // catch cases where a default my never be applied such as `struct Foo<T: Copy = String>`.
    // Therefore we check if a predicate which contains a single type param
    // with a concrete default is WF with that default substituted.
    // For more examples see tests `defaults-well-formedness.rs` and `type-check-defaults.rs`.
    //
    // First we build the defaulted substitution.
    let substs = InternalSubsts::for_item(fcx.tcx, def_id, |param, _| {
        match param.kind {
            GenericParamDefKind::Lifetime => {
                // All regions are identity.
                fcx.tcx.mk_param_from_def(param)
            }

            GenericParamDefKind::Type { .. } => {
                // If the param has a default, ...
                if is_our_default(param) {
                    let default_ty = fcx.tcx.type_of(param.def_id);
                    // ... and it's not a dependent default, ...
                    if !default_ty.needs_subst() {
                        // ... then substitute it with the default.
                        return default_ty.into();
                    }
                }

                fcx.tcx.mk_param_from_def(param)
            }

            GenericParamDefKind::Const => {
                // FIXME(const_generics:defaults)
                fcx.tcx.mk_param_from_def(param)
            }
        }
    });

    // Now we build the substituted predicates.
    let default_obligations = predicates
        .predicates
        .iter()
        .flat_map(|&(pred, sp)| {
            #[derive(Default)]
            struct CountParams {
                params: FxHashSet<u32>,
            }
            impl<'tcx> ty::fold::TypeVisitor<'tcx> for CountParams {
                fn visit_ty(&mut self, t: Ty<'tcx>) -> ControlFlow<()> {
                    if let ty::Param(param) = t.kind() {
                        self.params.insert(param.index);
                    }
                    t.super_visit_with(self)
                }

                fn visit_region(&mut self, _: ty::Region<'tcx>) -> ControlFlow<()> {
                    ControlFlow::BREAK
                }

                fn visit_const(&mut self, c: &'tcx ty::Const<'tcx>) -> ControlFlow<()> {
                    if let ty::ConstKind::Param(param) = c.val {
                        self.params.insert(param.index);
                    }
                    c.super_visit_with(self)
                }
            }
            let mut param_count = CountParams::default();
            let has_region = pred.visit_with(&mut param_count).is_break();
            let substituted_pred = pred.subst(fcx.tcx, substs);
            // Don't check non-defaulted params, dependent defaults (including lifetimes)
            // or preds with multiple params.
            if substituted_pred.has_param_types_or_consts()
                || param_count.params.len() > 1
                || has_region
            {
                None
            } else if predicates.predicates.iter().any(|&(p, _)| p == substituted_pred) {
                // Avoid duplication of predicates that contain no parameters, for example.
                None
            } else {
                Some((substituted_pred, sp))
            }
        })
        .map(|(pred, sp)| {
            // Convert each of those into an obligation. So if you have
            // something like `struct Foo<T: Copy = String>`, we would
            // take that predicate `T: Copy`, substitute to `String: Copy`
            // (actually that happens in the previous `flat_map` call),
            // and then try to prove it (in this case, we'll fail).
            //
            // Note the subtle difference from how we handle `predicates`
            // below: there, we are not trying to prove those predicates
            // to be *true* but merely *well-formed*.
            let pred = fcx.normalize_associated_types_in(sp, &pred);
            let cause =
                traits::ObligationCause::new(sp, fcx.body_id, traits::ItemObligation(def_id));
            traits::Obligation::new(cause, fcx.param_env, pred)
        });

    let predicates = predicates.instantiate_identity(fcx.tcx);

    if let Some((mut return_ty, span)) = return_ty {
        if return_ty.has_infer_types_or_consts() {
            fcx.select_obligations_where_possible(false, |_| {});
            return_ty = fcx.resolve_vars_if_possible(&return_ty);
        }
        check_opaque_types(tcx, fcx, def_id.expect_local(), span, return_ty);
    }

    let predicates = fcx.normalize_associated_types_in(span, &predicates);

    debug!("check_where_clauses: predicates={:?}", predicates.predicates);
    assert_eq!(predicates.predicates.len(), predicates.spans.len());
    let wf_obligations =
        predicates.predicates.iter().zip(predicates.spans.iter()).flat_map(|(&p, &sp)| {
            traits::wf::predicate_obligations(fcx, fcx.param_env, fcx.body_id, p, sp)
        });

    for obligation in wf_obligations.chain(default_obligations) {
        debug!("next obligation cause: {:?}", obligation.cause);
        fcx.register_predicate(obligation);
    }
}

fn check_fn_or_method<'fcx, 'tcx>(
    tcx: TyCtxt<'tcx>,
    fcx: &FnCtxt<'fcx, 'tcx>,
    span: Span,
    sig: ty::PolyFnSig<'tcx>,
    hir_decl: &hir::FnDecl<'_>,
    def_id: DefId,
    implied_bounds: &mut Vec<Ty<'tcx>>,
) {
    let sig = fcx.normalize_associated_types_in(span, &sig);
    let sig = fcx.tcx.liberate_late_bound_regions(def_id, &sig);

    for (&input_ty, span) in sig.inputs().iter().zip(hir_decl.inputs.iter().map(|t| t.span)) {
        fcx.register_wf_obligation(input_ty.into(), span, ObligationCauseCode::MiscObligation);
    }
    implied_bounds.extend(sig.inputs());

    fcx.register_wf_obligation(
        sig.output().into(),
        hir_decl.output.span(),
        ObligationCauseCode::ReturnType,
    );

    // FIXME(#25759) return types should not be implied bounds
    implied_bounds.push(sig.output());

    check_where_clauses(tcx, fcx, span, def_id, Some((sig.output(), hir_decl.output.span())));
}

/// Checks "defining uses" of opaque `impl Trait` types to ensure that they meet the restrictions
/// laid for "higher-order pattern unification".
/// This ensures that inference is tractable.
/// In particular, definitions of opaque types can only use other generics as arguments,
/// and they cannot repeat an argument. Example:
///
/// ```rust
/// type Foo<A, B> = impl Bar<A, B>;
///
/// // Okay -- `Foo` is applied to two distinct, generic types.
/// fn a<T, U>() -> Foo<T, U> { .. }
///
/// // Not okay -- `Foo` is applied to `T` twice.
/// fn b<T>() -> Foo<T, T> { .. }
///
/// // Not okay -- `Foo` is applied to a non-generic type.
/// fn b<T>() -> Foo<T, u32> { .. }
/// ```
///
fn check_opaque_types<'fcx, 'tcx>(
    tcx: TyCtxt<'tcx>,
    fcx: &FnCtxt<'fcx, 'tcx>,
    fn_def_id: LocalDefId,
    span: Span,
    ty: Ty<'tcx>,
) {
    trace!("check_opaque_types(ty={:?})", ty);
    ty.fold_with(&mut ty::fold::BottomUpFolder {
        tcx: fcx.tcx,
        ty_op: |ty| {
            if let ty::Opaque(def_id, substs) = *ty.kind() {
                trace!("check_opaque_types: opaque_ty, {:?}, {:?}", def_id, substs);
                let generics = tcx.generics_of(def_id);

                let opaque_hir_id = if let Some(local_id) = def_id.as_local() {
                    tcx.hir().local_def_id_to_hir_id(local_id)
                } else {
                    // Opaque types from other crates won't have defining uses in this crate.
                    return ty;
                };
                if let hir::ItemKind::OpaqueTy(hir::OpaqueTy { impl_trait_fn: Some(_), .. }) =
                    tcx.hir().expect_item(opaque_hir_id).kind
                {
                    // No need to check return position impl trait (RPIT)
                    // because for type and const parameters they are correct
                    // by construction: we convert
                    //
                    // fn foo<P0..Pn>() -> impl Trait
                    //
                    // into
                    //
                    // type Foo<P0...Pn>
                    // fn foo<P0..Pn>() -> Foo<P0...Pn>.
                    //
                    // For lifetime parameters we convert
                    //
                    // fn foo<'l0..'ln>() -> impl Trait<'l0..'lm>
                    //
                    // into
                    //
                    // type foo::<'p0..'pn>::Foo<'q0..'qm>
                    // fn foo<l0..'ln>() -> foo::<'static..'static>::Foo<'l0..'lm>.
                    //
                    // which would error here on all of the `'static` args.
                    return ty;
                }
                if !may_define_opaque_type(tcx, fn_def_id, opaque_hir_id) {
                    return ty;
                }
                trace!("check_opaque_types: may define, generics={:#?}", generics);
                let mut seen_params: FxHashMap<_, Vec<_>> = FxHashMap::default();
                for (i, arg) in substs.iter().enumerate() {
                    let arg_is_param = match arg.unpack() {
                        GenericArgKind::Type(ty) => matches!(ty.kind(), ty::Param(_)),

                        GenericArgKind::Lifetime(region) => {
                            if let ty::ReStatic = region {
                                tcx.sess
                                    .struct_span_err(
                                        span,
                                        "non-defining opaque type use in defining scope",
                                    )
                                    .span_label(
                                        tcx.def_span(generics.param_at(i, tcx).def_id),
                                        "cannot use static lifetime; use a bound lifetime \
                                                 instead or remove the lifetime parameter from the \
                                                 opaque type",
                                    )
                                    .emit();
                                continue;
                            }

                            true
                        }

                        GenericArgKind::Const(ct) => matches!(ct.val, ty::ConstKind::Param(_)),
                    };

                    if arg_is_param {
                        seen_params.entry(arg).or_default().push(i);
                    } else {
                        // Prevent `fn foo() -> Foo<u32>` from being defining.
                        let opaque_param = generics.param_at(i, tcx);
                        tcx.sess
                            .struct_span_err(span, "non-defining opaque type use in defining scope")
                            .span_note(
                                tcx.def_span(opaque_param.def_id),
                                &format!(
                                    "used non-generic {} `{}` for generic parameter",
                                    opaque_param.kind.descr(),
                                    arg,
                                ),
                            )
                            .emit();
                    }
                } // for (arg, param)

                for (_, indices) in seen_params {
                    if indices.len() > 1 {
                        let descr = generics.param_at(indices[0], tcx).kind.descr();
                        let spans: Vec<_> = indices
                            .into_iter()
                            .map(|i| tcx.def_span(generics.param_at(i, tcx).def_id))
                            .collect();
                        tcx.sess
                            .struct_span_err(span, "non-defining opaque type use in defining scope")
                            .span_note(spans, &format!("{} used multiple times", descr))
                            .emit();
                    }
                }
            } // if let Opaque
            ty
        },
        lt_op: |lt| lt,
        ct_op: |ct| ct,
    });
}

const HELP_FOR_SELF_TYPE: &str = "consider changing to `self`, `&self`, `&mut self`, `self: Box<Self>`, \
     `self: Rc<Self>`, `self: Arc<Self>`, or `self: Pin<P>` (where P is one \
     of the previous types except `Self`)";

fn check_method_receiver<'fcx, 'tcx>(
    fcx: &FnCtxt<'fcx, 'tcx>,
    fn_sig: &hir::FnSig<'_>,
    method: &ty::AssocItem,
    self_ty: Ty<'tcx>,
) {
    // Check that the method has a valid receiver type, given the type `Self`.
    debug!("check_method_receiver({:?}, self_ty={:?})", method, self_ty);

    if !method.fn_has_self_parameter {
        return;
    }

    let span = fn_sig.decl.inputs[0].span;

    let sig = fcx.tcx.fn_sig(method.def_id);
    let sig = fcx.normalize_associated_types_in(span, &sig);
    let sig = fcx.tcx.liberate_late_bound_regions(method.def_id, &sig);

    debug!("check_method_receiver: sig={:?}", sig);

    let self_ty = fcx.normalize_associated_types_in(span, &self_ty);
    let self_ty = fcx.tcx.liberate_late_bound_regions(method.def_id, &ty::Binder::bind(self_ty));

    let receiver_ty = sig.inputs()[0];

    let receiver_ty = fcx.normalize_associated_types_in(span, &receiver_ty);
    let receiver_ty =
        fcx.tcx.liberate_late_bound_regions(method.def_id, &ty::Binder::bind(receiver_ty));

    if fcx.tcx.features().arbitrary_self_types {
        if !receiver_is_valid(fcx, span, receiver_ty, self_ty, true) {
            // Report error; `arbitrary_self_types` was enabled.
            e0307(fcx, span, receiver_ty);
        }
    } else {
        if !receiver_is_valid(fcx, span, receiver_ty, self_ty, false) {
            if receiver_is_valid(fcx, span, receiver_ty, self_ty, true) {
                // Report error; would have worked with `arbitrary_self_types`.
                feature_err(
                    &fcx.tcx.sess.parse_sess,
                    sym::arbitrary_self_types,
                    span,
                    &format!(
                        "`{}` cannot be used as the type of `self` without \
                         the `arbitrary_self_types` feature",
                        receiver_ty,
                    ),
                )
                .help(HELP_FOR_SELF_TYPE)
                .emit();
            } else {
                // Report error; would not have worked with `arbitrary_self_types`.
                e0307(fcx, span, receiver_ty);
            }
        }
    }
}

fn e0307(fcx: &FnCtxt<'fcx, 'tcx>, span: Span, receiver_ty: Ty<'_>) {
    struct_span_err!(
        fcx.tcx.sess.diagnostic(),
        span,
        E0307,
        "invalid `self` parameter type: {}",
        receiver_ty,
    )
    .note("type of `self` must be `Self` or a type that dereferences to it")
    .help(HELP_FOR_SELF_TYPE)
    .emit();
}

/// Returns whether `receiver_ty` would be considered a valid receiver type for `self_ty`. If
/// `arbitrary_self_types` is enabled, `receiver_ty` must transitively deref to `self_ty`, possibly
/// through a `*const/mut T` raw pointer. If the feature is not enabled, the requirements are more
/// strict: `receiver_ty` must implement `Receiver` and directly implement
/// `Deref<Target = self_ty>`.
///
/// N.B., there are cases this function returns `true` but causes an error to be emitted,
/// particularly when `receiver_ty` derefs to a type that is the same as `self_ty` but has the
/// wrong lifetime. Be careful of this if you are calling this function speculatively.
fn receiver_is_valid<'fcx, 'tcx>(
    fcx: &FnCtxt<'fcx, 'tcx>,
    span: Span,
    receiver_ty: Ty<'tcx>,
    self_ty: Ty<'tcx>,
    arbitrary_self_types_enabled: bool,
) -> bool {
    let cause = fcx.cause(span, traits::ObligationCauseCode::MethodReceiver);

    let can_eq_self = |ty| fcx.infcx.can_eq(fcx.param_env, self_ty, ty).is_ok();

    // `self: Self` is always valid.
    if can_eq_self(receiver_ty) {
        if let Some(mut err) = fcx.demand_eqtype_with_origin(&cause, self_ty, receiver_ty) {
            err.emit();
        }
        return true;
    }

    let mut autoderef = fcx.autoderef(span, receiver_ty);

    // The `arbitrary_self_types` feature allows raw pointer receivers like `self: *const Self`.
    if arbitrary_self_types_enabled {
        autoderef = autoderef.include_raw_pointers();
    }

    // The first type is `receiver_ty`, which we know its not equal to `self_ty`; skip it.
    autoderef.next();

    let receiver_trait_def_id = fcx.tcx.require_lang_item(LangItem::Receiver, None);

    // Keep dereferencing `receiver_ty` until we get to `self_ty`.
    loop {
        if let Some((potential_self_ty, _)) = autoderef.next() {
            debug!(
                "receiver_is_valid: potential self type `{:?}` to match `{:?}`",
                potential_self_ty, self_ty
            );

            if can_eq_self(potential_self_ty) {
                fcx.register_predicates(autoderef.into_obligations());

                if let Some(mut err) =
                    fcx.demand_eqtype_with_origin(&cause, self_ty, potential_self_ty)
                {
                    err.emit();
                }

                break;
            } else {
                // Without `feature(arbitrary_self_types)`, we require that each step in the
                // deref chain implement `receiver`
                if !arbitrary_self_types_enabled
                    && !receiver_is_implemented(
                        fcx,
                        receiver_trait_def_id,
                        cause.clone(),
                        potential_self_ty,
                    )
                {
                    return false;
                }
            }
        } else {
            debug!("receiver_is_valid: type `{:?}` does not deref to `{:?}`", receiver_ty, self_ty);
            // If he receiver already has errors reported due to it, consider it valid to avoid
            // unnecessary errors (#58712).
            return receiver_ty.references_error();
        }
    }

    // Without `feature(arbitrary_self_types)`, we require that `receiver_ty` implements `Receiver`.
    if !arbitrary_self_types_enabled
        && !receiver_is_implemented(fcx, receiver_trait_def_id, cause.clone(), receiver_ty)
    {
        return false;
    }

    true
}

fn receiver_is_implemented(
    fcx: &FnCtxt<'_, 'tcx>,
    receiver_trait_def_id: DefId,
    cause: ObligationCause<'tcx>,
    receiver_ty: Ty<'tcx>,
) -> bool {
    let trait_ref = ty::TraitRef {
        def_id: receiver_trait_def_id,
        substs: fcx.tcx.mk_substs_trait(receiver_ty, &[]),
    };

    let obligation = traits::Obligation::new(
        cause,
        fcx.param_env,
        trait_ref.without_const().to_predicate(fcx.tcx),
    );

    if fcx.predicate_must_hold_modulo_regions(&obligation) {
        true
    } else {
        debug!(
            "receiver_is_implemented: type `{:?}` does not implement `Receiver` trait",
            receiver_ty
        );
        false
    }
}

fn check_variances_for_type_defn<'tcx>(
    tcx: TyCtxt<'tcx>,
    item: &hir::Item<'tcx>,
    hir_generics: &hir::Generics<'_>,
) {
    let item_def_id = tcx.hir().local_def_id(item.hir_id);
    let ty = tcx.type_of(item_def_id);
    if tcx.has_error_field(ty) {
        return;
    }

    let ty_predicates = tcx.predicates_of(item_def_id);
    assert_eq!(ty_predicates.parent, None);
    let variances = tcx.variances_of(item_def_id);

    let mut constrained_parameters: FxHashSet<_> = variances
        .iter()
        .enumerate()
        .filter(|&(_, &variance)| variance != ty::Bivariant)
        .map(|(index, _)| Parameter(index as u32))
        .collect();

    identify_constrained_generic_params(tcx, ty_predicates, None, &mut constrained_parameters);

    for (index, _) in variances.iter().enumerate() {
        if constrained_parameters.contains(&Parameter(index as u32)) {
            continue;
        }

        let param = &hir_generics.params[index];

        match param.name {
            hir::ParamName::Error => {}
            _ => report_bivariance(tcx, param.span, param.name.ident().name),
        }
    }
}

fn report_bivariance(tcx: TyCtxt<'_>, span: Span, param_name: Symbol) {
    let mut err = error_392(tcx, span, param_name);

    let suggested_marker_id = tcx.lang_items().phantom_data();
    // Help is available only in presence of lang items.
    let msg = if let Some(def_id) = suggested_marker_id {
        format!(
            "consider removing `{}`, referring to it in a field, or using a marker such as `{}`",
            param_name,
            tcx.def_path_str(def_id),
        )
    } else {
        format!("consider removing `{}` or referring to it in a field", param_name)
    };
    err.help(&msg);
    err.emit();
}

/// Feature gates RFC 2056 -- trivial bounds, checking for global bounds that
/// aren't true.
fn check_false_global_bounds(fcx: &FnCtxt<'_, '_>, span: Span, id: hir::HirId) {
    let empty_env = ty::ParamEnv::empty();

    let def_id = fcx.tcx.hir().local_def_id(id);
    let predicates = fcx.tcx.predicates_of(def_id).predicates.iter().map(|(p, _)| *p);
    // Check elaborated bounds.
    let implied_obligations = traits::elaborate_predicates(fcx.tcx, predicates);

    for obligation in implied_obligations {
        let pred = obligation.predicate;
        // Match the existing behavior.
        if pred.is_global() && !pred.has_late_bound_regions() {
            let pred = fcx.normalize_associated_types_in(span, &pred);
            let obligation = traits::Obligation::new(
                traits::ObligationCause::new(span, id, traits::TrivialBound),
                empty_env,
                pred,
            );
            fcx.register_predicate(obligation);
        }
    }

    fcx.select_all_obligations_or_error();
}

#[derive(Clone, Copy)]
pub struct CheckTypeWellFormedVisitor<'tcx> {
    tcx: TyCtxt<'tcx>,
}

impl CheckTypeWellFormedVisitor<'tcx> {
    pub fn new(tcx: TyCtxt<'tcx>) -> CheckTypeWellFormedVisitor<'tcx> {
        CheckTypeWellFormedVisitor { tcx }
    }
}

impl ParItemLikeVisitor<'tcx> for CheckTypeWellFormedVisitor<'tcx> {
    fn visit_item(&self, i: &'tcx hir::Item<'tcx>) {
        Visitor::visit_item(&mut self.clone(), i);
    }

    fn visit_trait_item(&self, trait_item: &'tcx hir::TraitItem<'tcx>) {
        Visitor::visit_trait_item(&mut self.clone(), trait_item);
    }

    fn visit_impl_item(&self, impl_item: &'tcx hir::ImplItem<'tcx>) {
        Visitor::visit_impl_item(&mut self.clone(), impl_item);
    }
}

impl Visitor<'tcx> for CheckTypeWellFormedVisitor<'tcx> {
    type Map = hir_map::Map<'tcx>;

    fn nested_visit_map(&mut self) -> hir_visit::NestedVisitorMap<Self::Map> {
        hir_visit::NestedVisitorMap::OnlyBodies(self.tcx.hir())
    }

    fn visit_item(&mut self, i: &'tcx hir::Item<'tcx>) {
        debug!("visit_item: {:?}", i);
        let def_id = self.tcx.hir().local_def_id(i.hir_id);
        self.tcx.ensure().check_item_well_formed(def_id);
        hir_visit::walk_item(self, i);
    }

    fn visit_trait_item(&mut self, trait_item: &'tcx hir::TraitItem<'tcx>) {
        debug!("visit_trait_item: {:?}", trait_item);
        let def_id = self.tcx.hir().local_def_id(trait_item.hir_id);
        self.tcx.ensure().check_trait_item_well_formed(def_id);
        hir_visit::walk_trait_item(self, trait_item);
    }

    fn visit_impl_item(&mut self, impl_item: &'tcx hir::ImplItem<'tcx>) {
        debug!("visit_impl_item: {:?}", impl_item);
        let def_id = self.tcx.hir().local_def_id(impl_item.hir_id);
        self.tcx.ensure().check_impl_item_well_formed(def_id);
        hir_visit::walk_impl_item(self, impl_item);
    }

    fn visit_generic_param(&mut self, p: &'tcx hir::GenericParam<'tcx>) {
        check_param_wf(self.tcx, p);
        hir_visit::walk_generic_param(self, p);
    }
}

///////////////////////////////////////////////////////////////////////////
// ADT

// FIXME(eddyb) replace this with getting fields/discriminants through `ty::AdtDef`.
struct AdtVariant<'tcx> {
    /// Types of fields in the variant, that must be well-formed.
    fields: Vec<AdtField<'tcx>>,

    /// Explicit discriminant of this variant (e.g. `A = 123`),
    /// that must evaluate to a constant value.
    explicit_discr: Option<LocalDefId>,
}

struct AdtField<'tcx> {
    ty: Ty<'tcx>,
    span: Span,
}

impl<'a, 'tcx> FnCtxt<'a, 'tcx> {
    // FIXME(eddyb) replace this with getting fields through `ty::AdtDef`.
    fn non_enum_variant(&self, struct_def: &hir::VariantData<'_>) -> AdtVariant<'tcx> {
        let fields = struct_def
            .fields()
            .iter()
            .map(|field| {
                let field_ty = self.tcx.type_of(self.tcx.hir().local_def_id(field.hir_id));
                let field_ty = self.normalize_associated_types_in(field.ty.span, &field_ty);
                let field_ty = self.resolve_vars_if_possible(&field_ty);
                debug!("non_enum_variant: type of field {:?} is {:?}", field, field_ty);
                AdtField { ty: field_ty, span: field.ty.span }
            })
            .collect();
        AdtVariant { fields, explicit_discr: None }
    }

    fn enum_variants(&self, enum_def: &hir::EnumDef<'_>) -> Vec<AdtVariant<'tcx>> {
        enum_def
            .variants
            .iter()
            .map(|variant| AdtVariant {
                fields: self.non_enum_variant(&variant.data).fields,
                explicit_discr: variant
                    .disr_expr
                    .map(|explicit_discr| self.tcx.hir().local_def_id(explicit_discr.hir_id)),
            })
            .collect()
    }

    pub(super) fn impl_implied_bounds(&self, impl_def_id: DefId, span: Span) -> Vec<Ty<'tcx>> {
        match self.tcx.impl_trait_ref(impl_def_id) {
            Some(ref trait_ref) => {
                // Trait impl: take implied bounds from all types that
                // appear in the trait reference.
                let trait_ref = self.normalize_associated_types_in(span, trait_ref);
                trait_ref.substs.types().collect()
            }

            None => {
                // Inherent impl: take implied bounds from the `self` type.
                let self_ty = self.tcx.type_of(impl_def_id);
                let self_ty = self.normalize_associated_types_in(span, &self_ty);
                vec![self_ty]
            }
        }
    }
}

fn error_392(tcx: TyCtxt<'_>, span: Span, param_name: Symbol) -> DiagnosticBuilder<'_> {
    let mut err =
        struct_span_err!(tcx.sess, span, E0392, "parameter `{}` is never used", param_name);
    err.span_label(span, "unused parameter");
    err
}
