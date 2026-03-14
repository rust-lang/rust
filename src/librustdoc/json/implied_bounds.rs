use rustc_data_structures::fx::FxHashSet;
use rustc_data_structures::thin_vec::ThinVec;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::DefId;
use rustc_hir::{LangItem, OpaqueTyOrigin};
use rustc_infer::infer::region_constraints::GenericKind;
use rustc_infer::traits::util::transitive_bounds_that_define_assoc_item;
use rustc_middle::ty::{self, AliasTy, ParamTy, Ty, TyCtxt, TypingMode};
use rustc_span::symbol::Ident;
use rustc_trait_selection::infer::TyCtxtInferExt;
use rustc_trait_selection::infer::outlives::env::OutlivesEnvironment;
use rustc_trait_selection::regions::OutlivesEnvironmentBuildExt;
use rustdoc_json_types::GenericBound;

use crate::clean;
use crate::config::OutputFormat;
use crate::core::DocContext;
use crate::formats::cache::Cache;
use crate::json::JsonRenderer;
use crate::json::conversions::IntoJson;

pub(crate) fn implied_bounds_for_ty<'tcx>(
    target_ty: Ty<'tcx>,
    clauses: &[ty::Clause<'tcx>],
    explicit_bounds: &[GenericBound],
    owner_def_id: DefId,
    renderer: &JsonRenderer<'tcx>,
) -> Vec<GenericBound> {
    let mut seen: FxHashSet<_> = explicit_bounds.iter().cloned().collect();
    let mut implied_bounds = Vec::new();
    let mut clean_cx = implied_bounds_doc_context(renderer, owner_def_id);
    for clause in clauses {
        if !clause_targets_ty(*clause, target_ty) {
            continue;
        }

        if let Some(bound) = clause_to_generic_bound(*clause, clauses, &mut clean_cx, renderer) {
            if seen.insert(bound.clone()) {
                implied_bounds.push(bound);
            }
        }
    }

    implied_bounds
}

pub(crate) fn implied_bounds_for_type_param<'tcx>(
    owner_def_id: DefId,
    param_def_id: DefId,
    explicit_bounds: &[GenericBound],
    renderer: &JsonRenderer<'tcx>,
) -> Vec<GenericBound> {
    let target_param = param_ty_for_param(renderer.tcx, owner_def_id, param_def_id);
    let target_ty = target_param.to_ty(renderer.tcx);

    let clauses = renderer.tcx.param_env(owner_def_id).caller_bounds();
    let mut implied_bounds = implied_bounds_for_ty(
        target_ty,
        clauses.as_slice(),
        explicit_bounds,
        owner_def_id,
        renderer,
    );

    let extra_bounds = implied_outlives_bounds_for_param(renderer.tcx, owner_def_id, target_param);
    if !extra_bounds.is_empty() {
        let mut seen: FxHashSet<_> = explicit_bounds.iter().cloned().collect();
        seen.extend(implied_bounds.iter().cloned());
        for bound in extra_bounds {
            if seen.insert(bound.clone()) {
                implied_bounds.push(bound);
            }
        }
    }

    implied_bounds
}

pub(crate) fn implied_bounds_for_assoc_type<'tcx>(
    assoc_def_id: DefId,
    explicit_bounds: &[GenericBound],
    renderer: &JsonRenderer<'tcx>,
) -> Vec<GenericBound> {
    let assoc_item = renderer.tcx.associated_item(assoc_def_id);
    if !matches!(assoc_item.container, ty::AssocContainer::Trait) {
        return Vec::new();
    }

    let args = ty::GenericArgs::identity_for_item(renderer.tcx, assoc_def_id);
    let target_ty = Ty::new_alias(
        renderer.tcx,
        ty::Projection,
        AliasTy::new_from_args(renderer.tcx, assoc_def_id, args),
    );
    let clauses = renderer.tcx.item_bounds(assoc_def_id).instantiate(renderer.tcx, args);

    implied_bounds_for_ty(target_ty, &clauses, explicit_bounds, assoc_def_id, renderer)
}

pub(crate) fn implied_bounds_for_impl_trait<'tcx>(
    origin: &clean::ImplTraitOrigin,
    explicit_bounds: &[GenericBound],
    renderer: &JsonRenderer<'tcx>,
) -> Vec<GenericBound> {
    match origin {
        clean::ImplTraitOrigin::Param { def_id } => {
            let owner_def_id = renderer.tcx.parent(*def_id);
            implied_bounds_for_type_param(owner_def_id, *def_id, explicit_bounds, renderer)
        }
        clean::ImplTraitOrigin::Opaque { def_id } => {
            let args = ty::GenericArgs::identity_for_item(renderer.tcx, *def_id);
            let target_ty = Ty::new_alias(
                renderer.tcx,
                ty::Opaque,
                AliasTy::new_from_args(renderer.tcx, *def_id, args),
            );
            let clauses = renderer.tcx.item_bounds(*def_id).instantiate(renderer.tcx, args);

            let mut implied_bounds =
                implied_bounds_for_ty(target_ty, &clauses, explicit_bounds, *def_id, renderer);

            let extra_bounds = implied_outlives_bounds_for_opaque(renderer.tcx, *def_id);
            if !extra_bounds.is_empty() {
                let mut seen: FxHashSet<_> = explicit_bounds.iter().cloned().collect();
                seen.extend(implied_bounds.iter().cloned());
                for bound in extra_bounds {
                    if seen.insert(bound.clone()) {
                        implied_bounds.push(bound);
                    }
                }
            }

            implied_bounds
        }
    }
}

/// Build a minimal [`DocContext`] for implied-bounds rendering.
///
/// This is intentionally a narrow, JSON-only shim that lets us reuse existing `clean::*`
/// conversion helpers when turning `ty::Clause` data into `rustdoc_json_types`:
/// - The implied-bounds logic starts from [`ty::Clause`] (param-env predicates) rather than
///   from HIR, so we don't have a preexisting clean representation to convert.
/// - The relevant clean helpers ([`crate::clean::clean_trait_ref_with_constraints`],
///   [`crate::clean::projection_to_path_segment`], [`crate::clean::clean_middle_term`],
///   [`crate::clean::clean_bound_vars`]) require a [`DocContext`] to access `tcx`, `param_env`, and
///   path/generic normalization logic. We don't want to duplicate them here.
///
/// This context is read-only and intentionally minimal: it only carries the fields needed by
/// the clean helpers above (e.g., `tcx`, `param_env`, `auto_traits`, and a fresh [`Cache`] to
/// satisfy path lookups). It does not run passes, does not mutate global caches, and does not
/// depend on the rest of the cleaning pipeline.
///
/// If this ever shows up as a hot path or becomes too heavyweight, the alternatives are:
/// - reimplement the clean logic directly in JSON and accept some duplication;
/// - move implied-bounds computation into `clean` itself, making it shared with rustdoc HTML, or
/// - refactor JSON rendering to get access to the main [`DocContext`].
fn implied_bounds_doc_context<'tcx>(
    renderer: &JsonRenderer<'tcx>,
    owner_def_id: DefId,
) -> DocContext<'tcx> {
    let auto_traits = renderer
        .tcx
        .visible_traits()
        .filter(|&trait_def_id| renderer.tcx.trait_is_auto(trait_def_id))
        .collect();
    DocContext {
        tcx: renderer.tcx,
        param_env: renderer.tcx.param_env(owner_def_id),
        external_traits: Default::default(),
        active_extern_traits: Default::default(),
        args: Default::default(),
        current_type_aliases: Default::default(),
        impl_trait_bounds: Default::default(),
        generated_synthetics: Default::default(),
        auto_traits,
        cache: Cache::new(renderer.cache.document_private, renderer.cache.document_hidden),
        inlined: Default::default(),
        output_format: OutputFormat::Json,
        show_coverage: false,
    }
}

fn is_allowed_lang_item_trait(tcx: TyCtxt<'_>, def_id: DefId) -> bool {
    match tcx.as_lang_item(def_id) {
        None => true,

        // These are all the language item traits that are stable in Rust today.
        Some(
            LangItem::Sized
            | LangItem::Clone
            | LangItem::Copy
            | LangItem::Sync
            | LangItem::Drop
            | LangItem::Add
            | LangItem::Sub
            | LangItem::Mul
            | LangItem::Div
            | LangItem::Rem
            | LangItem::Neg
            | LangItem::Not
            | LangItem::BitXor
            | LangItem::BitAnd
            | LangItem::BitOr
            | LangItem::Shl
            | LangItem::Shr
            | LangItem::AddAssign
            | LangItem::SubAssign
            | LangItem::MulAssign
            | LangItem::DivAssign
            | LangItem::RemAssign
            | LangItem::BitXorAssign
            | LangItem::BitAndAssign
            | LangItem::BitOrAssign
            | LangItem::ShlAssign
            | LangItem::ShrAssign
            | LangItem::Index
            | LangItem::IndexMut
            | LangItem::Deref
            | LangItem::DerefMut
            | LangItem::Fn
            | LangItem::FnMut
            | LangItem::FnOnce
            | LangItem::AsyncFn
            | LangItem::AsyncFnMut
            | LangItem::AsyncFnOnce
            | LangItem::Iterator
            | LangItem::FusedIterator
            | LangItem::Future
            | LangItem::Unpin
            | LangItem::PartialEq
            | LangItem::PartialOrd,
        ) => true,

        Some(_) => false,
    }
}

fn clause_targets_ty<'tcx>(clause: ty::Clause<'tcx>, target: Ty<'tcx>) -> bool {
    if let Some(trait_clause) = clause.as_trait_clause() {
        trait_clause.self_ty().skip_binder() == target
    } else if let Some(type_outlives) = clause.as_type_outlives_clause() {
        type_outlives.skip_binder().0 == target
    } else {
        false
    }
}

fn clause_to_generic_bound<'tcx>(
    clause: ty::Clause<'tcx>,
    all_clauses: &[ty::Clause<'tcx>],
    clean_cx: &mut DocContext<'tcx>,
    renderer: &JsonRenderer<'tcx>,
) -> Option<GenericBound> {
    if let Some(trait_clause) = clause.as_trait_clause() {
        let def_id = trait_clause.def_id();
        if !is_allowed_lang_item_trait(renderer.tcx, def_id) {
            None
        } else {
            let poly_trait_ref = trait_clause.map_bound(|pred| pred.trait_ref);
            let constraints =
                assoc_item_constraints_for_trait_ref(all_clauses, poly_trait_ref, clean_cx);
            let bound =
                clean::clean_poly_trait_ref_with_constraints(clean_cx, poly_trait_ref, constraints);
            Some(bound.into_json(renderer))
        }
    } else if let Some(type_outlives) = clause.as_type_outlives_clause() {
        let ty::OutlivesPredicate(_, region) = type_outlives.skip_binder();
        region.get_name(renderer.tcx).map(|name| GenericBound::Outlives(name.to_string()))
    } else {
        None
    }
}

fn assoc_item_constraints_for_trait_ref<'tcx>(
    clauses: &[ty::Clause<'tcx>],
    poly_trait_ref: ty::Binder<'tcx, ty::TraitRef<'tcx>>,
    clean_cx: &mut DocContext<'tcx>,
) -> ThinVec<clean::AssocItemConstraint> {
    let mut constraints = ThinVec::new();
    for clause in clauses {
        if let Some(proj_clause) = clause.as_projection_clause() {
            let proj_pred = proj_clause.skip_binder();
            let proj_trait_ref = proj_pred.projection_term.trait_ref(clean_cx.tcx);
            let Some(assoc_name) = clean_cx.tcx.opt_item_ident(proj_pred.projection_term.def_id)
            else {
                continue;
            };
            if !projection_applies_to_trait_ref(
                clean_cx.tcx,
                proj_trait_ref,
                poly_trait_ref,
                assoc_name,
            ) {
                continue;
            }

            constraints.push(clean::AssocItemConstraint {
                assoc: clean::projection_to_path_segment(
                    proj_clause.map_bound(|pred| pred.projection_term),
                    clean_cx,
                ),
                kind: clean::AssocItemConstraintKind::Equality {
                    term: clean::clean_middle_term(
                        proj_clause.map_bound(|pred| pred.term),
                        clean_cx,
                    ),
                },
            });
            continue;
        }

        let Some(trait_clause) = clause.as_trait_clause() else { continue };
        let self_ty = trait_clause.skip_binder().trait_ref.self_ty();
        let ty::Alias(ty::Projection, alias_ty) = *self_ty.kind() else { continue };
        let proj_trait_ref = alias_ty.trait_ref(clean_cx.tcx);
        let Some(assoc_name) = clean_cx.tcx.opt_item_ident(alias_ty.def_id) else { continue };
        if !projection_applies_to_trait_ref(
            clean_cx.tcx,
            proj_trait_ref,
            poly_trait_ref,
            assoc_name,
        ) {
            continue;
        }

        let bound_trait_def_id = trait_clause.def_id();
        if !is_allowed_lang_item_trait(clean_cx.tcx, bound_trait_def_id) {
            continue;
        }

        let bound_trait_ref = trait_clause.map_bound(|pred| pred.trait_ref);
        let bound =
            clean::clean_poly_trait_ref_with_constraints(clean_cx, bound_trait_ref, ThinVec::new());

        let assoc =
            clean::projection_to_path_segment(trait_clause.rebind(alias_ty.into()), clean_cx);
        let mut merged = false;
        for existing in constraints.iter_mut() {
            if existing.assoc == assoc {
                if let clean::AssocItemConstraintKind::Bound { bounds } = &mut existing.kind {
                    if !bounds.contains(&bound) {
                        bounds.push(bound.clone());
                    }
                    merged = true;
                    break;
                }
            }
        }

        if !merged {
            constraints.push(clean::AssocItemConstraint {
                assoc,
                kind: clean::AssocItemConstraintKind::Bound { bounds: vec![bound] },
            });
        }
    }
    constraints
}

/// Returns `true` if a projection predicate on `proj_trait_ref` should be attached to `trait_ref`.
///
/// Most projections target the exact trait that defines the associated item, which is the
/// `proj_trait_ref == trait_ref` fast path. When a bound uses a trait whose associated items are
/// defined on one of its supertraits (e.g. `Sub: Deref` and `Sub<Target = ()>`), we attach those
/// projections to the supertrait so the implied bound carries the full constraint.
fn projection_applies_to_trait_ref<'tcx>(
    tcx: TyCtxt<'tcx>,
    proj_trait_ref: ty::TraitRef<'tcx>,
    trait_ref: ty::Binder<'tcx, ty::TraitRef<'tcx>>,
    assoc_name: Ident,
) -> bool {
    if proj_trait_ref == trait_ref.skip_binder() {
        return true;
    }

    transitive_bounds_that_define_assoc_item(tcx, std::iter::once(trait_ref), assoc_name)
        .any(|supertrait_ref| supertrait_ref.skip_binder() == proj_trait_ref)
}

fn param_ty_for_param<'tcx>(
    tcx: TyCtxt<'tcx>,
    owner_def_id: DefId,
    param_def_id: DefId,
) -> ParamTy {
    let generics = tcx.generics_of(owner_def_id);
    let index = generics
        .param_def_id_to_index(tcx, param_def_id)
        .unwrap_or_else(|| tcx.dcx().bug("param_def_id should belong to owner_def_id"));
    let param_def = generics.param_at(index as usize, tcx);
    match param_def.kind {
        ty::GenericParamDefKind::Type { .. } => ParamTy::new(index, tcx.item_name(param_def_id)),
        _ => tcx.dcx().bug("param_def_id should refer to a type parameter"),
    }
}

fn implied_outlives_bounds_for_param<'tcx>(
    tcx: TyCtxt<'tcx>,
    owner_def_id: DefId,
    param_ty: ParamTy,
) -> Vec<GenericBound> {
    let Some(local_def_id) = owner_def_id.as_local() else {
        return Vec::new();
    };

    let assumed_wf = tcx.assumed_wf_types(local_def_id);
    if assumed_wf.is_empty() {
        return Vec::new();
    }

    let param_env = tcx.param_env(owner_def_id);
    let infcx = tcx.infer_ctxt().build(TypingMode::non_body_analysis());
    let env = OutlivesEnvironment::new(
        &infcx,
        local_def_id,
        param_env,
        assumed_wf.iter().map(|(ty, _)| *ty),
    );

    env.region_bound_pairs()
        .iter()
        .filter_map(|predicate| match predicate {
            ty::OutlivesPredicate(GenericKind::Param(param), region)
                if param.index == param_ty.index =>
            {
                region.get_name(tcx).map(|name| GenericBound::Outlives(name.to_string()))
            }
            _ => None,
        })
        .collect()
}

/// If the opaque is a TAIT / ATPIT, return any additional outlives bounds.
///
/// For example, `type Foo<'a, T> = &'a impl PartialEq<T>;`
/// has an implied `+ 'a` bound that would be returned here.
///
/// If this function is called with a different kind of opaque, it returns no bounds.
///
/// We also aren't able to return any bounds for cross-crate TAITs due to missing metadata.
fn implied_outlives_bounds_for_opaque<'tcx>(
    tcx: TyCtxt<'tcx>,
    opaque_def_id: DefId,
) -> Vec<GenericBound> {
    if tcx.def_kind(opaque_def_id) != DefKind::OpaqueTy {
        return Vec::new();
    }

    let Some(local_def_id) = opaque_def_id.as_local() else {
        // Cross-crate TAITs don't carry the parent WF info in metadata,
        // so we can't infer outlives bounds here.
        // FIXME: Get metadata on extern opaques, then make this precise.
        return Vec::new();
    };

    let OpaqueTyOrigin::TyAlias { parent, .. } = tcx.opaque_ty_origin(local_def_id.to_def_id())
    else {
        return Vec::new();
    };

    let Some(local_parent) = parent.as_local() else {
        // Cross-crate TAITs don't carry the parent WF info in metadata,
        // so we can't infer outlives bounds here.
        return Vec::new();
    };

    let param_env = tcx.param_env(parent);
    let parent_ty = tcx.type_of(parent).instantiate_identity();
    let infcx = tcx.infer_ctxt().build(TypingMode::non_body_analysis());
    let env = OutlivesEnvironment::new(&infcx, local_parent, param_env, [parent_ty]);

    let target_args = ty::GenericArgs::identity_for_item(tcx, parent).extend_to(
        tcx,
        opaque_def_id,
        |param, _| tcx.map_opaque_lifetime_to_parent_lifetime(param.def_id.expect_local()).into(),
    );
    let target_alias = AliasTy::new_from_args(tcx, opaque_def_id, target_args);
    env.region_bound_pairs()
        .iter()
        .filter_map(|predicate| match predicate {
            ty::OutlivesPredicate(GenericKind::Alias(alias), region) if *alias == target_alias => {
                region.get_name(tcx).map(|name| GenericBound::Outlives(name.to_string()))
            }
            _ => None,
        })
        .collect()
}
