use rustc_hir::def::DefKind;
use rustc_hir::def_id::DefId;
use rustc_middle::ty::{self, GenericArg, GenericArgKind, Ty, TyCtxt};
use rustc_span::Span;
use tracing::debug;

use super::explicit::GlobalExplicitOutlivesClauses;
use super::utils::*;

/// Infer outlives-clauses for the items in the local crate.
#[tracing::instrument(level = "debug", skip_all)]
pub(super) fn infer_outlives_clauses(tcx: TyCtxt<'_>) -> GlobalOutlivesClauses<'_> {
    let mut global_explicit_clauses = GlobalExplicitOutlivesClauses::default();
    let mut global_inferred_clauses = GlobalOutlivesClauses::default();

    for i in 0.. {
        // Whenever new clauses get added we need to re-calculate the set of
        // clauses for all items since there could be new implied clauses.
        let mut items_with_added_clauses = vec![];

        // Visit all free items in the local crate and infer outlives-clauses.
        // We simply don't consider other kinds of items at the time of writing.
        for id in tcx.hir_free_items() {
            let item_did = id.owner_id;
            debug!(?item_did);

            let mut item_required_clauses = OutlivesClauses::default();
            match tcx.def_kind(item_did) {
                DefKind::Union | DefKind::Enum | DefKind::Struct => {
                    let adt_def = tcx.adt_def(item_did.to_def_id());

                    for field_def in adt_def.all_fields() {
                        insert_required_outlives_clauses_to_be_wf(
                            tcx,
                            tcx.type_of(field_def.did).instantiate_identity().skip_norm_wip(),
                            tcx.def_span(field_def.did),
                            &mut item_required_clauses,
                            &global_inferred_clauses,
                            &mut global_explicit_clauses,
                        );
                    }
                }
                DefKind::TyAlias if tcx.type_alias_is_checked(item_did) => {
                    insert_required_outlives_clauses_to_be_wf(
                        tcx,
                        tcx.type_of(item_did).instantiate_identity().skip_norm_wip(),
                        tcx.def_span(item_did),
                        &mut item_required_clauses,
                        &global_inferred_clauses,
                        &mut global_explicit_clauses,
                    );
                }
                _ => {}
            };

            // If there are more required clauses than inferred ones, it means new clauses were
            // added which might result in implied clauses for their overarching types.
            // Register the item to ensure we visit all items again and re-calculate clauses.
            let item_inferred_clauses: usize = global_inferred_clauses
                .get(&item_did.to_def_id())
                .map_or(0, |c| c.as_ref().skip_binder().len());
            if item_required_clauses.len() > item_inferred_clauses {
                items_with_added_clauses.push(item_did);
                global_inferred_clauses.insert(
                    item_did.to_def_id(),
                    ty::EarlyBinder::bind_iter(item_required_clauses),
                );
            }
        }

        if items_with_added_clauses.is_empty() {
            // We've reached a fixed point.
            break;
        }

        if !tcx.recursion_limit().value_within_limit(i) {
            let msg = if let &[id] = &items_with_added_clauses[..] {
                format!("overflow computing implied lifetime bounds for `{}`", tcx.def_path_str(id),)
            } else {
                "overflow computing implied lifetime bounds".to_string()
            };
            tcx.dcx().span_fatal(
                items_with_added_clauses.iter().map(|id| tcx.def_span(*id)).collect::<Vec<_>>(),
                msg,
            );
        }
    }

    global_inferred_clauses
}

fn insert_required_outlives_clauses_to_be_wf<'tcx>(
    tcx: TyCtxt<'tcx>,
    ty: Ty<'tcx>,
    span: Span,
    required_clauses: &mut OutlivesClauses<'tcx>,
    global_inferred_clauses: &GlobalOutlivesClauses<'tcx>,
    global_explicit_clauses: &mut GlobalExplicitOutlivesClauses<'tcx>,
) {
    for arg in ty.walk() {
        let leaf_ty = match arg.kind() {
            GenericArgKind::Type(ty) => ty,
            // No clauses from lifetimes or constants, except potentially
            // constants' types, but `walk` will get to them as well.
            GenericArgKind::Lifetime(_) | GenericArgKind::Const(_) => continue,
        };

        match *leaf_ty.kind() {
            ty::Ref(region, rty, _) => {
                // The type `&'a T` inherently has the requirement `T: 'a`.
                insert_outlives_clause(tcx, rty.into(), region, span, required_clauses);
            }

            ty::Adt(def, args) => {
                check_inferred_clauses(
                    tcx,
                    def.did(),
                    args,
                    global_inferred_clauses,
                    required_clauses,
                );
                check_explicit_clauses(
                    tcx,
                    def.did(),
                    args,
                    required_clauses,
                    global_explicit_clauses,
                    IgnoreClausesReferencingSelf::No,
                );
            }

            ty::Alias(_, ty::AliasTy { kind: ty::Free { def_id }, args, .. }) => {
                check_inferred_clauses(
                    tcx,
                    def_id,
                    args,
                    global_inferred_clauses,
                    required_clauses,
                );
                check_explicit_clauses(
                    tcx,
                    def_id,
                    args,
                    required_clauses,
                    global_explicit_clauses,
                    IgnoreClausesReferencingSelf::No,
                );
            }

            ty::Dynamic(obj, ..) if let Some(principal_trait_ref) = obj.principal() => {
                let args = principal_trait_ref
                    .with_self_ty(tcx, tcx.types.trait_object_dummy_self)
                    .skip_binder()
                    .args;
                // We skip clauses that reference the `Self` type parameter since we don't
                // want to leak the dummy Self to the clauses map.
                //
                // While filtering out bounds like `Self: 'a` as in `trait Trait<'a, T>: 'a {}`
                // doesn't matter since they can't affect the lifetime / type parameters anyway,
                // for bounds like `Self::AssocTy: 'b` which we of course currently also ignore
                // (see also #54467) it might conceivably be better to extract the binding
                // `AssocTy = U` from the trait object type (which must exist) and thus infer
                // an outlives requirement that `U: 'b`.
                check_explicit_clauses(
                    tcx,
                    principal_trait_ref.def_id(),
                    args,
                    required_clauses,
                    global_explicit_clauses,
                    IgnoreClausesReferencingSelf::Yes,
                );
            }

            ty::Alias(_, ty::AliasTy { kind: ty::Projection { def_id }, args, .. }) => {
                // We only use the explicit clauses of the trait but not the ones of the associated
                // type itself. FIXME(#141692): Ideally we would consider them, too, though.
                //
                // E.g., for `<() as Trait<'b, U>>::Type` we will infer `U: 'b` (modulo component
                // splitting) given `trait Trait<'a, T: 'a> { type Type; }` if `'b` is early-bound.
                check_explicit_clauses(
                    tcx,
                    tcx.parent(def_id),
                    args,
                    required_clauses,
                    global_explicit_clauses,
                    IgnoreClausesReferencingSelf::No,
                );
            }

            // FIXME(inherent_associated_types): Use the explicit clauses from the parent impl.
            ty::Alias(_, ty::AliasTy { kind: ty::Inherent { .. }, .. }) => {}

            _ => {}
        }
    }
}

/// Check the explicit clauses declared on the type.
///
/// ### Example
///
/// ```ignore (illustrative)
/// struct Outer<'a, T> {
///     outer: Inner<'a, T>,
/// }
///
/// struct Inner<'b, U: Trait + 'b> {
///     inner: U,
///     _marker: &'b (),
/// }
/// ```
///
/// Here, when processing the type of field `outer`, namely ADT `Inner<'a, T>`, we fetch the set of
/// explicit clauses which gives us `U: Trait` and `U: 'b`. We ignore the former and instantiate
/// the latter using `['b => 'a, U => T]` to obtain the requirement that `T: 'a` holds which we then
/// insert into the set of `required_clauses` (which belongs to `Outer`).
#[tracing::instrument(level = "debug", skip(tcx))]
fn check_explicit_clauses<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: DefId,
    args: &[GenericArg<'tcx>],
    required_clauses: &mut OutlivesClauses<'tcx>,
    global_explicit_clauses: &mut GlobalExplicitOutlivesClauses<'tcx>,
    ignore_clauses_refing_self: IgnoreClausesReferencingSelf,
) {
    let explicit_clauses = global_explicit_clauses.explicit_outlives_clauses_of(tcx, def_id);

    for (&clause @ ty::OutlivesClause(arg, _), &span) in explicit_clauses.as_ref().skip_binder() {
        if let IgnoreClausesReferencingSelf::Yes = ignore_clauses_refing_self
            && arg.walk().any(|arg| arg == tcx.types.self_param.into())
        {
            debug!("ignoring clause `{clause}` since it references `Self`");
            continue;
        }

        let ty::OutlivesClause(arg, region) =
            explicit_clauses.rebind(clause).instantiate(tcx, args).skip_norm_wip();
        insert_outlives_clause(tcx, arg, region, span, required_clauses);
    }
}

#[derive(Debug)]
enum IgnoreClausesReferencingSelf {
    Yes,
    No,
}

/// Check the inferred clauses of the type.
///
/// ### Example
///
/// ```ignore (illustrative)
/// struct Outer<'a, T> {
///     outer: Inner<'a, T>,
/// }
///
/// struct Inner<'b, U> {
///     inner: &'b U,
/// }
/// ```
///
/// Here, when processing the type of field `outer`, namely ADT `Inner<'a, T>`, we request the set
/// of implicit clauses computed for `Inner` thus far. Initially it comes back empty but in the next
/// round we get `U: 'b`. We then apply the instantiation `['b => 'a, U => T]` and thus get the
/// requirement that `T: 'a` holds which we then insert into the set of `required_clauses` (which
/// belongs to `Outer`).
fn check_inferred_clauses<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: DefId,
    args: ty::GenericArgsRef<'tcx>,
    global_inferred_clauses: &GlobalOutlivesClauses<'tcx>,
    required_clauses: &mut OutlivesClauses<'tcx>,
) {
    let Some(clauses) = global_inferred_clauses.get(&def_id) else { return };

    for (&clause, &span) in clauses.as_ref().skip_binder() {
        let ty::OutlivesClause(arg, region) =
            clauses.rebind(clause).instantiate(tcx, args).skip_norm_wip();
        insert_outlives_clause(tcx, arg, region, span, required_clauses);
    }
}
