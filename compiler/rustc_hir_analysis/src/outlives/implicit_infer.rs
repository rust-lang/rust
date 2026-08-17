use rustc_data_structures::fx::FxIndexMap;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::DefId;
use rustc_middle::ty::{self, GenericArg, GenericArgKind, Ty, TyCtxt};
use rustc_span::Span;
use tracing::debug;

use super::explicit::ExplicitClausesMap;
use super::utils::*;

/// Infer outlives-clauses for the items in the local crate.
pub(super) fn infer_clauses(
    tcx: TyCtxt<'_>,
) -> FxIndexMap<DefId, ty::EarlyBinder<'_, RequiredClauses<'_>>> {
    debug!("infer_clauses");

    let mut explicit_map = ExplicitClausesMap::new();

    let mut global_inferred_outlives = FxIndexMap::default();

    // If new clauses were added then we need to re-calculate
    // all crates since there could be new implied clauses.
    for i in 0.. {
        let mut clauses_added = vec![];

        // Visit all the crates and infer clauses
        for id in tcx.hir_free_items() {
            let item_did = id.owner_id;

            debug!("InferVisitor::visit_item(item={:?})", item_did);

            let mut item_required_clauses = RequiredClauses::default();
            match tcx.def_kind(item_did) {
                DefKind::Union | DefKind::Enum | DefKind::Struct => {
                    let adt_def = tcx.adt_def(item_did.to_def_id());

                    // Iterate over all fields in item_did
                    for field_def in adt_def.all_fields() {
                        // Calculating the clause requirements necessary
                        // for item_did.
                        //
                        // For field of type &'a T (reference) or Adt
                        // (struct/enum/union) there will be outlive
                        // requirements for adt_def.
                        let field_ty =
                            tcx.type_of(field_def.did).instantiate_identity().skip_norm_wip();
                        let field_span = tcx.def_span(field_def.did);
                        insert_required_clauses_to_be_wf(
                            tcx,
                            field_ty,
                            field_span,
                            &global_inferred_outlives,
                            &mut item_required_clauses,
                            &mut explicit_map,
                        );
                    }
                }

                DefKind::TyAlias if tcx.type_alias_is_checked(item_did) => {
                    insert_required_clauses_to_be_wf(
                        tcx,
                        tcx.type_of(item_did).instantiate_identity().skip_norm_wip(),
                        tcx.def_span(item_did),
                        &global_inferred_outlives,
                        &mut item_required_clauses,
                        &mut explicit_map,
                    );
                }

                _ => {}
            };

            // If new clauses were added (`local_clause_map` has more
            // clauses than the `global_inferred_outlives`), the new clauses
            // might result in implied clauses for their parent types.
            // Therefore mark `clauses_added` as true and which will ensure
            // we walk the crates again and re-calculate clauses for all
            // items.
            let item_clauses_len: usize = global_inferred_outlives
                .get(&item_did.to_def_id())
                .map_or(0, |c| c.as_ref().skip_binder().len());
            if item_required_clauses.len() > item_clauses_len {
                clauses_added.push(item_did);
                global_inferred_outlives.insert(
                    item_did.to_def_id(),
                    ty::EarlyBinder::bind_iter(item_required_clauses),
                );
            }
        }

        if clauses_added.is_empty() {
            // We've reached a fixed point.
            break;
        } else if !tcx.recursion_limit().value_within_limit(i) {
            let msg = if let &[id] = &clauses_added[..] {
                format!("overflow computing implied lifetime bounds for `{}`", tcx.def_path_str(id),)
            } else {
                "overflow computing implied lifetime bounds".to_string()
            };
            tcx.dcx()
                .struct_span_fatal(
                    clauses_added.iter().map(|id| tcx.def_span(*id)).collect::<Vec<_>>(),
                    msg,
                )
                .emit();
        }
    }

    global_inferred_outlives
}

fn insert_required_clauses_to_be_wf<'tcx>(
    tcx: TyCtxt<'tcx>,
    ty: Ty<'tcx>,
    span: Span,
    global_inferred_outlives: &FxIndexMap<DefId, ty::EarlyBinder<'tcx, RequiredClauses<'tcx>>>,
    required_clauses: &mut RequiredClauses<'tcx>,
    explicit_map: &mut ExplicitClausesMap<'tcx>,
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
                // The type is `&'a T` which means that we will have
                // a clause requirement of `T: 'a` (`T` outlives `'a`).
                //
                // We also want to calculate potential clauses for the `T`.
                debug!("Ref");
                insert_outlives_clause(tcx, rty.into(), region, span, required_clauses);
            }

            ty::Adt(def, args) => {
                // For ADTs (structs/enums/unions), we check inferred and explicit clauses.
                debug!("Adt");
                check_inferred_clauses(
                    tcx,
                    def.did(),
                    args,
                    global_inferred_outlives,
                    required_clauses,
                );
                check_explicit_clauses(
                    tcx,
                    def.did(),
                    args,
                    required_clauses,
                    explicit_map,
                    IgnoreClausesReferencingSelf::No,
                );
            }

            ty::Alias(_, ty::AliasTy { kind: ty::Free { def_id }, args, .. }) => {
                // This corresponds to a type like `Type<'a, T>`.
                // We check inferred and explicit clauses.
                debug!("Free");
                check_inferred_clauses(
                    tcx,
                    def_id,
                    args,
                    global_inferred_outlives,
                    required_clauses,
                );
                check_explicit_clauses(
                    tcx,
                    def_id,
                    args,
                    required_clauses,
                    explicit_map,
                    IgnoreClausesReferencingSelf::No,
                );
            }

            ty::Dynamic(obj, ..) => {
                // This corresponds to `dyn Trait<..>`. In this case, we should
                // use the explicit clauses as well.
                debug!("Dynamic");
                if let Some(trait_ref) = obj.principal() {
                    let args = trait_ref
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
                        trait_ref.def_id(),
                        args,
                        required_clauses,
                        explicit_map,
                        IgnoreClausesReferencingSelf::Yes,
                    );
                }
            }

            ty::Alias(_, ty::AliasTy { kind: ty::Projection { def_id }, args, .. }) => {
                // This corresponds to a type like `<() as Trait<'a, T>>::Type`.
                // We only use the explicit clauses of the trait but
                // not the ones of the associated type itself.
                debug!("Projection");
                check_explicit_clauses(
                    tcx,
                    tcx.parent(def_id),
                    args,
                    required_clauses,
                    explicit_map,
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
///     field: Inner<T>,
/// }
///
/// struct Inner<U> where U: 'static, U: Outer {
///     // ...
/// }
/// ```
/// Here, we should fetch the explicit clauses, which
/// will give us `U: 'static` and `U: Outer`. The latter we
/// can ignore, but we will want to process `U: 'static`,
/// applying the instantiation as above.
#[tracing::instrument(level = "debug", skip(tcx))]
fn check_explicit_clauses<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: DefId,
    args: &[GenericArg<'tcx>],
    required_clauses: &mut RequiredClauses<'tcx>,
    explicit_map: &mut ExplicitClausesMap<'tcx>,
    ignore_clauses_refing_self: IgnoreClausesReferencingSelf,
) {
    let explicit_clauses = explicit_map.explicit_clauses_of(tcx, def_id);

    for (&clause @ ty::OutlivesClause(arg, _), &span) in explicit_clauses.as_ref().skip_binder() {
        debug!(?clause);

        if let IgnoreClausesReferencingSelf::Yes = ignore_clauses_refing_self
            && arg.walk().any(|arg| arg == tcx.types.self_param.into())
        {
            debug!("ignoring clause since it references `Self`");
            continue;
        }

        let clause @ ty::OutlivesClause(arg, region) =
            explicit_clauses.rebind(clause).instantiate(tcx, args).skip_norm_wip();
        debug!(?clause);

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
/// Here, when processing the type of field `outer`, we would request the
/// set of implicit clauses computed for `Inner` thus far. This will
/// initially come back empty, but in next round we will get `U: 'b`.
/// We then apply the instantiation `['b => 'a, U => T]` and thus get the
/// requirement that `T: 'a` holds for `Outer`.
fn check_inferred_clauses<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: DefId,
    args: ty::GenericArgsRef<'tcx>,
    global_inferred_outlives: &FxIndexMap<DefId, ty::EarlyBinder<'tcx, RequiredClauses<'tcx>>>,
    required_clauses: &mut RequiredClauses<'tcx>,
) {
    // Load the current set of inferred and explicit clauses from `global_inferred_outlives`
    // and filter the ones that are `TypeOutlives`.

    let Some(clauses) = global_inferred_outlives.get(&def_id) else {
        return;
    };

    for (&clause, &span) in clauses.as_ref().skip_binder() {
        // `clause` is `U: 'b` in the example above.
        // So apply the instantiation to get `T: 'a`.
        let ty::OutlivesClause(arg, region) =
            clauses.rebind(clause).instantiate(tcx, args).skip_norm_wip();
        insert_outlives_clause(tcx, arg, region, span, required_clauses);
    }
}
