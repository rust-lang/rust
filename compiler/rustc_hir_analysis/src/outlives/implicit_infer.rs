use rustc_data_structures::fx::FxIndexMap;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::DefId;
use rustc_middle::ty::{self, GenericArg, GenericArgKind, Ty, TyCtxt};
use rustc_span::Span;
use tracing::debug;

use super::explicit::ExplicitPredicatesMap;
use super::utils::*;

/// Infer outlives-predicates for the items in the local crate.
pub(super) fn infer_predicates(
    tcx: TyCtxt<'_>,
) -> FxIndexMap<DefId, ty::EarlyBinder<'_, RequiredPredicates<'_>>> {
    debug!("infer_predicates");

    let mut explicit_map = ExplicitPredicatesMap::new();

    let mut global_inferred_outlives = FxIndexMap::default();

    // If new predicates were added then we need to re-calculate
    // all crates since there could be new implied predicates.
    for i in 0.. {
        let mut predicates_added = vec![];

        // Visit all the crates and infer predicates
        for id in tcx.hir_free_items() {
            let item_did = id.owner_id;

            debug!("InferVisitor::visit_item(item={:?})", item_did);

            let mut item_required_predicates = RequiredPredicates::default();
            match tcx.def_kind(item_did) {
                DefKind::Union | DefKind::Enum | DefKind::Struct => {
                    let adt_def = tcx.adt_def(item_did.to_def_id());

                    // Iterate over all fields in item_did
                    for field_def in adt_def.all_fields() {
                        // Calculating the predicate requirements necessary
                        // for item_did.
                        //
                        // For field of type &'a T (reference) or Adt
                        // (struct/enum/union) there will be outlive
                        // requirements for adt_def.
                        let field_ty = tcx.type_of(field_def.did).instantiate_identity();
                        let field_span = tcx.def_span(field_def.did);
                        insert_required_predicates_to_be_wf(
                            tcx,
                            field_ty,
                            field_span,
                            &global_inferred_outlives,
                            &mut item_required_predicates,
                            &mut explicit_map,
                        );
                    }
                }

                DefKind::TyAlias if tcx.type_alias_is_lazy(item_did) => {
                    insert_required_predicates_to_be_wf(
                        tcx,
                        tcx.type_of(item_did).instantiate_identity(),
                        tcx.def_span(item_did),
                        &global_inferred_outlives,
                        &mut item_required_predicates,
                        &mut explicit_map,
                    );
                }

                _ => {}
            };

            // If new predicates were added (`local_predicate_map` has more
            // predicates than the `global_inferred_outlives`), the new predicates
            // might result in implied predicates for their parent types.
            // Therefore mark `predicates_added` as true and which will ensure
            // we walk the crates again and re-calculate predicates for all
            // items.
            let item_predicates_len: usize = global_inferred_outlives
                .get(&item_did.to_def_id())
                .map_or(0, |p| p.as_ref().skip_binder().len());
            if item_required_predicates.len() > item_predicates_len {
                predicates_added.push(item_did);
                global_inferred_outlives
                    .insert(item_did.to_def_id(), ty::EarlyBinder::bind(item_required_predicates));
            }
        }

        if predicates_added.is_empty() {
            // We've reached a fixed point.
            break;
        } else if !tcx.recursion_limit().value_within_limit(i) {
            let msg = if let &[id] = &predicates_added[..] {
                format!("overflow computing implied lifetime bounds for `{}`", tcx.def_path_str(id),)
            } else {
                "overflow computing implied lifetime bounds".to_string()
            };
            tcx.dcx()
                .struct_span_fatal(
                    predicates_added.iter().map(|id| tcx.def_span(*id)).collect::<Vec<_>>(),
                    msg,
                )
                .emit();
        }
    }

    global_inferred_outlives
}

fn insert_required_predicates_to_be_wf<'tcx>(
    tcx: TyCtxt<'tcx>,
    ty: Ty<'tcx>,
    span: Span,
    global_inferred_outlives: &FxIndexMap<DefId, ty::EarlyBinder<'tcx, RequiredPredicates<'tcx>>>,
    required_predicates: &mut RequiredPredicates<'tcx>,
    explicit_map: &mut ExplicitPredicatesMap<'tcx>,
) {
    for arg in ty.walk() {
        let leaf_ty = match arg.kind() {
            GenericArgKind::Type(ty) => ty,

            // No predicates from lifetimes or constants, except potentially
            // constants' types, but `walk` will get to them as well.
            GenericArgKind::Lifetime(_) | GenericArgKind::Const(_) => continue,
        };

        match *leaf_ty.kind() {
            ty::Ref(region, rty, _) => {
                // The type is `&'a T` which means that we will have
                // a predicate requirement of `T: 'a` (`T` outlives `'a`).
                //
                // We also want to calculate potential predicates for the `T`.
                debug!("Ref");
                insert_outlives_predicate(tcx, rty.into(), region, span, required_predicates);
            }

            ty::Adt(def, args) => {
                // For ADTs (structs/enums/unions), we check inferred and explicit predicates.
                debug!("Adt");
                check_inferred_predicates(
                    tcx,
                    def.did(),
                    args,
                    global_inferred_outlives,
                    required_predicates,
                );
                check_explicit_predicates(
                    tcx,
                    def.did(),
                    args,
                    required_predicates,
                    explicit_map,
                    IgnorePredicatesReferencingSelf::No,
                );
            }

            ty::Alias(ty::AliasTy { kind: ty::Free { def_id }, args, .. }) => {
                // This corresponds to a type like `Type<'a, T>`.
                // We check inferred and explicit predicates.
                debug!("Free");
                check_inferred_predicates(
                    tcx,
                    def_id,
                    args,
                    global_inferred_outlives,
                    required_predicates,
                );
                check_explicit_predicates(
                    tcx,
                    def_id,
                    args,
                    required_predicates,
                    explicit_map,
                    IgnorePredicatesReferencingSelf::No,
                );
            }

            ty::Dynamic(obj, ..) => {
                // This corresponds to `dyn Trait<..>`. In this case, we should
                // use the explicit predicates as well.
                debug!("Dynamic");
                if let Some(trait_ref) = obj.principal() {
                    let args = trait_ref
                        .with_self_ty(tcx, tcx.types.trait_object_dummy_self)
                        .skip_binder()
                        .args;
                    // We skip predicates that reference the `Self` type parameter since we don't
                    // want to leak the dummy Self to the predicates map.
                    //
                    // While filtering out bounds like `Self: 'a` as in `trait Trait<'a, T>: 'a {}`
                    // doesn't matter since they can't affect the lifetime / type parameters anyway,
                    // for bounds like `Self::AssocTy: 'b` which we of course currently also ignore
                    // (see also #54467) it might conceivably be better to extract the binding
                    // `AssocTy = U` from the trait object type (which must exist) and thus infer
                    // an outlives requirement that `U: 'b`.
                    check_explicit_predicates(
                        tcx,
                        trait_ref.def_id(),
                        args,
                        required_predicates,
                        explicit_map,
                        IgnorePredicatesReferencingSelf::Yes,
                    );
                }
            }

            ty::Alias(ty::AliasTy { kind: ty::Projection { def_id }, args, .. }) => {
                // This corresponds to a type like `<() as Trait<'a, T>>::Type`.
                // We only use the explicit predicates of the trait but
                // not the ones of the associated type itself.
                debug!("Projection");
                check_explicit_predicates(
                    tcx,
                    tcx.parent(def_id),
                    args,
                    required_predicates,
                    explicit_map,
                    IgnorePredicatesReferencingSelf::No,
                );
            }

            // FIXME(inherent_associated_types): Use the explicit predicates from the parent impl.
            ty::Alias(ty::AliasTy { kind: ty::Inherent { .. }, .. }) => {}

            _ => {}
        }
    }
}

/// Check the explicit predicates declared on the type.
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
/// Here, we should fetch the explicit predicates, which
/// will give us `U: 'static` and `U: Outer`. The latter we
/// can ignore, but we will want to process `U: 'static`,
/// applying the instantiation as above.
#[tracing::instrument(level = "debug", skip(tcx))]
fn check_explicit_predicates<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: DefId,
    args: &[GenericArg<'tcx>],
    required_predicates: &mut RequiredPredicates<'tcx>,
    explicit_map: &mut ExplicitPredicatesMap<'tcx>,
    ignore_preds_refing_self: IgnorePredicatesReferencingSelf,
) {
    let explicit_predicates = explicit_map.explicit_predicates_of(tcx, def_id);

    for (&predicate @ ty::OutlivesPredicate(arg, _), &span) in
        explicit_predicates.as_ref().skip_binder()
    {
        debug!(?predicate);

        if let IgnorePredicatesReferencingSelf::Yes = ignore_preds_refing_self
            && arg.walk().any(|arg| arg == tcx.types.self_param.into())
        {
            debug!("ignoring predicate since it references `Self`");
            continue;
        }

        let predicate @ ty::OutlivesPredicate(arg, region) =
            explicit_predicates.rebind(predicate).instantiate(tcx, args);
        debug!(?predicate);

        insert_outlives_predicate(tcx, arg, region, span, required_predicates);
    }
}

#[derive(Debug)]
enum IgnorePredicatesReferencingSelf {
    Yes,
    No,
}

/// Check the inferred predicates of the type.
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
/// set of implicit predicates computed for `Inner` thus far. This will
/// initially come back empty, but in next round we will get `U: 'b`.
/// We then apply the instantiation `['b => 'a, U => T]` and thus get the
/// requirement that `T: 'a` holds for `Outer`.
fn check_inferred_predicates<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: DefId,
    args: ty::GenericArgsRef<'tcx>,
    global_inferred_outlives: &FxIndexMap<DefId, ty::EarlyBinder<'tcx, RequiredPredicates<'tcx>>>,
    required_predicates: &mut RequiredPredicates<'tcx>,
) {
    // Load the current set of inferred and explicit predicates from `global_inferred_outlives`
    // and filter the ones that are `TypeOutlives`.

    let Some(predicates) = global_inferred_outlives.get(&def_id) else {
        return;
    };

    for (&predicate, &span) in predicates.as_ref().skip_binder() {
        // `predicate` is `U: 'b` in the example above.
        // So apply the instantiation to get `T: 'a`.
        let ty::OutlivesPredicate(arg, region) =
            predicates.rebind(predicate).instantiate(tcx, args);
        insert_outlives_predicate(tcx, arg, region, span, required_predicates);
    }
}
