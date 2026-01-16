use rustc_data_structures::fx::FxHashSet;
use rustc_errors::{Applicability, ErrorGuaranteed, MultiSpan, pluralize, struct_span_code_err};
use rustc_hir as hir;
use rustc_hir::def::{DefKind, Res};
use rustc_index::bit_set::DenseBitSet;
use rustc_middle::bug;
use rustc_middle::dep_graph::dep_kinds;
use rustc_middle::query::plumbing::CycleError;
use rustc_middle::ty::{self, Representability, Ty, TyCtxt};
use rustc_middle::util::Providers;
use rustc_span::Span;
use rustc_span::def_id::LocalDefId;

pub(crate) fn provide(providers: &mut Providers) {
    providers.representability = representability;
    providers.fallback_queries.representability =
        |tcx, _key, cycle, _guar| representability_from_cycle(tcx, cycle);
    providers.representability_adt_ty = representability_adt_ty;
    providers.fallback_queries.representability_adt_ty =
        |tcx, _key, cycle, _guar| representability_from_cycle(tcx, cycle);
    providers.params_in_repr = params_in_repr;
}

macro_rules! rtry {
    ($e:expr) => {
        match $e {
            e @ Representability::Infinite(_) => return e,
            Representability::Representable => {}
        }
    };
}

fn representability(tcx: TyCtxt<'_>, def_id: LocalDefId) -> Representability {
    match tcx.def_kind(def_id) {
        DefKind::Struct | DefKind::Union | DefKind::Enum => {
            for variant in tcx.adt_def(def_id).variants() {
                for field in variant.fields.iter() {
                    rtry!(tcx.representability(field.did.expect_local()));
                }
            }
            Representability::Representable
        }
        DefKind::Field => representability_ty(tcx, tcx.type_of(def_id).instantiate_identity()),
        def_kind => bug!("unexpected {def_kind:?}"),
    }
}

fn representability_ty<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> Representability {
    match *ty.kind() {
        ty::Adt(..) => tcx.representability_adt_ty(ty),
        // FIXME(#11924) allow zero-length arrays?
        ty::Array(ty, _) => representability_ty(tcx, ty),
        ty::Tuple(tys) => {
            for ty in tys {
                rtry!(representability_ty(tcx, ty));
            }
            Representability::Representable
        }
        _ => Representability::Representable,
    }
}

/*
The reason for this being a separate query is very subtle:
Consider this infinitely sized struct: `struct Foo(Box<Foo>, Bar<Foo>)`:
When calling representability(Foo), a query cycle will occur:
  representability(Foo)
    -> representability_adt_ty(Bar<Foo>)
    -> representability(Foo)
For the diagnostic output (in `Value::from_cycle_error`), we want to detect that
the `Foo` in the *second* field of the struct is culpable. This requires
traversing the HIR of the struct and calling `params_in_repr(Bar)`. But we can't
call params_in_repr for a given type unless it is known to be representable.
params_in_repr will cycle/panic on infinitely sized types. Looking at the query
cycle above, we know that `Bar` is representable because
representability_adt_ty(Bar<..>) is in the cycle and representability(Bar) is
*not* in the cycle.
*/
fn representability_adt_ty<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> Representability {
    let ty::Adt(adt, args) = ty.kind() else { bug!("expected adt") };
    if let Some(def_id) = adt.did().as_local() {
        rtry!(tcx.representability(def_id));
    }
    // At this point, we know that the item of the ADT type is representable;
    // but the type parameters may cause a cycle with an upstream type
    let params_in_repr = tcx.params_in_repr(adt.did());
    for (i, arg) in args.iter().enumerate() {
        if let ty::GenericArgKind::Type(ty) = arg.kind() {
            if params_in_repr.contains(i as u32) {
                rtry!(representability_ty(tcx, ty));
            }
        }
    }
    Representability::Representable
}

fn params_in_repr(tcx: TyCtxt<'_>, def_id: LocalDefId) -> DenseBitSet<u32> {
    let adt_def = tcx.adt_def(def_id);
    let generics = tcx.generics_of(def_id);
    let mut params_in_repr = DenseBitSet::new_empty(generics.own_params.len());
    for variant in adt_def.variants() {
        for field in variant.fields.iter() {
            params_in_repr_ty(
                tcx,
                tcx.type_of(field.did).instantiate_identity(),
                &mut params_in_repr,
            );
        }
    }
    params_in_repr
}

fn params_in_repr_ty<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>, params_in_repr: &mut DenseBitSet<u32>) {
    match *ty.kind() {
        ty::Adt(adt, args) => {
            let inner_params_in_repr = tcx.params_in_repr(adt.did());
            for (i, arg) in args.iter().enumerate() {
                if let ty::GenericArgKind::Type(ty) = arg.kind() {
                    if inner_params_in_repr.contains(i as u32) {
                        params_in_repr_ty(tcx, ty, params_in_repr);
                    }
                }
            }
        }
        ty::Array(ty, _) => params_in_repr_ty(tcx, ty, params_in_repr),
        ty::Tuple(tys) => tys.iter().for_each(|ty| params_in_repr_ty(tcx, ty, params_in_repr)),
        ty::Param(param) => {
            params_in_repr.insert(param.index);
        }
        _ => {}
    }
}

fn representability_from_cycle<'tcx>(
    tcx: TyCtxt<'tcx>,
    cycle_error: &CycleError,
) -> Representability {
    let mut item_and_field_ids = Vec::new();
    let mut representable_ids = FxHashSet::default();
    for info in &cycle_error.cycle {
        if info.query.dep_kind == dep_kinds::representability
            && let Some(field_id) = info.query.def_id
            && let Some(field_id) = field_id.as_local()
            && let Some(DefKind::Field) = info.query.info.def_kind
        {
            let parent_id = tcx.parent(field_id.to_def_id());
            let item_id = match tcx.def_kind(parent_id) {
                DefKind::Variant => tcx.parent(parent_id),
                _ => parent_id,
            };
            item_and_field_ids.push((item_id.expect_local(), field_id));
        }
    }
    for info in &cycle_error.cycle {
        if info.query.dep_kind == dep_kinds::representability_adt_ty
            && let Some(def_id) = info.query.def_id_for_ty_in_cycle
            && let Some(def_id) = def_id.as_local()
            && !item_and_field_ids.iter().any(|&(id, _)| id == def_id)
        {
            representable_ids.insert(def_id);
        }
    }
    let guar = recursive_type_error(tcx, item_and_field_ids, &representable_ids);
    Representability::Infinite(guar)
}

// item_and_field_ids should form a cycle where each field contains the
// type in the next element in the list
fn recursive_type_error(
    tcx: TyCtxt<'_>,
    mut item_and_field_ids: Vec<(LocalDefId, LocalDefId)>,
    representable_ids: &FxHashSet<LocalDefId>,
) -> ErrorGuaranteed {
    use std::fmt::Write as _;

    const ITEM_LIMIT: usize = 5;

    // Rotate the cycle so that the item with the lowest span is first
    let start_index = item_and_field_ids
        .iter()
        .enumerate()
        .min_by_key(|&(_, &(id, _))| tcx.def_span(id))
        .unwrap()
        .0;
    item_and_field_ids.rotate_left(start_index);

    let cycle_len = item_and_field_ids.len();
    let show_cycle_len = cycle_len.min(ITEM_LIMIT);

    let mut err_span = MultiSpan::from_spans(
        item_and_field_ids[..show_cycle_len]
            .iter()
            .map(|(id, _)| tcx.def_span(id.to_def_id()))
            .collect(),
    );
    let mut suggestion = Vec::with_capacity(show_cycle_len * 2);
    for i in 0..show_cycle_len {
        let (_, field_id) = item_and_field_ids[i];
        let (next_item_id, _) = item_and_field_ids[(i + 1) % cycle_len];
        // Find the span(s) that contain the next item in the cycle
        let hir::Node::Field(field) = tcx.hir_node_by_def_id(field_id) else {
            bug!("expected field")
        };
        let mut found = Vec::new();
        find_item_ty_spans(tcx, field.ty, next_item_id, &mut found, representable_ids);

        // Couldn't find the type. Maybe it's behind a type alias?
        // In any case, we'll just suggest boxing the whole field.
        if found.is_empty() {
            found.push(field.ty.span);
        }

        for span in found {
            err_span.push_span_label(span, "recursive without indirection");
            // FIXME(compiler-errors): This suggestion might be erroneous if Box is shadowed
            suggestion.push((span.shrink_to_lo(), "Box<".to_string()));
            suggestion.push((span.shrink_to_hi(), ">".to_string()));
        }
    }
    let items_list = {
        let mut s = String::new();
        for (i, &(item_id, _)) in item_and_field_ids.iter().enumerate() {
            let path = tcx.def_path_str(item_id);
            write!(&mut s, "`{path}`").unwrap();
            if i == (ITEM_LIMIT - 1) && cycle_len > ITEM_LIMIT {
                write!(&mut s, " and {} more", cycle_len - 5).unwrap();
                break;
            }
            if cycle_len > 1 && i < cycle_len - 2 {
                s.push_str(", ");
            } else if cycle_len > 1 && i == cycle_len - 2 {
                s.push_str(" and ")
            }
        }
        s
    };
    struct_span_code_err!(
        tcx.dcx(),
        err_span,
        rustc_errors::E0072,
        "recursive type{} {} {} infinite size",
        pluralize!(cycle_len),
        items_list,
        pluralize!("has", cycle_len),
    )
    .with_multipart_suggestion(
        "insert some indirection (e.g., a `Box`, `Rc`, or `&`) to break the cycle",
        suggestion,
        Applicability::HasPlaceholders,
    )
    .emit()
}

fn find_item_ty_spans(
    tcx: TyCtxt<'_>,
    ty: &hir::Ty<'_>,
    needle: LocalDefId,
    spans: &mut Vec<Span>,
    seen_representable: &FxHashSet<LocalDefId>,
) {
    match ty.kind {
        hir::TyKind::Path(hir::QPath::Resolved(_, path)) => {
            if let Res::Def(kind, def_id) = path.res
                && matches!(kind, DefKind::Enum | DefKind::Struct | DefKind::Union)
            {
                let check_params = def_id.as_local().is_none_or(|def_id| {
                    if def_id == needle {
                        spans.push(ty.span);
                    }
                    seen_representable.contains(&def_id)
                });
                if check_params && let Some(args) = path.segments.last().unwrap().args {
                    let params_in_repr = tcx.params_in_repr(def_id);
                    // the domain size check is needed because the HIR may not be well-formed at this point
                    for (i, arg) in args.args.iter().enumerate().take(params_in_repr.domain_size())
                    {
                        if let hir::GenericArg::Type(ty) = arg
                            && params_in_repr.contains(i as u32)
                        {
                            find_item_ty_spans(
                                tcx,
                                ty.as_unambig_ty(),
                                needle,
                                spans,
                                seen_representable,
                            );
                        }
                    }
                }
            }
        }
        hir::TyKind::Array(ty, _) => find_item_ty_spans(tcx, ty, needle, spans, seen_representable),
        hir::TyKind::Tup(tys) => {
            tys.iter().for_each(|ty| find_item_ty_spans(tcx, ty, needle, spans, seen_representable))
        }
        _ => {}
    }
}
