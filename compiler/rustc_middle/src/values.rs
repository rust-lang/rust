use crate::dep_graph::DepKind;
use rustc_data_structures::fx::FxHashSet;
use rustc_errors::{pluralize, struct_span_err, Applicability, MultiSpan};
use rustc_hir as hir;
use rustc_hir::def::{DefKind, Res};
use rustc_middle::ty::Representability;
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_query_system::query::QueryInfo;
use rustc_query_system::Value;
use rustc_span::def_id::LocalDefId;
use rustc_span::Span;

use std::fmt::Write;

impl<'tcx> Value<TyCtxt<'tcx>, DepKind> for Ty<'_> {
    fn from_cycle_error(tcx: TyCtxt<'tcx>, _: &[QueryInfo<DepKind>]) -> Self {
        // SAFETY: This is never called when `Self` is not `Ty<'tcx>`.
        // FIXME: Represent the above fact in the trait system somehow.
        unsafe { std::mem::transmute::<Ty<'tcx>, Ty<'_>>(tcx.ty_error_misc()) }
    }
}

impl<'tcx> Value<TyCtxt<'tcx>, DepKind> for ty::SymbolName<'_> {
    fn from_cycle_error(tcx: TyCtxt<'tcx>, _: &[QueryInfo<DepKind>]) -> Self {
        // SAFETY: This is never called when `Self` is not `SymbolName<'tcx>`.
        // FIXME: Represent the above fact in the trait system somehow.
        unsafe {
            std::mem::transmute::<ty::SymbolName<'tcx>, ty::SymbolName<'_>>(ty::SymbolName::new(
                tcx, "<error>",
            ))
        }
    }
}

impl<'tcx> Value<TyCtxt<'tcx>, DepKind> for ty::Binder<'_, ty::FnSig<'_>> {
    fn from_cycle_error(tcx: TyCtxt<'tcx>, stack: &[QueryInfo<DepKind>]) -> Self {
        let err = tcx.ty_error_misc();

        let arity = if let Some(frame) = stack.get(0)
            && frame.query.dep_kind == DepKind::fn_sig
            && let Some(def_id) = frame.query.def_id
            && let Some(node) = tcx.hir().get_if_local(def_id)
            && let Some(sig) = node.fn_sig()
        {
            sig.decl.inputs.len() + sig.decl.implicit_self.has_implicit_self() as usize
        } else {
            tcx.sess.abort_if_errors();
            unreachable!()
        };

        let fn_sig = ty::Binder::dummy(tcx.mk_fn_sig(
            std::iter::repeat(err).take(arity),
            err,
            false,
            rustc_hir::Unsafety::Normal,
            rustc_target::spec::abi::Abi::Rust,
        ));

        // SAFETY: This is never called when `Self` is not `ty::Binder<'tcx, ty::FnSig<'tcx>>`.
        // FIXME: Represent the above fact in the trait system somehow.
        unsafe { std::mem::transmute::<ty::PolyFnSig<'tcx>, ty::Binder<'_, ty::FnSig<'_>>>(fn_sig) }
    }
}

impl<'tcx> Value<TyCtxt<'tcx>, DepKind> for Representability {
    fn from_cycle_error(tcx: TyCtxt<'tcx>, cycle: &[QueryInfo<DepKind>]) -> Self {
        let mut item_and_field_ids = Vec::new();
        let mut representable_ids = FxHashSet::default();
        for info in cycle {
            if info.query.dep_kind == DepKind::representability
                && let Some(field_id) = info.query.def_id
                && let Some(field_id) = field_id.as_local()
                && let Some(DefKind::Field) = info.query.def_kind
            {
                let parent_id = tcx.parent(field_id.to_def_id());
                let item_id = match tcx.def_kind(parent_id) {
                    DefKind::Variant => tcx.parent(parent_id),
                    _ => parent_id,
                };
                item_and_field_ids.push((item_id.expect_local(), field_id));
            }
        }
        for info in cycle {
            if info.query.dep_kind == DepKind::representability_adt_ty
                && let Some(def_id) = info.query.ty_adt_id
                && let Some(def_id) = def_id.as_local()
                && !item_and_field_ids.iter().any(|&(id, _)| id == def_id)
            {
                representable_ids.insert(def_id);
            }
        }
        recursive_type_error(tcx, item_and_field_ids, &representable_ids);
        Representability::Infinite
    }
}

impl<'tcx> Value<TyCtxt<'tcx>, DepKind> for ty::EarlyBinder<Ty<'_>> {
    fn from_cycle_error(tcx: TyCtxt<'tcx>, cycle: &[QueryInfo<DepKind>]) -> Self {
        ty::EarlyBinder::bind(Ty::from_cycle_error(tcx, cycle))
    }
}

impl<'tcx> Value<TyCtxt<'tcx>, DepKind> for ty::EarlyBinder<ty::Binder<'_, ty::FnSig<'_>>> {
    fn from_cycle_error(tcx: TyCtxt<'tcx>, cycle: &[QueryInfo<DepKind>]) -> Self {
        ty::EarlyBinder::bind(ty::Binder::from_cycle_error(tcx, cycle))
    }
}

impl<'tcx, T> Value<TyCtxt<'tcx>, DepKind> for Result<T, ty::layout::LayoutError<'_>> {
    fn from_cycle_error(_tcx: TyCtxt<'tcx>, _cycle: &[QueryInfo<DepKind>]) -> Self {
        Err(ty::layout::LayoutError::Cycle)
    }
}

// item_and_field_ids should form a cycle where each field contains the
// type in the next element in the list
pub fn recursive_type_error(
    tcx: TyCtxt<'_>,
    mut item_and_field_ids: Vec<(LocalDefId, LocalDefId)>,
    representable_ids: &FxHashSet<LocalDefId>,
) {
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
        let hir_id = tcx.hir().local_def_id_to_hir_id(field_id);
        let hir::Node::Field(field) = tcx.hir().get(hir_id) else { bug!("expected field") };
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
    let mut err = struct_span_err!(
        tcx.sess,
        err_span,
        E0072,
        "recursive type{} {} {} infinite size",
        pluralize!(cycle_len),
        items_list,
        pluralize!("has", cycle_len),
    );
    err.multipart_suggestion(
        "insert some indirection (e.g., a `Box`, `Rc`, or `&`) to break the cycle",
        suggestion,
        Applicability::HasPlaceholders,
    );
    err.emit();
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
                && kind != DefKind::TyAlias {
                let check_params = def_id.as_local().map_or(true, |def_id| {
                    if def_id == needle {
                        spans.push(ty.span);
                    }
                    seen_representable.contains(&def_id)
                });
                if check_params && let Some(args) = path.segments.last().unwrap().args {
                    let params_in_repr = tcx.params_in_repr(def_id);
                    // the domain size check is needed because the HIR may not be well-formed at this point
                    for (i, arg) in args.args.iter().enumerate().take(params_in_repr.domain_size()) {
                        if let hir::GenericArg::Type(ty) = arg && params_in_repr.contains(i as u32) {
                            find_item_ty_spans(tcx, ty, needle, spans, seen_representable);
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
