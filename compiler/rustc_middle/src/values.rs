use std::collections::VecDeque;
use std::fmt::Write;
use std::ops::ControlFlow;

use rustc_data_structures::fx::FxHashSet;
use rustc_errors::codes::*;
use rustc_errors::{Applicability, MultiSpan, pluralize, struct_span_code_err};
use rustc_hir as hir;
use rustc_hir::def::{DefKind, Res};
use rustc_query_system::Value;
use rustc_query_system::query::{CycleError, report_cycle};
use rustc_span::def_id::LocalDefId;
use rustc_span::{ErrorGuaranteed, Span};

use crate::dep_graph::dep_kinds;
use crate::query::plumbing::CyclePlaceholder;
use crate::ty::{self, Representability, Ty, TyCtxt};

impl<'tcx> Value<TyCtxt<'tcx>> for Ty<'_> {
    fn from_cycle_error(tcx: TyCtxt<'tcx>, _: &CycleError, guar: ErrorGuaranteed) -> Self {
        // SAFETY: This is never called when `Self` is not `Ty<'tcx>`.
        // FIXME: Represent the above fact in the trait system somehow.
        unsafe { std::mem::transmute::<Ty<'tcx>, Ty<'_>>(Ty::new_error(tcx, guar)) }
    }
}

impl<'tcx> Value<TyCtxt<'tcx>> for Result<ty::EarlyBinder<'_, Ty<'_>>, CyclePlaceholder> {
    fn from_cycle_error(_tcx: TyCtxt<'tcx>, _: &CycleError, guar: ErrorGuaranteed) -> Self {
        Err(CyclePlaceholder(guar))
    }
}

impl<'tcx> Value<TyCtxt<'tcx>> for ty::SymbolName<'_> {
    fn from_cycle_error(tcx: TyCtxt<'tcx>, _: &CycleError, _guar: ErrorGuaranteed) -> Self {
        // SAFETY: This is never called when `Self` is not `SymbolName<'tcx>`.
        // FIXME: Represent the above fact in the trait system somehow.
        unsafe {
            std::mem::transmute::<ty::SymbolName<'tcx>, ty::SymbolName<'_>>(ty::SymbolName::new(
                tcx, "<error>",
            ))
        }
    }
}

impl<'tcx> Value<TyCtxt<'tcx>> for ty::Binder<'_, ty::FnSig<'_>> {
    fn from_cycle_error(
        tcx: TyCtxt<'tcx>,
        cycle_error: &CycleError,
        guar: ErrorGuaranteed,
    ) -> Self {
        let err = Ty::new_error(tcx, guar);

        let arity = if let Some(frame) = cycle_error.cycle.get(0)
            && frame.query.dep_kind == dep_kinds::fn_sig
            && let Some(def_id) = frame.query.def_id
            && let Some(node) = tcx.hir_get_if_local(def_id)
            && let Some(sig) = node.fn_sig()
        {
            sig.decl.inputs.len()
        } else {
            tcx.dcx().abort_if_errors();
            unreachable!()
        };

        let fn_sig = ty::Binder::dummy(tcx.mk_fn_sig(
            std::iter::repeat(err).take(arity),
            err,
            false,
            rustc_hir::Safety::Safe,
            rustc_abi::ExternAbi::Rust,
        ));

        // SAFETY: This is never called when `Self` is not `ty::Binder<'tcx, ty::FnSig<'tcx>>`.
        // FIXME: Represent the above fact in the trait system somehow.
        unsafe { std::mem::transmute::<ty::PolyFnSig<'tcx>, ty::Binder<'_, ty::FnSig<'_>>>(fn_sig) }
    }
}

impl<'tcx> Value<TyCtxt<'tcx>> for Representability {
    fn from_cycle_error(
        tcx: TyCtxt<'tcx>,
        cycle_error: &CycleError,
        _guar: ErrorGuaranteed,
    ) -> Self {
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
}

impl<'tcx> Value<TyCtxt<'tcx>> for ty::EarlyBinder<'_, Ty<'_>> {
    fn from_cycle_error(
        tcx: TyCtxt<'tcx>,
        cycle_error: &CycleError,
        guar: ErrorGuaranteed,
    ) -> Self {
        ty::EarlyBinder::bind(Ty::from_cycle_error(tcx, cycle_error, guar))
    }
}

impl<'tcx> Value<TyCtxt<'tcx>> for ty::EarlyBinder<'_, ty::Binder<'_, ty::FnSig<'_>>> {
    fn from_cycle_error(
        tcx: TyCtxt<'tcx>,
        cycle_error: &CycleError,
        guar: ErrorGuaranteed,
    ) -> Self {
        ty::EarlyBinder::bind(ty::Binder::from_cycle_error(tcx, cycle_error, guar))
    }
}

impl<'tcx> Value<TyCtxt<'tcx>> for &[ty::Variance] {
    fn from_cycle_error(
        tcx: TyCtxt<'tcx>,
        cycle_error: &CycleError,
        _guar: ErrorGuaranteed,
    ) -> Self {
        search_for_cycle_permutation(
            &cycle_error.cycle,
            |cycle| {
                if let Some(frame) = cycle.get(0)
                    && frame.query.dep_kind == dep_kinds::variances_of
                    && let Some(def_id) = frame.query.def_id
                {
                    let n = tcx.generics_of(def_id).own_params.len();
                    ControlFlow::Break(vec![ty::Bivariant; n].leak())
                } else {
                    ControlFlow::Continue(())
                }
            },
            || {
                span_bug!(
                    cycle_error.usage.as_ref().unwrap().0,
                    "only `variances_of` returns `&[ty::Variance]`"
                )
            },
        )
    }
}

// Take a cycle of `Q` and try `try_cycle` on every permutation, falling back to `otherwise`.
fn search_for_cycle_permutation<Q, T>(
    cycle: &[Q],
    try_cycle: impl Fn(&mut VecDeque<&Q>) -> ControlFlow<T, ()>,
    otherwise: impl FnOnce() -> T,
) -> T {
    let mut cycle: VecDeque<_> = cycle.iter().collect();
    for _ in 0..cycle.len() {
        match try_cycle(&mut cycle) {
            ControlFlow::Continue(_) => {
                cycle.rotate_left(1);
            }
            ControlFlow::Break(t) => return t,
        }
    }

    otherwise()
}

impl<'tcx, T> Value<TyCtxt<'tcx>> for Result<T, &'_ ty::layout::LayoutError<'_>> {
    fn from_cycle_error(
        tcx: TyCtxt<'tcx>,
        cycle_error: &CycleError,
        _guar: ErrorGuaranteed,
    ) -> Self {
        let diag = search_for_cycle_permutation(
            &cycle_error.cycle,
            |cycle| {
                if cycle[0].query.dep_kind == dep_kinds::layout_of
                    && let Some(def_id) = cycle[0].query.def_id_for_ty_in_cycle
                    && let Some(def_id) = def_id.as_local()
                    && let def_kind = tcx.def_kind(def_id)
                    && matches!(def_kind, DefKind::Closure)
                    && let Some(coroutine_kind) = tcx.coroutine_kind(def_id)
                {
                    // FIXME: `def_span` for an fn-like coroutine will point to the fn's body
                    // due to interactions between the desugaring into a closure expr and the
                    // def_span code. I'm not motivated to fix it, because I tried and it was
                    // not working, so just hack around it by grabbing the parent fn's span.
                    let span = if coroutine_kind.is_fn_like() {
                        tcx.def_span(tcx.local_parent(def_id))
                    } else {
                        tcx.def_span(def_id)
                    };
                    let mut diag = struct_span_code_err!(
                        tcx.sess.dcx(),
                        span,
                        E0733,
                        "recursion in {} {} requires boxing",
                        tcx.def_kind_descr_article(def_kind, def_id.to_def_id()),
                        tcx.def_kind_descr(def_kind, def_id.to_def_id()),
                    );
                    for (i, frame) in cycle.iter().enumerate() {
                        if frame.query.dep_kind != dep_kinds::layout_of {
                            continue;
                        }
                        let Some(frame_def_id) = frame.query.def_id_for_ty_in_cycle else {
                            continue;
                        };
                        let Some(frame_coroutine_kind) = tcx.coroutine_kind(frame_def_id) else {
                            continue;
                        };
                        let frame_span =
                            frame.query.info.default_span(cycle[(i + 1) % cycle.len()].span);
                        if frame_span.is_dummy() {
                            continue;
                        }
                        if i == 0 {
                            diag.span_label(frame_span, "recursive call here");
                        } else {
                            let coroutine_span: Span = if frame_coroutine_kind.is_fn_like() {
                                tcx.def_span(tcx.parent(frame_def_id))
                            } else {
                                tcx.def_span(frame_def_id)
                            };
                            let mut multispan = MultiSpan::from_span(coroutine_span);
                            multispan
                                .push_span_label(frame_span, "...leading to this recursive call");
                            diag.span_note(
                                multispan,
                                format!("which leads to this {}", tcx.def_descr(frame_def_id)),
                            );
                        }
                    }
                    // FIXME: We could report a structured suggestion if we had
                    // enough info here... Maybe we can use a hacky HIR walker.
                    if matches!(
                        coroutine_kind,
                        hir::CoroutineKind::Desugared(hir::CoroutineDesugaring::Async, _)
                    ) {
                        diag.note("a recursive `async fn` call must introduce indirection such as `Box::pin` to avoid an infinitely sized future");
                    }

                    ControlFlow::Break(diag)
                } else {
                    ControlFlow::Continue(())
                }
            },
            || report_cycle(tcx.sess, cycle_error),
        );

        let guar = diag.emit();

        // tcx.arena.alloc cannot be used because we are not allowed to use &'tcx LayoutError under
        // min_specialization. Since this is an error path anyways, leaking doesn't matter (and really,
        // tcx.arena.alloc is pretty much equal to leaking).
        Err(Box::leak(Box::new(ty::layout::LayoutError::Cycle(guar))))
    }
}

// item_and_field_ids should form a cycle where each field contains the
// type in the next element in the list
pub fn recursive_type_error(
    tcx: TyCtxt<'_>,
    mut item_and_field_ids: Vec<(LocalDefId, LocalDefId)>,
    representable_ids: &FxHashSet<LocalDefId>,
) -> ErrorGuaranteed {
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
        E0072,
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
