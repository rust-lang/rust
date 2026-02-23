use std::collections::VecDeque;
use std::ops::ControlFlow;

use rustc_errors::codes::*;
use rustc_errors::{MultiSpan, struct_span_code_err};
use rustc_hir as hir;
use rustc_hir::def::DefKind;
use rustc_middle::dep_graph::DepKind;
use rustc_middle::query::CycleError;
use rustc_middle::query::plumbing::CyclePlaceholder;
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_middle::{bug, span_bug};
use rustc_span::{ErrorGuaranteed, Span};

use crate::job::report_cycle;

pub(crate) trait Value<'tcx>: Sized {
    fn from_cycle_error(tcx: TyCtxt<'tcx>, cycle_error: &CycleError, guar: ErrorGuaranteed)
    -> Self;
}

impl<'tcx, T> Value<'tcx> for T {
    default fn from_cycle_error(
        tcx: TyCtxt<'tcx>,
        cycle_error: &CycleError,
        _guar: ErrorGuaranteed,
    ) -> T {
        tcx.sess.dcx().abort_if_errors();
        bug!(
            "<{} as Value>::from_cycle_error called without errors: {:#?}",
            std::any::type_name::<T>(),
            cycle_error.cycle,
        );
    }
}

impl<'tcx> Value<'tcx> for Ty<'_> {
    fn from_cycle_error(tcx: TyCtxt<'tcx>, _: &CycleError, guar: ErrorGuaranteed) -> Self {
        // SAFETY: This is never called when `Self` is not `Ty<'tcx>`.
        // FIXME: Represent the above fact in the trait system somehow.
        unsafe { std::mem::transmute::<Ty<'tcx>, Ty<'_>>(Ty::new_error(tcx, guar)) }
    }
}

impl<'tcx> Value<'tcx> for Result<ty::EarlyBinder<'_, Ty<'_>>, CyclePlaceholder> {
    fn from_cycle_error(_tcx: TyCtxt<'tcx>, _: &CycleError, guar: ErrorGuaranteed) -> Self {
        Err(CyclePlaceholder(guar))
    }
}

impl<'tcx> Value<'tcx> for ty::SymbolName<'_> {
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

impl<'tcx> Value<'tcx> for ty::Binder<'_, ty::FnSig<'_>> {
    fn from_cycle_error(
        tcx: TyCtxt<'tcx>,
        cycle_error: &CycleError,
        guar: ErrorGuaranteed,
    ) -> Self {
        let err = Ty::new_error(tcx, guar);

        let arity = if let Some(info) = cycle_error.cycle.get(0)
            && info.frame.dep_kind == DepKind::fn_sig
            && let Some(def_id) = info.frame.def_id
            && let Some(node) = tcx.hir_get_if_local(def_id)
            && let Some(sig) = node.fn_sig()
        {
            sig.decl.inputs.len()
        } else {
            tcx.dcx().abort_if_errors();
            unreachable!()
        };

        let fn_sig = ty::Binder::dummy(tcx.mk_fn_sig(
            std::iter::repeat_n(err, arity),
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

impl<'tcx> Value<'tcx> for ty::EarlyBinder<'_, Ty<'_>> {
    fn from_cycle_error(
        tcx: TyCtxt<'tcx>,
        cycle_error: &CycleError,
        guar: ErrorGuaranteed,
    ) -> Self {
        ty::EarlyBinder::bind(Ty::from_cycle_error(tcx, cycle_error, guar))
    }
}

impl<'tcx> Value<'tcx> for ty::EarlyBinder<'_, ty::Binder<'_, ty::FnSig<'_>>> {
    fn from_cycle_error(
        tcx: TyCtxt<'tcx>,
        cycle_error: &CycleError,
        guar: ErrorGuaranteed,
    ) -> Self {
        ty::EarlyBinder::bind(ty::Binder::from_cycle_error(tcx, cycle_error, guar))
    }
}

impl<'tcx> Value<'tcx> for &[ty::Variance] {
    fn from_cycle_error(
        tcx: TyCtxt<'tcx>,
        cycle_error: &CycleError,
        _guar: ErrorGuaranteed,
    ) -> Self {
        search_for_cycle_permutation(
            &cycle_error.cycle,
            |cycle| {
                if let Some(info) = cycle.get(0)
                    && info.frame.dep_kind == DepKind::variances_of
                    && let Some(def_id) = info.frame.def_id
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

impl<'tcx, T> Value<'tcx> for Result<T, &'_ ty::layout::LayoutError<'_>> {
    fn from_cycle_error(
        tcx: TyCtxt<'tcx>,
        cycle_error: &CycleError,
        _guar: ErrorGuaranteed,
    ) -> Self {
        let diag = search_for_cycle_permutation(
            &cycle_error.cycle,
            |cycle| {
                if cycle[0].frame.dep_kind == DepKind::layout_of
                    && let Some(def_id) = cycle[0].frame.def_id_for_ty_in_cycle
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
                    for (i, info) in cycle.iter().enumerate() {
                        if info.frame.dep_kind != DepKind::layout_of {
                            continue;
                        }
                        let Some(frame_def_id) = info.frame.def_id_for_ty_in_cycle else {
                            continue;
                        };
                        let Some(frame_coroutine_kind) = tcx.coroutine_kind(frame_def_id) else {
                            continue;
                        };
                        let frame_span =
                            info.frame.info.default_span(cycle[(i + 1) % cycle.len()].span);
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
