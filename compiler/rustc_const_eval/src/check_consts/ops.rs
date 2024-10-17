//! Concrete error types for all operations which may be invalid in a certain const context.

use hir::def_id::LocalDefId;
use hir::{ConstContext, LangItem};
use rustc_errors::Diag;
use rustc_errors::codes::*;
use rustc_hir as hir;
use rustc_hir::def_id::DefId;
use rustc_infer::infer::TyCtxtInferExt;
use rustc_infer::traits::{ImplSource, Obligation, ObligationCause};
use rustc_middle::mir::CallSource;
use rustc_middle::span_bug;
use rustc_middle::ty::print::{PrintTraitRefExt as _, with_no_trimmed_paths};
use rustc_middle::ty::{
    self, Closure, FnDef, FnPtr, GenericArgKind, GenericArgsRef, Param, TraitRef, Ty,
    suggest_constraining_type_param,
};
use rustc_middle::util::{CallDesugaringKind, CallKind, call_kind};
use rustc_span::symbol::sym;
use rustc_span::{BytePos, Pos, Span, Symbol};
use rustc_trait_selection::traits::SelectionContext;
use tracing::debug;

use super::ConstCx;
use crate::{errors, fluent_generated};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    Allowed,
    Unstable(Symbol),
    Forbidden,
}

#[derive(Clone, Copy)]
pub enum DiagImportance {
    /// An operation that must be removed for const-checking to pass.
    Primary,

    /// An operation that causes const-checking to fail, but is usually a side-effect of a `Primary` operation elsewhere.
    Secondary,
}

/// An operation that is not *always* allowed in a const context.
pub trait NonConstOp<'tcx>: std::fmt::Debug {
    /// Returns an enum indicating whether this operation is allowed within the given item.
    fn status_in_item(&self, _ccx: &ConstCx<'_, 'tcx>) -> Status {
        Status::Forbidden
    }

    fn importance(&self) -> DiagImportance {
        DiagImportance::Primary
    }

    fn build_error(&self, ccx: &ConstCx<'_, 'tcx>, span: Span) -> Diag<'tcx>;
}

/// A function call where the callee is a pointer.
#[derive(Debug)]
pub(crate) struct FnCallIndirect;
impl<'tcx> NonConstOp<'tcx> for FnCallIndirect {
    fn build_error(&self, ccx: &ConstCx<'_, 'tcx>, span: Span) -> Diag<'tcx> {
        ccx.dcx().create_err(errors::UnallowedFnPointerCall { span, kind: ccx.const_kind() })
    }
}

/// A function call where the callee is not marked as `const`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct FnCallNonConst<'tcx> {
    pub caller: LocalDefId,
    pub callee: DefId,
    pub args: GenericArgsRef<'tcx>,
    pub span: Span,
    pub call_source: CallSource,
    pub feature: Option<Symbol>,
}

impl<'tcx> NonConstOp<'tcx> for FnCallNonConst<'tcx> {
    // FIXME: make this translatable
    #[allow(rustc::diagnostic_outside_of_impl)]
    #[allow(rustc::untranslatable_diagnostic)]
    fn build_error(&self, ccx: &ConstCx<'_, 'tcx>, _: Span) -> Diag<'tcx> {
        let FnCallNonConst { caller, callee, args, span, call_source, feature } = *self;
        let ConstCx { tcx, param_env, body, .. } = *ccx;

        let diag_trait = |err, self_ty: Ty<'_>, trait_id| {
            let trait_ref = TraitRef::from_method(tcx, trait_id, args);

            match self_ty.kind() {
                Param(param_ty) => {
                    debug!(?param_ty);
                    if let Some(generics) = tcx.hir_node_by_def_id(caller).generics() {
                        let constraint = with_no_trimmed_paths!(format!(
                            "~const {}",
                            trait_ref.print_trait_sugared(),
                        ));
                        suggest_constraining_type_param(
                            tcx,
                            generics,
                            err,
                            param_ty.name.as_str(),
                            &constraint,
                            None,
                            None,
                        );
                    }
                }
                ty::Adt(..) => {
                    let obligation =
                        Obligation::new(tcx, ObligationCause::dummy(), param_env, trait_ref);

                    let infcx = tcx.infer_ctxt().build();
                    let mut selcx = SelectionContext::new(&infcx);
                    let implsrc = selcx.select(&obligation);

                    if let Ok(Some(ImplSource::UserDefined(data))) = implsrc {
                        // FIXME(effects) revisit this
                        if !tcx.is_const_trait_impl_raw(data.impl_def_id) {
                            let span = tcx.def_span(data.impl_def_id);
                            err.subdiagnostic(errors::NonConstImplNote { span });
                        }
                    }
                }
                _ => {}
            }
        };

        let call_kind =
            call_kind(tcx, ccx.param_env, callee, args, span, call_source.from_hir_call(), None);

        debug!(?call_kind);

        let mut err = match call_kind {
            CallKind::Normal { desugaring: Some((kind, self_ty)), .. } => {
                macro_rules! error {
                    ($err:ident) => {
                        tcx.dcx().create_err(errors::$err {
                            span,
                            ty: self_ty,
                            kind: ccx.const_kind(),
                        })
                    };
                }

                let mut err = match kind {
                    CallDesugaringKind::ForLoopIntoIter => {
                        error!(NonConstForLoopIntoIter)
                    }
                    CallDesugaringKind::QuestionBranch => {
                        error!(NonConstQuestionBranch)
                    }
                    CallDesugaringKind::QuestionFromResidual => {
                        error!(NonConstQuestionFromResidual)
                    }
                    CallDesugaringKind::TryBlockFromOutput => {
                        error!(NonConstTryBlockFromOutput)
                    }
                    CallDesugaringKind::Await => {
                        error!(NonConstAwait)
                    }
                };

                diag_trait(&mut err, self_ty, kind.trait_def_id(tcx));
                err
            }
            CallKind::FnCall { fn_trait_id, self_ty } => {
                let note = match self_ty.kind() {
                    FnDef(def_id, ..) => {
                        let span = tcx.def_span(*def_id);
                        if ccx.tcx.is_const_fn_raw(*def_id) {
                            span_bug!(span, "calling const FnDef errored when it shouldn't");
                        }

                        Some(errors::NonConstClosureNote::FnDef { span })
                    }
                    FnPtr(..) => Some(errors::NonConstClosureNote::FnPtr),
                    Closure(..) => Some(errors::NonConstClosureNote::Closure),
                    _ => None,
                };

                let mut err = tcx.dcx().create_err(errors::NonConstClosure {
                    span,
                    kind: ccx.const_kind(),
                    note,
                });

                diag_trait(&mut err, self_ty, fn_trait_id);
                err
            }
            CallKind::Operator { trait_id, self_ty, .. } => {
                let mut err = if let CallSource::MatchCmp = call_source {
                    tcx.dcx().create_err(errors::NonConstMatchEq {
                        span,
                        kind: ccx.const_kind(),
                        ty: self_ty,
                    })
                } else {
                    let mut sugg = None;

                    if ccx.tcx.is_lang_item(trait_id, LangItem::PartialEq) {
                        match (args[0].unpack(), args[1].unpack()) {
                            (GenericArgKind::Type(self_ty), GenericArgKind::Type(rhs_ty))
                                if self_ty == rhs_ty
                                    && self_ty.is_ref()
                                    && self_ty.peel_refs().is_primitive() =>
                            {
                                let mut num_refs = 0;
                                let mut tmp_ty = self_ty;
                                while let rustc_middle::ty::Ref(_, inner_ty, _) = tmp_ty.kind() {
                                    num_refs += 1;
                                    tmp_ty = *inner_ty;
                                }
                                let deref = "*".repeat(num_refs);

                                if let Ok(call_str) =
                                    ccx.tcx.sess.source_map().span_to_snippet(span)
                                {
                                    if let Some(eq_idx) = call_str.find("==") {
                                        if let Some(rhs_idx) = call_str[(eq_idx + 2)..]
                                            .find(|c: char| !c.is_whitespace())
                                        {
                                            let rhs_pos = span.lo()
                                                + BytePos::from_usize(eq_idx + 2 + rhs_idx);
                                            let rhs_span = span.with_lo(rhs_pos).with_hi(rhs_pos);
                                            sugg = Some(errors::ConsiderDereferencing {
                                                deref,
                                                span: span.shrink_to_lo(),
                                                rhs_span,
                                            });
                                        }
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                    tcx.dcx().create_err(errors::NonConstOperator {
                        span,
                        kind: ccx.const_kind(),
                        sugg,
                    })
                };

                diag_trait(&mut err, self_ty, trait_id);
                err
            }
            CallKind::DerefCoercion { deref_target, deref_target_ty, self_ty } => {
                // Check first whether the source is accessible (issue #87060)
                let target = if tcx.sess.source_map().is_span_accessible(deref_target) {
                    Some(deref_target)
                } else {
                    None
                };

                let mut err = tcx.dcx().create_err(errors::NonConstDerefCoercion {
                    span,
                    ty: self_ty,
                    kind: ccx.const_kind(),
                    target_ty: deref_target_ty,
                    deref_target: target,
                });

                diag_trait(&mut err, self_ty, tcx.require_lang_item(LangItem::Deref, Some(span)));
                err
            }
            _ if tcx.opt_parent(callee) == tcx.get_diagnostic_item(sym::ArgumentMethods) => {
                ccx.dcx().create_err(errors::NonConstFmtMacroCall { span, kind: ccx.const_kind() })
            }
            _ => ccx.dcx().create_err(errors::NonConstFnCall {
                span,
                def_path_str: ccx.tcx.def_path_str_with_args(callee, args),
                kind: ccx.const_kind(),
            }),
        };

        err.note(format!(
            "calls in {}s are limited to constant functions, \
             tuple structs and tuple variants",
            ccx.const_kind(),
        ));

        if let Some(feature) = feature {
            ccx.tcx.disabled_nightly_features(
                &mut err,
                body.source.def_id().as_local().map(|local| ccx.tcx.local_def_id_to_hir_id(local)),
                [(String::new(), feature)],
            );
        }

        if let ConstContext::Static(_) = ccx.const_kind() {
            err.note(fluent_generated::const_eval_lazy_lock);
        }

        err
    }
}

/// A call to an `#[unstable]` const fn or `#[rustc_const_unstable]` function.
///
/// Contains the name of the feature that would allow the use of this function.
#[derive(Debug)]
pub(crate) struct FnCallUnstable(pub DefId, pub Option<Symbol>);

impl<'tcx> NonConstOp<'tcx> for FnCallUnstable {
    fn build_error(&self, ccx: &ConstCx<'_, 'tcx>, span: Span) -> Diag<'tcx> {
        let FnCallUnstable(def_id, feature) = *self;

        let mut err = ccx
            .dcx()
            .create_err(errors::UnstableConstFn { span, def_path: ccx.tcx.def_path_str(def_id) });

        // FIXME: make this translatable
        #[allow(rustc::untranslatable_diagnostic)]
        if ccx.is_const_stable_const_fn() {
            err.help(fluent_generated::const_eval_const_stable);
        } else if ccx.tcx.sess.is_nightly_build() {
            if let Some(feature) = feature {
                err.help(format!("add `#![feature({feature})]` to the crate attributes to enable"));
            }
        }

        err
    }
}

#[derive(Debug)]
pub(crate) struct Coroutine(pub hir::CoroutineKind);
impl<'tcx> NonConstOp<'tcx> for Coroutine {
    fn status_in_item(&self, _: &ConstCx<'_, 'tcx>) -> Status {
        if let hir::CoroutineKind::Desugared(
            hir::CoroutineDesugaring::Async,
            hir::CoroutineSource::Block,
        ) = self.0
        {
            Status::Unstable(sym::const_async_blocks)
        } else {
            Status::Forbidden
        }
    }

    fn build_error(&self, ccx: &ConstCx<'_, 'tcx>, span: Span) -> Diag<'tcx> {
        let msg = format!("{:#}s are not allowed in {}s", self.0, ccx.const_kind());
        if let hir::CoroutineKind::Desugared(
            hir::CoroutineDesugaring::Async,
            hir::CoroutineSource::Block,
        ) = self.0
        {
            ccx.tcx.sess.create_feature_err(
                errors::UnallowedOpInConstContext { span, msg },
                sym::const_async_blocks,
            )
        } else {
            ccx.dcx().create_err(errors::UnallowedOpInConstContext { span, msg })
        }
    }
}

#[derive(Debug)]
pub(crate) struct HeapAllocation;
impl<'tcx> NonConstOp<'tcx> for HeapAllocation {
    fn build_error(&self, ccx: &ConstCx<'_, 'tcx>, span: Span) -> Diag<'tcx> {
        ccx.dcx().create_err(errors::UnallowedHeapAllocations {
            span,
            kind: ccx.const_kind(),
            teach: ccx.tcx.sess.teach(E0010),
        })
    }
}

#[derive(Debug)]
pub(crate) struct InlineAsm;
impl<'tcx> NonConstOp<'tcx> for InlineAsm {
    fn build_error(&self, ccx: &ConstCx<'_, 'tcx>, span: Span) -> Diag<'tcx> {
        ccx.dcx().create_err(errors::UnallowedInlineAsm { span, kind: ccx.const_kind() })
    }
}

#[derive(Debug)]
pub(crate) struct LiveDrop<'tcx> {
    pub dropped_at: Option<Span>,
    pub dropped_ty: Ty<'tcx>,
}
impl<'tcx> NonConstOp<'tcx> for LiveDrop<'tcx> {
    fn build_error(&self, ccx: &ConstCx<'_, 'tcx>, span: Span) -> Diag<'tcx> {
        ccx.dcx().create_err(errors::LiveDrop {
            span,
            dropped_ty: self.dropped_ty,
            kind: ccx.const_kind(),
            dropped_at: self.dropped_at,
        })
    }
}

#[derive(Debug)]
/// A borrow of a type that contains an `UnsafeCell` somewhere. The borrow might escape to
/// the final value of the constant, and thus we cannot allow this (for now). We may allow
/// it in the future for static items.
pub(crate) struct EscapingCellBorrow;
impl<'tcx> NonConstOp<'tcx> for EscapingCellBorrow {
    fn importance(&self) -> DiagImportance {
        // Most likely the code will try to do mutation with these borrows, which
        // triggers its own errors. Only show this one if that does not happen.
        DiagImportance::Secondary
    }
    fn build_error(&self, ccx: &ConstCx<'_, 'tcx>, span: Span) -> Diag<'tcx> {
        ccx.dcx().create_err(errors::InteriorMutableRefEscaping {
            span,
            opt_help: matches!(ccx.const_kind(), hir::ConstContext::Static(_)),
            kind: ccx.const_kind(),
            teach: ccx.tcx.sess.teach(E0492),
        })
    }
}

#[derive(Debug)]
/// This op is for `&mut` borrows in the trailing expression of a constant
/// which uses the "enclosing scopes rule" to leak its locals into anonymous
/// static or const items.
pub(crate) struct EscapingMutBorrow(pub hir::BorrowKind);

impl<'tcx> NonConstOp<'tcx> for EscapingMutBorrow {
    fn status_in_item(&self, _ccx: &ConstCx<'_, 'tcx>) -> Status {
        Status::Forbidden
    }

    fn importance(&self) -> DiagImportance {
        // Most likely the code will try to do mutation with these borrows, which
        // triggers its own errors. Only show this one if that does not happen.
        DiagImportance::Secondary
    }

    fn build_error(&self, ccx: &ConstCx<'_, 'tcx>, span: Span) -> Diag<'tcx> {
        match self.0 {
            hir::BorrowKind::Raw => ccx.tcx.dcx().create_err(errors::MutableRawEscaping {
                span,
                kind: ccx.const_kind(),
                teach: ccx.tcx.sess.teach(E0764),
            }),
            hir::BorrowKind::Ref => ccx.dcx().create_err(errors::MutableRefEscaping {
                span,
                kind: ccx.const_kind(),
                teach: ccx.tcx.sess.teach(E0764),
            }),
        }
    }
}

/// A call to a `panic()` lang item where the first argument is _not_ a `&str`.
#[derive(Debug)]
pub(crate) struct PanicNonStr;
impl<'tcx> NonConstOp<'tcx> for PanicNonStr {
    fn build_error(&self, ccx: &ConstCx<'_, 'tcx>, span: Span) -> Diag<'tcx> {
        ccx.dcx().create_err(errors::PanicNonStrErr { span })
    }
}

/// Comparing raw pointers for equality.
/// Not currently intended to ever be allowed, even behind a feature gate: operation depends on
/// allocation base addresses that are not known at compile-time.
#[derive(Debug)]
pub(crate) struct RawPtrComparison;
impl<'tcx> NonConstOp<'tcx> for RawPtrComparison {
    fn build_error(&self, ccx: &ConstCx<'_, 'tcx>, span: Span) -> Diag<'tcx> {
        // FIXME(const_trait_impl): revert to span_bug?
        ccx.dcx().create_err(errors::RawPtrComparisonErr { span })
    }
}

/// Casting raw pointer or function pointer to an integer.
/// Not currently intended to ever be allowed, even behind a feature gate: operation depends on
/// allocation base addresses that are not known at compile-time.
#[derive(Debug)]
pub(crate) struct RawPtrToIntCast;
impl<'tcx> NonConstOp<'tcx> for RawPtrToIntCast {
    fn build_error(&self, ccx: &ConstCx<'_, 'tcx>, span: Span) -> Diag<'tcx> {
        ccx.dcx().create_err(errors::RawPtrToIntErr { span })
    }
}

/// An access to a thread-local `static`.
#[derive(Debug)]
pub(crate) struct ThreadLocalAccess;
impl<'tcx> NonConstOp<'tcx> for ThreadLocalAccess {
    fn build_error(&self, ccx: &ConstCx<'_, 'tcx>, span: Span) -> Diag<'tcx> {
        ccx.dcx().create_err(errors::ThreadLocalAccessErr { span })
    }
}
