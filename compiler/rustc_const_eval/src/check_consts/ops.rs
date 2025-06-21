//! Concrete error types for all operations which may be invalid in a certain const context.

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
use rustc_session::parse::add_feature_diagnostics;
use rustc_span::{BytePos, Pos, Span, Symbol, sym};
use rustc_trait_selection::error_reporting::traits::call_kind::{
    CallDesugaringKind, CallKind, call_kind,
};
use rustc_trait_selection::traits::SelectionContext;
use tracing::debug;

use super::ConstCx;
use crate::{errors, fluent_generated};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    Unstable {
        /// The feature that must be enabled to use this operation.
        gate: Symbol,
        /// Whether the feature gate was already checked (because the logic is a bit more
        /// complicated than just checking a single gate).
        gate_already_checked: bool,
        /// Whether it is allowed to use this operation from stable `const fn`.
        /// This will usually be `false`.
        safe_to_expose_on_stable: bool,
        /// We indicate whether this is a function call, since we can use targeted
        /// diagnostics for "callee is not safe to expose om stable".
        is_function_call: bool,
    },
    Forbidden,
}

#[derive(Clone, Copy)]
pub enum DiagImportance {
    /// An operation that must be removed for const-checking to pass.
    Primary,

    /// An operation that causes const-checking to fail, but is usually a side-effect of a `Primary` operation elsewhere.
    Secondary,
}

/// An operation that is *not allowed* in a const context.
pub trait NonConstOp<'tcx>: std::fmt::Debug {
    /// Returns an enum indicating whether this operation can be enabled with a feature gate.
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

/// A call to a function that is in a trait, or has trait bounds that make it conditionally-const.
#[derive(Debug)]
pub(crate) struct ConditionallyConstCall<'tcx> {
    pub callee: DefId,
    pub args: GenericArgsRef<'tcx>,
    pub span: Span,
    pub call_source: CallSource,
}

impl<'tcx> NonConstOp<'tcx> for ConditionallyConstCall<'tcx> {
    fn status_in_item(&self, _ccx: &ConstCx<'_, 'tcx>) -> Status {
        // We use the `const_trait_impl` gate for all conditionally-const calls.
        Status::Unstable {
            gate: sym::const_trait_impl,
            gate_already_checked: false,
            safe_to_expose_on_stable: false,
            // We don't want the "mark the callee as `#[rustc_const_stable_indirect]`" hint
            is_function_call: false,
        }
    }

    fn build_error(&self, ccx: &ConstCx<'_, 'tcx>, _: Span) -> Diag<'tcx> {
        let mut diag = build_error_for_const_call(
            ccx,
            self.callee,
            self.args,
            self.span,
            self.call_source,
            "conditionally",
            |_, _, _| {},
        );

        // Override code and mention feature.
        diag.code(E0658);
        add_feature_diagnostics(&mut diag, ccx.tcx.sess, sym::const_trait_impl);

        diag
    }
}

/// A function call where the callee is not marked as `const`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct FnCallNonConst<'tcx> {
    pub callee: DefId,
    pub args: GenericArgsRef<'tcx>,
    pub span: Span,
    pub call_source: CallSource,
}

impl<'tcx> NonConstOp<'tcx> for FnCallNonConst<'tcx> {
    // FIXME: make this translatable
    #[allow(rustc::diagnostic_outside_of_impl)]
    #[allow(rustc::untranslatable_diagnostic)]
    fn build_error(&self, ccx: &ConstCx<'_, 'tcx>, _: Span) -> Diag<'tcx> {
        let tcx = ccx.tcx;
        let caller = ccx.def_id();

        let mut err = build_error_for_const_call(
            ccx,
            self.callee,
            self.args,
            self.span,
            self.call_source,
            "non",
            |err, self_ty, trait_id| {
                // FIXME(const_trait_impl): Do we need any of this on the non-const codepath?

                let trait_ref = TraitRef::from_method(tcx, trait_id, self.args);

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
                                Some(trait_ref.def_id),
                                None,
                            );
                        }
                    }
                    ty::Adt(..) => {
                        let (infcx, param_env) =
                            tcx.infer_ctxt().build_with_typing_env(ccx.typing_env);
                        let obligation =
                            Obligation::new(tcx, ObligationCause::dummy(), param_env, trait_ref);
                        let mut selcx = SelectionContext::new(&infcx);
                        let implsrc = selcx.select(&obligation);
                        if let Ok(Some(ImplSource::UserDefined(data))) = implsrc {
                            // FIXME(const_trait_impl) revisit this
                            if !tcx.is_const_trait_impl(data.impl_def_id) {
                                let span = tcx.def_span(data.impl_def_id);
                                err.subdiagnostic(errors::NonConstImplNote { span });
                            }
                        }
                    }
                    _ => {}
                }
            },
        );

        if let ConstContext::Static(_) = ccx.const_kind() {
            err.note(fluent_generated::const_eval_lazy_lock);
        }

        err
    }
}

/// Build an error message reporting that a function call is not const (or only
/// conditionally const). In case that this call is desugared (like an operator
/// or sugar from something like a `for` loop), try to build a better error message
/// that doesn't call it a method.
fn build_error_for_const_call<'tcx>(
    ccx: &ConstCx<'_, 'tcx>,
    callee: DefId,
    args: ty::GenericArgsRef<'tcx>,
    span: Span,
    call_source: CallSource,
    non_or_conditionally: &'static str,
    note_trait_if_possible: impl FnOnce(&mut Diag<'tcx>, Ty<'tcx>, DefId),
) -> Diag<'tcx> {
    let tcx = ccx.tcx;

    let call_kind =
        call_kind(tcx, ccx.typing_env, callee, args, span, call_source.from_hir_call(), None);

    debug!(?call_kind);

    let mut err = match call_kind {
        CallKind::Normal { desugaring: Some((kind, self_ty)), .. } => {
            macro_rules! error {
                ($err:ident) => {
                    tcx.dcx().create_err(errors::$err {
                        span,
                        ty: self_ty,
                        kind: ccx.const_kind(),
                        non_or_conditionally,
                    })
                };
            }

            // Don't point at the trait if this is a desugaring...
            // FIXME(const_trait_impl): we could perhaps do this for `Iterator`.
            match kind {
                CallDesugaringKind::ForLoopIntoIter | CallDesugaringKind::ForLoopNext => {
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
            }
        }
        CallKind::FnCall { fn_trait_id, self_ty } => {
            let note = match self_ty.kind() {
                FnDef(def_id, ..) => {
                    let span = tcx.def_span(*def_id);
                    if ccx.tcx.is_const_fn(*def_id) {
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
                non_or_conditionally,
            });

            note_trait_if_possible(&mut err, self_ty, fn_trait_id);
            err
        }
        CallKind::Operator { trait_id, self_ty, .. } => {
            let mut err = if let CallSource::MatchCmp = call_source {
                tcx.dcx().create_err(errors::NonConstMatchEq {
                    span,
                    kind: ccx.const_kind(),
                    ty: self_ty,
                    non_or_conditionally,
                })
            } else {
                let mut sugg = None;

                if ccx.tcx.is_lang_item(trait_id, LangItem::PartialEq) {
                    match (args[0].kind(), args[1].kind()) {
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

                            if let Ok(call_str) = ccx.tcx.sess.source_map().span_to_snippet(span) {
                                if let Some(eq_idx) = call_str.find("==") {
                                    if let Some(rhs_idx) =
                                        call_str[(eq_idx + 2)..].find(|c: char| !c.is_whitespace())
                                    {
                                        let rhs_pos =
                                            span.lo() + BytePos::from_usize(eq_idx + 2 + rhs_idx);
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
                    non_or_conditionally,
                })
            };

            note_trait_if_possible(&mut err, self_ty, trait_id);
            err
        }
        CallKind::DerefCoercion { deref_target_span, deref_target_ty, self_ty } => {
            // Check first whether the source is accessible (issue #87060)
            let target = if let Some(deref_target_span) = deref_target_span
                && tcx.sess.source_map().is_span_accessible(deref_target_span)
            {
                Some(deref_target_span)
            } else {
                None
            };

            let mut err = tcx.dcx().create_err(errors::NonConstDerefCoercion {
                span,
                ty: self_ty,
                kind: ccx.const_kind(),
                target_ty: deref_target_ty,
                deref_target: target,
                non_or_conditionally,
            });

            note_trait_if_possible(&mut err, self_ty, tcx.require_lang_item(LangItem::Deref, span));
            err
        }
        _ if tcx.opt_parent(callee) == tcx.get_diagnostic_item(sym::FmtArgumentsNew) => {
            ccx.dcx().create_err(errors::NonConstFmtMacroCall {
                span,
                kind: ccx.const_kind(),
                non_or_conditionally,
            })
        }
        _ => ccx.dcx().create_err(errors::NonConstFnCall {
            span,
            def_descr: ccx.tcx.def_descr(callee),
            def_path_str: ccx.tcx.def_path_str_with_args(callee, args),
            kind: ccx.const_kind(),
            non_or_conditionally,
        }),
    };

    err.note(format!(
        "calls in {}s are limited to constant functions, \
             tuple structs and tuple variants",
        ccx.const_kind(),
    ));

    err
}

/// A call to an `#[unstable]` const fn, `#[rustc_const_unstable]` function or trait.
///
/// Contains the name of the feature that would allow the use of this function/trait.
#[derive(Debug)]
pub(crate) struct CallUnstable {
    pub def_id: DefId,
    pub feature: Symbol,
    /// If this is true, then the feature is enabled, but we need to still check if it is safe to
    /// expose on stable.
    pub feature_enabled: bool,
    pub safe_to_expose_on_stable: bool,
    /// true if `def_id` is the function we are calling, false if `def_id` is an unstable trait.
    pub is_function_call: bool,
}

impl<'tcx> NonConstOp<'tcx> for CallUnstable {
    fn status_in_item(&self, _ccx: &ConstCx<'_, 'tcx>) -> Status {
        Status::Unstable {
            gate: self.feature,
            gate_already_checked: self.feature_enabled,
            safe_to_expose_on_stable: self.safe_to_expose_on_stable,
            is_function_call: self.is_function_call,
        }
    }

    fn build_error(&self, ccx: &ConstCx<'_, 'tcx>, span: Span) -> Diag<'tcx> {
        assert!(!self.feature_enabled);
        let mut err = if self.is_function_call {
            ccx.dcx().create_err(errors::UnstableConstFn {
                span,
                def_path: ccx.tcx.def_path_str(self.def_id),
            })
        } else {
            ccx.dcx().create_err(errors::UnstableConstTrait {
                span,
                def_path: ccx.tcx.def_path_str(self.def_id),
            })
        };
        ccx.tcx.disabled_nightly_features(&mut err, [(String::new(), self.feature)]);
        err
    }
}

/// A call to an intrinsic that is just not const-callable at all.
#[derive(Debug)]
pub(crate) struct IntrinsicNonConst {
    pub name: Symbol,
}

impl<'tcx> NonConstOp<'tcx> for IntrinsicNonConst {
    fn build_error(&self, ccx: &ConstCx<'_, 'tcx>, span: Span) -> Diag<'tcx> {
        ccx.dcx().create_err(errors::NonConstIntrinsic {
            span,
            name: self.name,
            kind: ccx.const_kind(),
        })
    }
}

/// A call to an intrinsic that is just not const-callable at all.
#[derive(Debug)]
pub(crate) struct IntrinsicUnstable {
    pub name: Symbol,
    pub feature: Symbol,
    pub const_stable_indirect: bool,
}

impl<'tcx> NonConstOp<'tcx> for IntrinsicUnstable {
    fn status_in_item(&self, _ccx: &ConstCx<'_, 'tcx>) -> Status {
        Status::Unstable {
            gate: self.feature,
            gate_already_checked: false,
            safe_to_expose_on_stable: self.const_stable_indirect,
            // We do *not* want to suggest to mark the intrinsic as `const_stable_indirect`,
            // that's not a trivial change!
            is_function_call: false,
        }
    }

    fn build_error(&self, ccx: &ConstCx<'_, 'tcx>, span: Span) -> Diag<'tcx> {
        ccx.dcx().create_err(errors::UnstableIntrinsic {
            span,
            name: self.name,
            feature: self.feature,
            suggestion: ccx.tcx.crate_level_attribute_injection_span(),
        })
    }
}

#[derive(Debug)]
pub(crate) struct Coroutine(pub hir::CoroutineKind);
impl<'tcx> NonConstOp<'tcx> for Coroutine {
    fn status_in_item(&self, _: &ConstCx<'_, 'tcx>) -> Status {
        match self.0 {
            hir::CoroutineKind::Desugared(
                hir::CoroutineDesugaring::Async,
                hir::CoroutineSource::Block,
            )
            // FIXME(coroutines): eventually we want to gate const coroutine coroutines behind a
            // different feature.
            | hir::CoroutineKind::Coroutine(_) => Status::Unstable {
                gate: sym::const_async_blocks,
                gate_already_checked: false,
                safe_to_expose_on_stable: false,
                is_function_call: false,
            },
            _ => Status::Forbidden,
        }
    }

    fn build_error(&self, ccx: &ConstCx<'_, 'tcx>, span: Span) -> Diag<'tcx> {
        let msg = format!("{} are not allowed in {}s", self.0.to_plural_string(), ccx.const_kind());
        if let Status::Unstable { gate, .. } = self.status_in_item(ccx) {
            ccx.tcx.sess.create_feature_err(errors::UnallowedOpInConstContext { span, msg }, gate)
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
    pub dropped_at: Span,
    pub dropped_ty: Ty<'tcx>,
    pub needs_non_const_drop: bool,
}
impl<'tcx> NonConstOp<'tcx> for LiveDrop<'tcx> {
    fn status_in_item(&self, _ccx: &ConstCx<'_, 'tcx>) -> Status {
        if self.needs_non_const_drop {
            Status::Forbidden
        } else {
            Status::Unstable {
                gate: sym::const_destruct,
                gate_already_checked: false,
                safe_to_expose_on_stable: false,
                is_function_call: false,
            }
        }
    }

    fn build_error(&self, ccx: &ConstCx<'_, 'tcx>, span: Span) -> Diag<'tcx> {
        if self.needs_non_const_drop {
            ccx.dcx().create_err(errors::LiveDrop {
                span,
                dropped_ty: self.dropped_ty,
                kind: ccx.const_kind(),
                dropped_at: self.dropped_at,
            })
        } else {
            ccx.tcx.sess.create_feature_err(
                errors::LiveDrop {
                    span,
                    dropped_ty: self.dropped_ty,
                    kind: ccx.const_kind(),
                    dropped_at: self.dropped_at,
                },
                sym::const_destruct,
            )
        }
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
            hir::BorrowKind::Ref | hir::BorrowKind::Pin => {
                ccx.dcx().create_err(errors::MutableRefEscaping {
                    span,
                    kind: ccx.const_kind(),
                    teach: ccx.tcx.sess.teach(E0764),
                })
            }
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
