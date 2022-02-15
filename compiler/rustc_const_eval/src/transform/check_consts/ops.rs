//! Concrete error types for all operations which may be invalid in a certain const context.

use rustc_errors::{struct_span_err, Applicability, DiagnosticBuilder};
use rustc_hir as hir;
use rustc_hir::def_id::DefId;
use rustc_infer::infer::TyCtxtInferExt;
use rustc_infer::traits::{ImplSource, Obligation, ObligationCause};
use rustc_middle::mir;
use rustc_middle::ty::print::with_no_trimmed_paths;
use rustc_middle::ty::subst::{GenericArgKind, SubstsRef};
use rustc_middle::ty::{
    suggest_constraining_type_param, Adt, Closure, FnDef, FnPtr, Param, TraitPredicate, Ty,
};
use rustc_middle::ty::{Binder, BoundConstness, ImplPolarity, TraitRef};
use rustc_session::parse::feature_err;
use rustc_span::symbol::sym;
use rustc_span::{BytePos, Pos, Span, Symbol};
use rustc_trait_selection::traits::SelectionContext;

use super::ConstCx;
use crate::util::{call_kind, CallDesugaringKind, CallKind};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    Allowed,
    Unstable(Symbol),
    Forbidden,
}

#[derive(Clone, Copy)]
pub enum DiagnosticImportance {
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

    fn importance(&self) -> DiagnosticImportance {
        DiagnosticImportance::Primary
    }

    fn build_error(&self, ccx: &ConstCx<'_, 'tcx>, span: Span) -> DiagnosticBuilder<'tcx>;
}

#[derive(Debug)]
pub struct FloatingPointOp;
impl<'tcx> NonConstOp<'tcx> for FloatingPointOp {
    fn status_in_item(&self, ccx: &ConstCx<'_, 'tcx>) -> Status {
        if ccx.const_kind() == hir::ConstContext::ConstFn {
            Status::Unstable(sym::const_fn_floating_point_arithmetic)
        } else {
            Status::Allowed
        }
    }

    fn build_error(&self, ccx: &ConstCx<'_, 'tcx>, span: Span) -> DiagnosticBuilder<'tcx> {
        feature_err(
            &ccx.tcx.sess.parse_sess,
            sym::const_fn_floating_point_arithmetic,
            span,
            &format!("floating point arithmetic is not allowed in {}s", ccx.const_kind()),
        )
    }
}

/// A function call where the callee is a pointer.
#[derive(Debug)]
pub struct FnCallIndirect;
impl<'tcx> NonConstOp<'tcx> for FnCallIndirect {
    fn build_error(&self, ccx: &ConstCx<'_, 'tcx>, span: Span) -> DiagnosticBuilder<'tcx> {
        ccx.tcx.sess.struct_span_err(span, "function pointers are not allowed in const fn")
    }
}

/// A function call where the callee is not marked as `const`.
#[derive(Debug, Clone, Copy)]
pub struct FnCallNonConst<'tcx> {
    pub caller: DefId,
    pub callee: DefId,
    pub substs: SubstsRef<'tcx>,
    pub span: Span,
    pub from_hir_call: bool,
}

impl<'tcx> NonConstOp<'tcx> for FnCallNonConst<'tcx> {
    fn build_error(&self, ccx: &ConstCx<'_, 'tcx>, _: Span) -> DiagnosticBuilder<'tcx> {
        let FnCallNonConst { caller, callee, substs, span, from_hir_call } = *self;
        let ConstCx { tcx, param_env, .. } = *ccx;

        let diag_trait = |mut err, self_ty: Ty<'_>, trait_id| {
            let trait_ref = TraitRef::from_method(tcx, trait_id, substs);

            match self_ty.kind() {
                Param(param_ty) => {
                    debug!(?param_ty);
                    if let Some(generics) = caller
                        .as_local()
                        .map(|id| tcx.hir().local_def_id_to_hir_id(id))
                        .map(|id| tcx.hir().get(id))
                        .as_ref()
                        .and_then(|node| node.generics())
                    {
                        let constraint = with_no_trimmed_paths(|| {
                            format!("~const {}", trait_ref.print_only_trait_path())
                        });
                        suggest_constraining_type_param(
                            tcx,
                            generics,
                            &mut err,
                            &param_ty.name.as_str(),
                            &constraint,
                            None,
                        );
                    }
                }
                Adt(..) => {
                    let obligation = Obligation::new(
                        ObligationCause::dummy(),
                        param_env,
                        Binder::dummy(TraitPredicate {
                            trait_ref,
                            constness: BoundConstness::NotConst,
                            polarity: ImplPolarity::Positive,
                        }),
                    );

                    let implsrc = tcx.infer_ctxt().enter(|infcx| {
                        let mut selcx = SelectionContext::new(&infcx);
                        selcx.select(&obligation)
                    });

                    if let Ok(Some(ImplSource::UserDefined(data))) = implsrc {
                        let span =
                            tcx.sess.source_map().guess_head_span(tcx.def_span(data.impl_def_id));
                        err.span_note(span, "impl defined here, but it is not `const`");
                    }
                }
                _ => {}
            }

            err
        };

        let call_kind = call_kind(tcx, ccx.param_env, callee, substs, span, from_hir_call, None);

        debug!(?call_kind);

        let mut err = match call_kind {
            CallKind::Normal { desugaring: Some((kind, self_ty)), .. } => {
                macro_rules! error {
                    ($fmt:literal) => {
                        struct_span_err!(tcx.sess, span, E0015, $fmt, self_ty, ccx.const_kind())
                    };
                }

                let err = match kind {
                    CallDesugaringKind::ForLoopIntoIter => {
                        error!("cannot convert `{}` into an iterator in {}s")
                    }
                    CallDesugaringKind::QuestionBranch => {
                        error!("`?` cannot determine the branch of `{}` in {}s")
                    }
                    CallDesugaringKind::QuestionFromResidual => {
                        error!("`?` cannot convert from residual of `{}` in {}s")
                    }
                    CallDesugaringKind::TryBlockFromOutput => {
                        error!("`try` block cannot convert `{}` to the result in {}s")
                    }
                };

                diag_trait(err, self_ty, kind.trait_def_id(tcx))
            }
            CallKind::FnCall { fn_trait_id, self_ty } => {
                let mut err = struct_span_err!(
                    tcx.sess,
                    span,
                    E0015,
                    "cannot call non-const closure in {}s",
                    ccx.const_kind(),
                );

                match self_ty.kind() {
                    FnDef(def_id, ..) => {
                        let span = tcx.sess.source_map().guess_head_span(tcx.def_span(*def_id));
                        if ccx.tcx.is_const_fn_raw(*def_id) {
                            span_bug!(span, "calling const FnDef errored when it shouldn't");
                        }

                        err.span_note(span, "function defined here, but it is not `const`");
                    }
                    FnPtr(..) => {
                        err.note(&format!(
                            "function pointers need an RFC before allowed to be called in {}s",
                            ccx.const_kind()
                        ));
                    }
                    Closure(..) => {
                        err.note(&format!(
                            "closures need an RFC before allowed to be called in {}s",
                            ccx.const_kind()
                        ));
                    }
                    _ => {}
                }

                diag_trait(err, self_ty, fn_trait_id)
            }
            CallKind::Operator { trait_id, self_ty, .. } => {
                let mut err = struct_span_err!(
                    tcx.sess,
                    span,
                    E0015,
                    "cannot call non-const operator in {}s",
                    ccx.const_kind()
                );

                if Some(trait_id) == ccx.tcx.lang_items().eq_trait() {
                    match (substs[0].unpack(), substs[1].unpack()) {
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
                                        err.multipart_suggestion(
                                            "consider dereferencing here",
                                            vec![
                                                (span.shrink_to_lo(), deref.clone()),
                                                (rhs_span, deref),
                                            ],
                                            Applicability::MachineApplicable,
                                        );
                                    }
                                }
                            }
                        }
                        _ => {}
                    }
                }

                diag_trait(err, self_ty, trait_id)
            }
            CallKind::DerefCoercion { deref_target, deref_target_ty, self_ty } => {
                let mut err = struct_span_err!(
                    tcx.sess,
                    span,
                    E0015,
                    "cannot perform deref coercion on `{}` in {}s",
                    self_ty,
                    ccx.const_kind()
                );

                err.note(&format!("attempting to deref into `{}`", deref_target_ty));

                // Check first whether the source is accessible (issue #87060)
                if tcx.sess.source_map().span_to_snippet(deref_target).is_ok() {
                    err.span_note(deref_target, "deref defined here");
                }

                diag_trait(err, self_ty, tcx.lang_items().deref_trait().unwrap())
            }
            _ => struct_span_err!(
                ccx.tcx.sess,
                span,
                E0015,
                "cannot call non-const fn `{}` in {}s",
                ccx.tcx.def_path_str_with_substs(callee, substs),
                ccx.const_kind(),
            ),
        };

        err.note(&format!(
            "calls in {}s are limited to constant functions, \
             tuple structs and tuple variants",
            ccx.const_kind(),
        ));

        err
    }
}

/// A call to an `#[unstable]` const fn or `#[rustc_const_unstable]` function.
///
/// Contains the name of the feature that would allow the use of this function.
#[derive(Debug)]
pub struct FnCallUnstable(pub DefId, pub Option<Symbol>);

impl<'tcx> NonConstOp<'tcx> for FnCallUnstable {
    fn build_error(&self, ccx: &ConstCx<'_, 'tcx>, span: Span) -> DiagnosticBuilder<'tcx> {
        let FnCallUnstable(def_id, feature) = *self;

        let mut err = ccx.tcx.sess.struct_span_err(
            span,
            &format!("`{}` is not yet stable as a const fn", ccx.tcx.def_path_str(def_id)),
        );

        if ccx.is_const_stable_const_fn() {
            err.help("const-stable functions can only call other const-stable functions");
        } else if ccx.tcx.sess.is_nightly_build() {
            if let Some(feature) = feature {
                err.help(&format!(
                    "add `#![feature({})]` to the crate attributes to enable",
                    feature
                ));
            }
        }

        err
    }
}

#[derive(Debug)]
pub struct FnPtrCast;
impl<'tcx> NonConstOp<'tcx> for FnPtrCast {
    fn status_in_item(&self, ccx: &ConstCx<'_, 'tcx>) -> Status {
        if ccx.const_kind() != hir::ConstContext::ConstFn {
            Status::Allowed
        } else {
            Status::Unstable(sym::const_fn_fn_ptr_basics)
        }
    }

    fn build_error(&self, ccx: &ConstCx<'_, 'tcx>, span: Span) -> DiagnosticBuilder<'tcx> {
        feature_err(
            &ccx.tcx.sess.parse_sess,
            sym::const_fn_fn_ptr_basics,
            span,
            &format!("function pointer casts are not allowed in {}s", ccx.const_kind()),
        )
    }
}

#[derive(Debug)]
pub struct Generator(pub hir::GeneratorKind);
impl<'tcx> NonConstOp<'tcx> for Generator {
    fn status_in_item(&self, _: &ConstCx<'_, 'tcx>) -> Status {
        if let hir::GeneratorKind::Async(hir::AsyncGeneratorKind::Block) = self.0 {
            Status::Unstable(sym::const_async_blocks)
        } else {
            Status::Forbidden
        }
    }

    fn build_error(&self, ccx: &ConstCx<'_, 'tcx>, span: Span) -> DiagnosticBuilder<'tcx> {
        let msg = format!("{}s are not allowed in {}s", self.0, ccx.const_kind());
        if let hir::GeneratorKind::Async(hir::AsyncGeneratorKind::Block) = self.0 {
            feature_err(&ccx.tcx.sess.parse_sess, sym::const_async_blocks, span, &msg)
        } else {
            ccx.tcx.sess.struct_span_err(span, &msg)
        }
    }
}

#[derive(Debug)]
pub struct HeapAllocation;
impl<'tcx> NonConstOp<'tcx> for HeapAllocation {
    fn build_error(&self, ccx: &ConstCx<'_, 'tcx>, span: Span) -> DiagnosticBuilder<'tcx> {
        let mut err = struct_span_err!(
            ccx.tcx.sess,
            span,
            E0010,
            "allocations are not allowed in {}s",
            ccx.const_kind()
        );
        err.span_label(span, format!("allocation not allowed in {}s", ccx.const_kind()));
        if ccx.tcx.sess.teach(&err.get_code().unwrap()) {
            err.note(
                "The value of statics and constants must be known at compile time, \
                 and they live for the entire lifetime of a program. Creating a boxed \
                 value allocates memory on the heap at runtime, and therefore cannot \
                 be done at compile time.",
            );
        }
        err
    }
}

#[derive(Debug)]
pub struct InlineAsm;
impl<'tcx> NonConstOp<'tcx> for InlineAsm {
    fn build_error(&self, ccx: &ConstCx<'_, 'tcx>, span: Span) -> DiagnosticBuilder<'tcx> {
        struct_span_err!(
            ccx.tcx.sess,
            span,
            E0015,
            "inline assembly is not allowed in {}s",
            ccx.const_kind()
        )
    }
}

#[derive(Debug)]
pub struct LiveDrop {
    pub dropped_at: Option<Span>,
}
impl<'tcx> NonConstOp<'tcx> for LiveDrop {
    fn build_error(&self, ccx: &ConstCx<'_, 'tcx>, span: Span) -> DiagnosticBuilder<'tcx> {
        let mut err = struct_span_err!(
            ccx.tcx.sess,
            span,
            E0493,
            "destructors cannot be evaluated at compile-time"
        );
        err.span_label(span, format!("{}s cannot evaluate destructors", ccx.const_kind()));
        if let Some(span) = self.dropped_at {
            err.span_label(span, "value is dropped here");
        }
        err
    }
}

#[derive(Debug)]
/// A borrow of a type that contains an `UnsafeCell` somewhere. The borrow never escapes to
/// the final value of the constant.
pub struct TransientCellBorrow;
impl<'tcx> NonConstOp<'tcx> for TransientCellBorrow {
    fn status_in_item(&self, _: &ConstCx<'_, 'tcx>) -> Status {
        Status::Unstable(sym::const_refs_to_cell)
    }
    fn importance(&self) -> DiagnosticImportance {
        // The cases that cannot possibly work will already emit a `CellBorrow`, so we should
        // not additionally emit a feature gate error if activating the feature gate won't work.
        DiagnosticImportance::Secondary
    }
    fn build_error(&self, ccx: &ConstCx<'_, 'tcx>, span: Span) -> DiagnosticBuilder<'tcx> {
        feature_err(
            &ccx.tcx.sess.parse_sess,
            sym::const_refs_to_cell,
            span,
            "cannot borrow here, since the borrowed element may contain interior mutability",
        )
    }
}

#[derive(Debug)]
/// A borrow of a type that contains an `UnsafeCell` somewhere. The borrow might escape to
/// the final value of the constant, and thus we cannot allow this (for now). We may allow
/// it in the future for static items.
pub struct CellBorrow;
impl<'tcx> NonConstOp<'tcx> for CellBorrow {
    fn build_error(&self, ccx: &ConstCx<'_, 'tcx>, span: Span) -> DiagnosticBuilder<'tcx> {
        let mut err = struct_span_err!(
            ccx.tcx.sess,
            span,
            E0492,
            "{}s cannot refer to interior mutable data",
            ccx.const_kind(),
        );
        err.span_label(
            span,
            "this borrow of an interior mutable value may end up in the final value",
        );
        if let hir::ConstContext::Static(_) = ccx.const_kind() {
            err.help(
                "to fix this, the value can be extracted to a separate \
                `static` item and then referenced",
            );
        }
        if ccx.tcx.sess.teach(&err.get_code().unwrap()) {
            err.note(
                "A constant containing interior mutable data behind a reference can allow you
                 to modify that data. This would make multiple uses of a constant to be able to
                 see different values and allow circumventing the `Send` and `Sync` requirements
                 for shared mutable data, which is unsound.",
            );
        }
        err
    }
}

#[derive(Debug)]
/// This op is for `&mut` borrows in the trailing expression of a constant
/// which uses the "enclosing scopes rule" to leak its locals into anonymous
/// static or const items.
pub struct MutBorrow(pub hir::BorrowKind);

impl<'tcx> NonConstOp<'tcx> for MutBorrow {
    fn status_in_item(&self, _ccx: &ConstCx<'_, 'tcx>) -> Status {
        Status::Forbidden
    }

    fn importance(&self) -> DiagnosticImportance {
        // If there were primary errors (like non-const function calls), do not emit further
        // errors about mutable references.
        DiagnosticImportance::Secondary
    }

    fn build_error(&self, ccx: &ConstCx<'_, 'tcx>, span: Span) -> DiagnosticBuilder<'tcx> {
        let raw = match self.0 {
            hir::BorrowKind::Raw => "raw ",
            hir::BorrowKind::Ref => "",
        };

        let mut err = struct_span_err!(
            ccx.tcx.sess,
            span,
            E0764,
            "{}mutable references are not allowed in the final value of {}s",
            raw,
            ccx.const_kind(),
        );

        if ccx.tcx.sess.teach(&err.get_code().unwrap()) {
            err.note(
                "References in statics and constants may only refer \
                      to immutable values.\n\n\
                      Statics are shared everywhere, and if they refer to \
                      mutable data one might violate memory safety since \
                      holding multiple mutable references to shared data \
                      is not allowed.\n\n\
                      If you really want global mutable state, try using \
                      static mut or a global UnsafeCell.",
            );
        }
        err
    }
}

#[derive(Debug)]
pub struct TransientMutBorrow(pub hir::BorrowKind);

impl<'tcx> NonConstOp<'tcx> for TransientMutBorrow {
    fn status_in_item(&self, _: &ConstCx<'_, 'tcx>) -> Status {
        Status::Unstable(sym::const_mut_refs)
    }

    fn build_error(&self, ccx: &ConstCx<'_, 'tcx>, span: Span) -> DiagnosticBuilder<'tcx> {
        let raw = match self.0 {
            hir::BorrowKind::Raw => "raw ",
            hir::BorrowKind::Ref => "",
        };

        feature_err(
            &ccx.tcx.sess.parse_sess,
            sym::const_mut_refs,
            span,
            &format!("{}mutable references are not allowed in {}s", raw, ccx.const_kind()),
        )
    }
}

#[derive(Debug)]
pub struct MutDeref;
impl<'tcx> NonConstOp<'tcx> for MutDeref {
    fn status_in_item(&self, _: &ConstCx<'_, 'tcx>) -> Status {
        Status::Unstable(sym::const_mut_refs)
    }

    fn importance(&self) -> DiagnosticImportance {
        // Usually a side-effect of a `TransientMutBorrow` somewhere.
        DiagnosticImportance::Secondary
    }

    fn build_error(&self, ccx: &ConstCx<'_, 'tcx>, span: Span) -> DiagnosticBuilder<'tcx> {
        feature_err(
            &ccx.tcx.sess.parse_sess,
            sym::const_mut_refs,
            span,
            &format!("mutation through a reference is not allowed in {}s", ccx.const_kind()),
        )
    }
}

/// A call to a `panic()` lang item where the first argument is _not_ a `&str`.
#[derive(Debug)]
pub struct PanicNonStr;
impl<'tcx> NonConstOp<'tcx> for PanicNonStr {
    fn build_error(&self, ccx: &ConstCx<'_, 'tcx>, span: Span) -> DiagnosticBuilder<'tcx> {
        ccx.tcx.sess.struct_span_err(
            span,
            "argument to `panic!()` in a const context must have type `&str`",
        )
    }
}

/// Comparing raw pointers for equality.
/// Not currently intended to ever be allowed, even behind a feature gate: operation depends on
/// allocation base addresses that are not known at compile-time.
#[derive(Debug)]
pub struct RawPtrComparison;
impl<'tcx> NonConstOp<'tcx> for RawPtrComparison {
    fn build_error(&self, ccx: &ConstCx<'_, 'tcx>, span: Span) -> DiagnosticBuilder<'tcx> {
        let mut err = ccx
            .tcx
            .sess
            .struct_span_err(span, "pointers cannot be reliably compared during const eval");
        err.note(
            "see issue #53020 <https://github.com/rust-lang/rust/issues/53020> \
            for more information",
        );
        err
    }
}

#[derive(Debug)]
pub struct RawMutPtrDeref;
impl<'tcx> NonConstOp<'tcx> for RawMutPtrDeref {
    fn status_in_item(&self, _: &ConstCx<'_, '_>) -> Status {
        Status::Unstable(sym::const_mut_refs)
    }

    fn build_error(&self, ccx: &ConstCx<'_, 'tcx>, span: Span) -> DiagnosticBuilder<'tcx> {
        feature_err(
            &ccx.tcx.sess.parse_sess,
            sym::const_mut_refs,
            span,
            &format!("dereferencing raw mutable pointers in {}s is unstable", ccx.const_kind(),),
        )
    }
}

/// Casting raw pointer or function pointer to an integer.
/// Not currently intended to ever be allowed, even behind a feature gate: operation depends on
/// allocation base addresses that are not known at compile-time.
#[derive(Debug)]
pub struct RawPtrToIntCast;
impl<'tcx> NonConstOp<'tcx> for RawPtrToIntCast {
    fn build_error(&self, ccx: &ConstCx<'_, 'tcx>, span: Span) -> DiagnosticBuilder<'tcx> {
        let mut err = ccx
            .tcx
            .sess
            .struct_span_err(span, "pointers cannot be cast to integers during const eval");
        err.note("at compile-time, pointers do not have an integer value");
        err.note(
            "avoiding this restriction via `transmute`, `union`, or raw pointers leads to compile-time undefined behavior",
        );
        err
    }
}

/// An access to a (non-thread-local) `static`.
#[derive(Debug)]
pub struct StaticAccess;
impl<'tcx> NonConstOp<'tcx> for StaticAccess {
    fn status_in_item(&self, ccx: &ConstCx<'_, 'tcx>) -> Status {
        if let hir::ConstContext::Static(_) = ccx.const_kind() {
            Status::Allowed
        } else {
            Status::Forbidden
        }
    }

    fn build_error(&self, ccx: &ConstCx<'_, 'tcx>, span: Span) -> DiagnosticBuilder<'tcx> {
        let mut err = struct_span_err!(
            ccx.tcx.sess,
            span,
            E0013,
            "{}s cannot refer to statics",
            ccx.const_kind()
        );
        err.help(
            "consider extracting the value of the `static` to a `const`, and referring to that",
        );
        if ccx.tcx.sess.teach(&err.get_code().unwrap()) {
            err.note(
                "`static` and `const` variables can refer to other `const` variables. \
                    A `const` variable, however, cannot refer to a `static` variable.",
            );
            err.help("To fix this, the value can be extracted to a `const` and then used.");
        }
        err
    }
}

/// An access to a thread-local `static`.
#[derive(Debug)]
pub struct ThreadLocalAccess;
impl<'tcx> NonConstOp<'tcx> for ThreadLocalAccess {
    fn build_error(&self, ccx: &ConstCx<'_, 'tcx>, span: Span) -> DiagnosticBuilder<'tcx> {
        struct_span_err!(
            ccx.tcx.sess,
            span,
            E0625,
            "thread-local statics cannot be \
            accessed at compile-time"
        )
    }
}

// Types that cannot appear in the signature or locals of a `const fn`.
pub mod ty {
    use super::*;

    #[derive(Debug)]
    pub struct MutRef(pub mir::LocalKind);
    impl<'tcx> NonConstOp<'tcx> for MutRef {
        fn status_in_item(&self, _ccx: &ConstCx<'_, 'tcx>) -> Status {
            Status::Unstable(sym::const_mut_refs)
        }

        fn importance(&self) -> DiagnosticImportance {
            match self.0 {
                mir::LocalKind::Var | mir::LocalKind::Temp => DiagnosticImportance::Secondary,
                mir::LocalKind::ReturnPointer | mir::LocalKind::Arg => {
                    DiagnosticImportance::Primary
                }
            }
        }

        fn build_error(&self, ccx: &ConstCx<'_, 'tcx>, span: Span) -> DiagnosticBuilder<'tcx> {
            feature_err(
                &ccx.tcx.sess.parse_sess,
                sym::const_mut_refs,
                span,
                &format!("mutable references are not allowed in {}s", ccx.const_kind()),
            )
        }
    }

    #[derive(Debug)]
    pub struct FnPtr(pub mir::LocalKind);
    impl<'tcx> NonConstOp<'tcx> for FnPtr {
        fn importance(&self) -> DiagnosticImportance {
            match self.0 {
                mir::LocalKind::Var | mir::LocalKind::Temp => DiagnosticImportance::Secondary,
                mir::LocalKind::ReturnPointer | mir::LocalKind::Arg => {
                    DiagnosticImportance::Primary
                }
            }
        }

        fn status_in_item(&self, ccx: &ConstCx<'_, 'tcx>) -> Status {
            if ccx.const_kind() != hir::ConstContext::ConstFn {
                Status::Allowed
            } else {
                Status::Unstable(sym::const_fn_fn_ptr_basics)
            }
        }

        fn build_error(&self, ccx: &ConstCx<'_, 'tcx>, span: Span) -> DiagnosticBuilder<'tcx> {
            feature_err(
                &ccx.tcx.sess.parse_sess,
                sym::const_fn_fn_ptr_basics,
                span,
                &format!("function pointers cannot appear in {}s", ccx.const_kind()),
            )
        }
    }

    #[derive(Debug)]
    pub struct ImplTrait;
    impl<'tcx> NonConstOp<'tcx> for ImplTrait {
        fn status_in_item(&self, _: &ConstCx<'_, '_>) -> Status {
            Status::Unstable(sym::const_impl_trait)
        }

        fn build_error(&self, ccx: &ConstCx<'_, 'tcx>, span: Span) -> DiagnosticBuilder<'tcx> {
            feature_err(
                &ccx.tcx.sess.parse_sess,
                sym::const_impl_trait,
                span,
                &format!("`impl Trait` is not allowed in {}s", ccx.const_kind()),
            )
        }
    }

    #[derive(Debug)]
    pub struct TraitBound(pub mir::LocalKind);
    impl<'tcx> NonConstOp<'tcx> for TraitBound {
        fn importance(&self) -> DiagnosticImportance {
            match self.0 {
                mir::LocalKind::Var | mir::LocalKind::Temp => DiagnosticImportance::Secondary,
                mir::LocalKind::ReturnPointer | mir::LocalKind::Arg => {
                    DiagnosticImportance::Primary
                }
            }
        }

        fn status_in_item(&self, ccx: &ConstCx<'_, 'tcx>) -> Status {
            if ccx.const_kind() != hir::ConstContext::ConstFn {
                Status::Allowed
            } else {
                Status::Unstable(sym::const_fn_trait_bound)
            }
        }

        fn build_error(&self, ccx: &ConstCx<'_, 'tcx>, span: Span) -> DiagnosticBuilder<'tcx> {
            let mut err = feature_err(
                &ccx.tcx.sess.parse_sess,
                sym::const_fn_trait_bound,
                span,
                "trait bounds other than `Sized` on const fn parameters are unstable",
            );

            match ccx.fn_sig() {
                Some(fn_sig) if !fn_sig.span.contains(span) => {
                    err.span_label(fn_sig.span, "function declared as const here");
                }
                _ => {}
            }

            err
        }
    }

    #[derive(Debug)]
    pub struct DynTrait(pub mir::LocalKind);
    impl<'tcx> NonConstOp<'tcx> for DynTrait {
        fn importance(&self) -> DiagnosticImportance {
            match self.0 {
                mir::LocalKind::Var | mir::LocalKind::Temp => DiagnosticImportance::Secondary,
                mir::LocalKind::ReturnPointer | mir::LocalKind::Arg => {
                    DiagnosticImportance::Primary
                }
            }
        }

        fn status_in_item(&self, ccx: &ConstCx<'_, 'tcx>) -> Status {
            if ccx.const_kind() != hir::ConstContext::ConstFn {
                Status::Allowed
            } else {
                Status::Unstable(sym::const_fn_trait_bound)
            }
        }

        fn build_error(&self, ccx: &ConstCx<'_, 'tcx>, span: Span) -> DiagnosticBuilder<'tcx> {
            let mut err = feature_err(
                &ccx.tcx.sess.parse_sess,
                sym::const_fn_trait_bound,
                span,
                "trait objects in const fn are unstable",
            );

            match ccx.fn_sig() {
                Some(fn_sig) if !fn_sig.span.contains(span) => {
                    err.span_label(fn_sig.span, "function declared as const here");
                }
                _ => {}
            }

            err
        }
    }

    /// A trait bound with the `?const Trait` opt-out
    #[derive(Debug)]
    pub struct TraitBoundNotConst;
    impl<'tcx> NonConstOp<'tcx> for TraitBoundNotConst {
        fn status_in_item(&self, _: &ConstCx<'_, 'tcx>) -> Status {
            Status::Unstable(sym::const_trait_bound_opt_out)
        }

        fn build_error(&self, ccx: &ConstCx<'_, 'tcx>, span: Span) -> DiagnosticBuilder<'tcx> {
            feature_err(
                &ccx.tcx.sess.parse_sess,
                sym::const_trait_bound_opt_out,
                span,
                "`?const Trait` syntax is unstable",
            )
        }
    }
}
