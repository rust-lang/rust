//! Concrete error types for all operations which may be invalid in a certain const context.

use rustc_errors::{struct_span_err, DiagnosticBuilder};
use rustc_hir as hir;
use rustc_hir::def_id::DefId;
use rustc_middle::mir;
use rustc_session::config::nightly_options;
use rustc_session::parse::feature_err;
use rustc_span::symbol::sym;
use rustc_span::{Span, Symbol};

use super::ConstCx;

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
pub trait NonConstOp: std::fmt::Debug {
    /// Returns an enum indicating whether this operation is allowed within the given item.
    fn status_in_item(&self, _ccx: &ConstCx<'_, '_>) -> Status {
        Status::Forbidden
    }

    fn importance(&self) -> DiagnosticImportance {
        DiagnosticImportance::Primary
    }

    fn build_error(&self, ccx: &ConstCx<'_, 'tcx>, span: Span) -> DiagnosticBuilder<'tcx>;
}

#[derive(Debug)]
pub struct FloatingPointOp;
impl NonConstOp for FloatingPointOp {
    fn status_in_item(&self, ccx: &ConstCx<'_, '_>) -> Status {
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
impl NonConstOp for FnCallIndirect {
    fn build_error(&self, ccx: &ConstCx<'_, 'tcx>, span: Span) -> DiagnosticBuilder<'tcx> {
        ccx.tcx.sess.struct_span_err(span, "function pointers are not allowed in const fn")
    }
}

/// A function call where the callee is not marked as `const`.
#[derive(Debug)]
pub struct FnCallNonConst(pub DefId);
impl NonConstOp for FnCallNonConst {
    fn build_error(&self, ccx: &ConstCx<'_, 'tcx>, span: Span) -> DiagnosticBuilder<'tcx> {
        struct_span_err!(
            ccx.tcx.sess,
            span,
            E0015,
            "calls in {}s are limited to constant functions, \
             tuple structs and tuple variants",
            ccx.const_kind(),
        )
    }
}

/// A call to a `#[unstable]` const fn or `#[rustc_const_unstable]` function.
///
/// Contains the name of the feature that would allow the use of this function.
#[derive(Debug)]
pub struct FnCallUnstable(pub DefId, pub Option<Symbol>);

impl NonConstOp for FnCallUnstable {
    fn build_error(&self, ccx: &ConstCx<'_, 'tcx>, span: Span) -> DiagnosticBuilder<'tcx> {
        let FnCallUnstable(def_id, feature) = *self;

        let mut err = ccx.tcx.sess.struct_span_err(
            span,
            &format!("`{}` is not yet stable as a const fn", ccx.tcx.def_path_str(def_id)),
        );

        if ccx.is_const_stable_const_fn() {
            err.help("Const-stable functions can only call other const-stable functions");
        } else if nightly_options::is_nightly_build() {
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
impl NonConstOp for FnPtrCast {
    fn status_in_item(&self, ccx: &ConstCx<'_, '_>) -> Status {
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
impl NonConstOp for Generator {
    fn status_in_item(&self, _: &ConstCx<'_, '_>) -> Status {
        Status::Forbidden
    }

    fn build_error(&self, ccx: &ConstCx<'_, 'tcx>, span: Span) -> DiagnosticBuilder<'tcx> {
        let msg = format!("{}s are not allowed in {}s", self.0, ccx.const_kind());
        ccx.tcx.sess.struct_span_err(span, &msg)
    }
}

#[derive(Debug)]
pub struct HeapAllocation;
impl NonConstOp for HeapAllocation {
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
impl NonConstOp for InlineAsm {
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
impl NonConstOp for LiveDrop {
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
pub struct CellBorrow;
impl NonConstOp for CellBorrow {
    fn build_error(&self, ccx: &ConstCx<'_, 'tcx>, span: Span) -> DiagnosticBuilder<'tcx> {
        struct_span_err!(
            ccx.tcx.sess,
            span,
            E0492,
            "cannot borrow a constant which may contain \
            interior mutability, create a static instead"
        )
    }
}

#[derive(Debug)]
pub struct MutBorrow(pub hir::BorrowKind);

impl NonConstOp for MutBorrow {
    fn status_in_item(&self, ccx: &ConstCx<'_, '_>) -> Status {
        // Forbid everywhere except in const fn with a feature gate
        if ccx.const_kind() == hir::ConstContext::ConstFn {
            Status::Unstable(sym::const_mut_refs)
        } else {
            Status::Forbidden
        }
    }

    fn build_error(&self, ccx: &ConstCx<'_, 'tcx>, span: Span) -> DiagnosticBuilder<'tcx> {
        let raw = match self.0 {
            hir::BorrowKind::Raw => "raw ",
            hir::BorrowKind::Ref => "",
        };

        let mut err = if ccx.const_kind() == hir::ConstContext::ConstFn {
            feature_err(
                &ccx.tcx.sess.parse_sess,
                sym::const_mut_refs,
                span,
                &format!("{}mutable references are not allowed in {}s", raw, ccx.const_kind()),
            )
        } else {
            let mut err = struct_span_err!(
                ccx.tcx.sess,
                span,
                E0764,
                "{}mutable references are not allowed in {}s",
                raw,
                ccx.const_kind(),
            );
            err.span_label(span, format!("`&{}mut` is only allowed in `const fn`", raw));
            err
        };
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
pub struct MutDeref;
impl NonConstOp for MutDeref {
    fn status_in_item(&self, _: &ConstCx<'_, '_>) -> Status {
        Status::Unstable(sym::const_mut_refs)
    }

    fn importance(&self) -> DiagnosticImportance {
        // Usually a side-effect of a `MutBorrow` somewhere.
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

#[derive(Debug)]
pub struct Panic;
impl NonConstOp for Panic {
    fn status_in_item(&self, _: &ConstCx<'_, '_>) -> Status {
        Status::Unstable(sym::const_panic)
    }

    fn build_error(&self, ccx: &ConstCx<'_, 'tcx>, span: Span) -> DiagnosticBuilder<'tcx> {
        feature_err(
            &ccx.tcx.sess.parse_sess,
            sym::const_panic,
            span,
            &format!("panicking in {}s is unstable", ccx.const_kind()),
        )
    }
}

#[derive(Debug)]
pub struct RawPtrComparison;
impl NonConstOp for RawPtrComparison {
    fn build_error(&self, ccx: &ConstCx<'_, 'tcx>, span: Span) -> DiagnosticBuilder<'tcx> {
        let mut err = ccx
            .tcx
            .sess
            .struct_span_err(span, "pointers cannot be reliably compared during const eval.");
        err.note(
            "see issue #53020 <https://github.com/rust-lang/rust/issues/53020> \
            for more information",
        );
        err
    }
}

#[derive(Debug)]
pub struct RawPtrDeref;
impl NonConstOp for RawPtrDeref {
    fn status_in_item(&self, _: &ConstCx<'_, '_>) -> Status {
        Status::Unstable(sym::const_raw_ptr_deref)
    }

    fn build_error(&self, ccx: &ConstCx<'_, 'tcx>, span: Span) -> DiagnosticBuilder<'tcx> {
        feature_err(
            &ccx.tcx.sess.parse_sess,
            sym::const_raw_ptr_deref,
            span,
            &format!("dereferencing raw pointers in {}s is unstable", ccx.const_kind(),),
        )
    }
}

#[derive(Debug)]
pub struct RawPtrToIntCast;
impl NonConstOp for RawPtrToIntCast {
    fn status_in_item(&self, _: &ConstCx<'_, '_>) -> Status {
        Status::Unstable(sym::const_raw_ptr_to_usize_cast)
    }

    fn build_error(&self, ccx: &ConstCx<'_, 'tcx>, span: Span) -> DiagnosticBuilder<'tcx> {
        feature_err(
            &ccx.tcx.sess.parse_sess,
            sym::const_raw_ptr_to_usize_cast,
            span,
            &format!("casting pointers to integers in {}s is unstable", ccx.const_kind(),),
        )
    }
}

/// An access to a (non-thread-local) `static`.
#[derive(Debug)]
pub struct StaticAccess;
impl NonConstOp for StaticAccess {
    fn status_in_item(&self, ccx: &ConstCx<'_, '_>) -> Status {
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
impl NonConstOp for ThreadLocalAccess {
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

#[derive(Debug)]
pub struct Transmute;
impl NonConstOp for Transmute {
    fn status_in_item(&self, ccx: &ConstCx<'_, '_>) -> Status {
        if ccx.const_kind() != hir::ConstContext::ConstFn {
            Status::Allowed
        } else {
            Status::Unstable(sym::const_fn_transmute)
        }
    }

    fn build_error(&self, ccx: &ConstCx<'_, 'tcx>, span: Span) -> DiagnosticBuilder<'tcx> {
        let mut err = feature_err(
            &ccx.tcx.sess.parse_sess,
            sym::const_fn_transmute,
            span,
            &format!("`transmute` is not allowed in {}s", ccx.const_kind()),
        );
        err.note("`transmute` is only allowed in constants and statics for now");
        err
    }
}

#[derive(Debug)]
pub struct UnionAccess;
impl NonConstOp for UnionAccess {
    fn status_in_item(&self, ccx: &ConstCx<'_, '_>) -> Status {
        // Union accesses are stable in all contexts except `const fn`.
        if ccx.const_kind() != hir::ConstContext::ConstFn {
            Status::Allowed
        } else {
            Status::Unstable(sym::const_fn_union)
        }
    }

    fn build_error(&self, ccx: &ConstCx<'_, 'tcx>, span: Span) -> DiagnosticBuilder<'tcx> {
        feature_err(
            &ccx.tcx.sess.parse_sess,
            sym::const_fn_union,
            span,
            "unions in const fn are unstable",
        )
    }
}

/// See [#64992].
///
/// [#64992]: https://github.com/rust-lang/rust/issues/64992
#[derive(Debug)]
pub struct UnsizingCast;
impl NonConstOp for UnsizingCast {
    fn status_in_item(&self, ccx: &ConstCx<'_, '_>) -> Status {
        mcf_status_in_item(ccx)
    }

    fn build_error(&self, ccx: &ConstCx<'_, 'tcx>, span: Span) -> DiagnosticBuilder<'tcx> {
        mcf_build_error(
            ccx,
            span,
            "unsizing casts to types besides slices are not allowed in const fn",
        )
    }
}

// Types that cannot appear in the signature or locals of a `const fn`.
pub mod ty {
    use super::*;

    #[derive(Debug)]
    pub struct MutRef(pub mir::LocalKind);
    impl NonConstOp for MutRef {
        fn status_in_item(&self, _ccx: &ConstCx<'_, '_>) -> Status {
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
    impl NonConstOp for FnPtr {
        fn importance(&self) -> DiagnosticImportance {
            match self.0 {
                mir::LocalKind::Var | mir::LocalKind::Temp => DiagnosticImportance::Secondary,
                mir::LocalKind::ReturnPointer | mir::LocalKind::Arg => {
                    DiagnosticImportance::Primary
                }
            }
        }

        fn status_in_item(&self, ccx: &ConstCx<'_, '_>) -> Status {
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
    impl NonConstOp for ImplTrait {
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
    impl NonConstOp for TraitBound {
        fn importance(&self) -> DiagnosticImportance {
            match self.0 {
                mir::LocalKind::Var | mir::LocalKind::Temp => DiagnosticImportance::Secondary,
                mir::LocalKind::ReturnPointer | mir::LocalKind::Arg => {
                    DiagnosticImportance::Primary
                }
            }
        }

        fn status_in_item(&self, ccx: &ConstCx<'_, '_>) -> Status {
            mcf_status_in_item(ccx)
        }

        fn build_error(&self, ccx: &ConstCx<'_, 'tcx>, span: Span) -> DiagnosticBuilder<'tcx> {
            mcf_build_error(
                ccx,
                span,
                "trait bounds other than `Sized` on const fn parameters are unstable",
            )
        }
    }

    /// A trait bound with the `?const Trait` opt-out
    #[derive(Debug)]
    pub struct TraitBoundNotConst;
    impl NonConstOp for TraitBoundNotConst {
        fn status_in_item(&self, _: &ConstCx<'_, '_>) -> Status {
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

fn mcf_status_in_item(ccx: &ConstCx<'_, '_>) -> Status {
    if ccx.const_kind() != hir::ConstContext::ConstFn {
        Status::Allowed
    } else {
        Status::Unstable(sym::const_fn)
    }
}

fn mcf_build_error(ccx: &ConstCx<'_, 'tcx>, span: Span, msg: &str) -> DiagnosticBuilder<'tcx> {
    let mut err = struct_span_err!(ccx.tcx.sess, span, E0723, "{}", msg);
    err.note(
        "see issue #57563 <https://github.com/rust-lang/rust/issues/57563> \
             for more information",
    );
    err.help("add `#![feature(const_fn)]` to the crate attributes to enable");
    err
}
