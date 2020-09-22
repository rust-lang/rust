//! Concrete error types for all operations which may be invalid in a certain const context.

use rustc_errors::{struct_span_err, Applicability};
use rustc_hir as hir;
use rustc_hir::def_id::DefId;
use rustc_session::config::nightly_options;
use rustc_session::parse::feature_err;
use rustc_span::symbol::sym;
use rustc_span::{Span, Symbol};

use super::ConstCx;

/// Emits an error and returns `true` if `op` is not allowed in the given const context.
pub fn non_const<O: NonConstOp>(ccx: &ConstCx<'_, '_>, op: O, span: Span) -> bool {
    debug!("illegal_op: op={:?}", op);

    let gate = match op.status_in_item(ccx) {
        Status::Allowed => return false,

        Status::Unstable(gate) if ccx.tcx.features().enabled(gate) => {
            let unstable_in_stable = ccx.is_const_stable_const_fn()
                && !super::allow_internal_unstable(ccx.tcx, ccx.def_id.to_def_id(), gate);

            if unstable_in_stable {
                ccx.tcx.sess
                    .struct_span_err(
                        span,
                        &format!("const-stable function cannot use `#[feature({})]`", gate.as_str()),
                    )
                    .span_suggestion(
                        ccx.body.span,
                        "if it is not part of the public API, make this function unstably const",
                        concat!(r#"#[rustc_const_unstable(feature = "...", issue = "...")]"#, '\n').to_owned(),
                        Applicability::HasPlaceholders,
                    )
                    .note("otherwise `#[allow_internal_unstable]` can be used to bypass stability checks")
                    .emit();
            }

            return unstable_in_stable;
        }

        Status::Unstable(gate) => Some(gate),
        Status::Forbidden => None,
    };

    if ccx.tcx.sess.opts.debugging_opts.unleash_the_miri_inside_of_you {
        ccx.tcx.sess.miri_unleashed_feature(span, gate);
        return false;
    }

    op.emit_error(ccx, span);
    true
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    Allowed,
    Unstable(Symbol),
    Forbidden,
}

/// An operation that is not *always* allowed in a const context.
pub trait NonConstOp: std::fmt::Debug {
    const STOPS_CONST_CHECKING: bool = false;

    /// Returns an enum indicating whether this operation is allowed within the given item.
    fn status_in_item(&self, _ccx: &ConstCx<'_, '_>) -> Status {
        Status::Forbidden
    }

    fn emit_error(&self, ccx: &ConstCx<'_, '_>, span: Span) {
        let mut err = struct_span_err!(
            ccx.tcx.sess,
            span,
            E0019,
            "{} contains unimplemented expression type",
            ccx.const_kind()
        );

        if let Status::Unstable(gate) = self.status_in_item(ccx) {
            if !ccx.tcx.features().enabled(gate) && nightly_options::is_nightly_build() {
                err.help(&format!("add `#![feature({})]` to the crate attributes to enable", gate));
            }
        }

        if ccx.tcx.sess.teach(&err.get_code().unwrap()) {
            err.note(
                "A function call isn't allowed in the const's initialization expression \
                      because the expression's value must be known at compile-time.",
            );
            err.note(
                "Remember: you can't use a function call inside a const's initialization \
                      expression! However, you can use it anywhere else.",
            );
        }
        err.emit();
    }
}

#[derive(Debug)]
pub struct Abort;
impl NonConstOp for Abort {
    const STOPS_CONST_CHECKING: bool = true;

    fn status_in_item(&self, ccx: &ConstCx<'_, '_>) -> Status {
        mcf_status_in_item(ccx)
    }

    fn emit_error(&self, ccx: &ConstCx<'_, '_>, span: Span) {
        mcf_emit_error(ccx, span, "abort is not stable in const fn")
    }
}

#[derive(Debug)]
pub struct NonPrimitiveOp;
impl NonConstOp for NonPrimitiveOp {
    const STOPS_CONST_CHECKING: bool = true;

    fn status_in_item(&self, ccx: &ConstCx<'_, '_>) -> Status {
        mcf_status_in_item(ccx)
    }

    fn emit_error(&self, ccx: &ConstCx<'_, '_>, span: Span) {
        mcf_emit_error(ccx, span, "only int, `bool` and `char` operations are stable in const fn")
    }
}

/// A function call where the callee is a pointer.
#[derive(Debug)]
pub struct FnCallIndirect;
impl NonConstOp for FnCallIndirect {
    fn emit_error(&self, ccx: &ConstCx<'_, '_>, span: Span) {
        let mut err =
            ccx.tcx.sess.struct_span_err(span, "function pointers are not allowed in const fn");
        err.emit();
    }
}

/// A function call where the callee is not marked as `const`.
#[derive(Debug)]
pub struct FnCallNonConst(pub DefId);
impl NonConstOp for FnCallNonConst {
    fn emit_error(&self, ccx: &ConstCx<'_, '_>, span: Span) {
        let mut err = struct_span_err!(
            ccx.tcx.sess,
            span,
            E0015,
            "calls in {}s are limited to constant functions, \
             tuple structs and tuple variants",
            ccx.const_kind(),
        );
        err.emit();
    }
}

/// A call to a `#[unstable]` const fn or `#[rustc_const_unstable]` function.
///
/// Contains the name of the feature that would allow the use of this function.
#[derive(Debug)]
pub struct FnCallUnstable(pub DefId, pub Option<Symbol>);

impl NonConstOp for FnCallUnstable {
    fn emit_error(&self, ccx: &ConstCx<'_, '_>, span: Span) {
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
        err.emit();
    }
}

#[derive(Debug)]
pub struct FnPtrCast;
impl NonConstOp for FnPtrCast {
    const STOPS_CONST_CHECKING: bool = true;

    fn status_in_item(&self, ccx: &ConstCx<'_, '_>) -> Status {
        mcf_status_in_item(ccx)
    }

    fn emit_error(&self, ccx: &ConstCx<'_, '_>, span: Span) {
        mcf_emit_error(ccx, span, "function pointer casts are not allowed in const fn");
    }
}

#[derive(Debug)]
pub struct Generator;
impl NonConstOp for Generator {
    const STOPS_CONST_CHECKING: bool = true;

    fn status_in_item(&self, ccx: &ConstCx<'_, '_>) -> Status {
        // FIXME: This means generator-only MIR is only forbidden in const fn. This is for
        // compatibility with the old code. Such MIR should be forbidden everywhere.
        mcf_status_in_item(ccx)
    }

    fn emit_error(&self, ccx: &ConstCx<'_, '_>, span: Span) {
        mcf_emit_error(ccx, span, "const fn generators are unstable");
    }
}

#[derive(Debug)]
pub struct HeapAllocation;
impl NonConstOp for HeapAllocation {
    fn emit_error(&self, ccx: &ConstCx<'_, '_>, span: Span) {
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
        err.emit();
    }
}

#[derive(Debug)]
pub struct InlineAsm;
impl NonConstOp for InlineAsm {}

#[derive(Debug)]
pub struct LiveDrop {
    pub dropped_at: Option<Span>,
}
impl NonConstOp for LiveDrop {
    fn emit_error(&self, ccx: &ConstCx<'_, '_>, span: Span) {
        let mut diagnostic = struct_span_err!(
            ccx.tcx.sess,
            span,
            E0493,
            "destructors cannot be evaluated at compile-time"
        );
        diagnostic.span_label(span, format!("{}s cannot evaluate destructors", ccx.const_kind()));
        if let Some(span) = self.dropped_at {
            diagnostic.span_label(span, "value is dropped here");
        }
        diagnostic.emit();
    }
}

#[derive(Debug)]
pub struct CellBorrow;
impl NonConstOp for CellBorrow {
    fn emit_error(&self, ccx: &ConstCx<'_, '_>, span: Span) {
        struct_span_err!(
            ccx.tcx.sess,
            span,
            E0492,
            "cannot borrow a constant which may contain \
            interior mutability, create a static instead"
        )
        .emit();
    }
}

#[derive(Debug)]
pub struct MutBorrow;
impl NonConstOp for MutBorrow {
    fn status_in_item(&self, ccx: &ConstCx<'_, '_>) -> Status {
        // Forbid everywhere except in const fn with a feature gate
        if ccx.const_kind() == hir::ConstContext::ConstFn {
            Status::Unstable(sym::const_mut_refs)
        } else {
            Status::Forbidden
        }
    }

    fn emit_error(&self, ccx: &ConstCx<'_, '_>, span: Span) {
        let mut err = if ccx.const_kind() == hir::ConstContext::ConstFn {
            feature_err(
                &ccx.tcx.sess.parse_sess,
                sym::const_mut_refs,
                span,
                &format!("mutable references are not allowed in {}s", ccx.const_kind()),
            )
        } else {
            let mut err = struct_span_err!(
                ccx.tcx.sess,
                span,
                E0764,
                "mutable references are not allowed in {}s",
                ccx.const_kind(),
            );
            err.span_label(span, format!("`&mut` is only allowed in `const fn`"));
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
        err.emit();
    }
}

// FIXME(ecstaticmorse): Unify this with `MutBorrow`. It has basically the same issues.
#[derive(Debug)]
pub struct MutAddressOf;
impl NonConstOp for MutAddressOf {
    fn status_in_item(&self, ccx: &ConstCx<'_, '_>) -> Status {
        // Forbid everywhere except in const fn with a feature gate
        if ccx.const_kind() == hir::ConstContext::ConstFn {
            Status::Unstable(sym::const_mut_refs)
        } else {
            Status::Forbidden
        }
    }

    fn emit_error(&self, ccx: &ConstCx<'_, '_>, span: Span) {
        feature_err(
            &ccx.tcx.sess.parse_sess,
            sym::const_mut_refs,
            span,
            &format!("`&raw mut` is not allowed in {}s", ccx.const_kind()),
        )
        .emit();
    }
}

#[derive(Debug)]
pub struct MutDeref;
impl NonConstOp for MutDeref {
    fn status_in_item(&self, _: &ConstCx<'_, '_>) -> Status {
        Status::Unstable(sym::const_mut_refs)
    }
}

#[derive(Debug)]
pub struct Panic;
impl NonConstOp for Panic {
    fn status_in_item(&self, _: &ConstCx<'_, '_>) -> Status {
        Status::Unstable(sym::const_panic)
    }

    fn emit_error(&self, ccx: &ConstCx<'_, '_>, span: Span) {
        feature_err(
            &ccx.tcx.sess.parse_sess,
            sym::const_panic,
            span,
            &format!("panicking in {}s is unstable", ccx.const_kind()),
        )
        .emit();
    }
}

#[derive(Debug)]
pub struct RawPtrComparison;
impl NonConstOp for RawPtrComparison {
    fn emit_error(&self, ccx: &ConstCx<'_, '_>, span: Span) {
        let mut err = ccx
            .tcx
            .sess
            .struct_span_err(span, "pointers cannot be reliably compared during const eval.");
        err.note(
            "see issue #53020 <https://github.com/rust-lang/rust/issues/53020> \
            for more information",
        );
        err.emit();
    }
}

#[derive(Debug)]
pub struct RawPtrDeref;
impl NonConstOp for RawPtrDeref {
    fn status_in_item(&self, _: &ConstCx<'_, '_>) -> Status {
        Status::Unstable(sym::const_raw_ptr_deref)
    }

    fn emit_error(&self, ccx: &ConstCx<'_, '_>, span: Span) {
        feature_err(
            &ccx.tcx.sess.parse_sess,
            sym::const_raw_ptr_deref,
            span,
            &format!("dereferencing raw pointers in {}s is unstable", ccx.const_kind(),),
        )
        .emit();
    }
}

#[derive(Debug)]
pub struct RawPtrToIntCast;
impl NonConstOp for RawPtrToIntCast {
    fn status_in_item(&self, _: &ConstCx<'_, '_>) -> Status {
        Status::Unstable(sym::const_raw_ptr_to_usize_cast)
    }

    fn emit_error(&self, ccx: &ConstCx<'_, '_>, span: Span) {
        feature_err(
            &ccx.tcx.sess.parse_sess,
            sym::const_raw_ptr_to_usize_cast,
            span,
            &format!("casting pointers to integers in {}s is unstable", ccx.const_kind(),),
        )
        .emit();
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

    fn emit_error(&self, ccx: &ConstCx<'_, '_>, span: Span) {
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
        err.emit();
    }
}

/// An access to a thread-local `static`.
#[derive(Debug)]
pub struct ThreadLocalAccess;
impl NonConstOp for ThreadLocalAccess {
    fn emit_error(&self, ccx: &ConstCx<'_, '_>, span: Span) {
        struct_span_err!(
            ccx.tcx.sess,
            span,
            E0625,
            "thread-local statics cannot be \
            accessed at compile-time"
        )
        .emit();
    }
}

#[derive(Debug)]
pub struct Transmute;
impl NonConstOp for Transmute {
    const STOPS_CONST_CHECKING: bool = true;

    fn status_in_item(&self, ccx: &ConstCx<'_, '_>) -> Status {
        if ccx.const_kind() != hir::ConstContext::ConstFn {
            Status::Allowed
        } else {
            Status::Unstable(sym::const_fn_transmute)
        }
    }

    fn emit_error(&self, ccx: &ConstCx<'_, '_>, span: Span) {
        mcf_emit_error(ccx, span, "can only call `transmute` from const items, not `const fn`");
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

    fn emit_error(&self, ccx: &ConstCx<'_, '_>, span: Span) {
        feature_err(
            &ccx.tcx.sess.parse_sess,
            sym::const_fn_union,
            span,
            "unions in const fn are unstable",
        )
        .emit();
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

    fn emit_error(&self, ccx: &ConstCx<'_, '_>, span: Span) {
        mcf_emit_error(
            ccx,
            span,
            "unsizing casts to types besides slices are not allowed in const fn",
        );
    }
}

pub mod ty {
    use super::*;

    #[derive(Debug)]
    pub struct MutRef;
    impl NonConstOp for MutRef {
        const STOPS_CONST_CHECKING: bool = true;

        fn status_in_item(&self, _ccx: &ConstCx<'_, '_>) -> Status {
            Status::Unstable(sym::const_mut_refs)
        }

        fn emit_error(&self, ccx: &ConstCx<'_, '_>, span: Span) {
            mcf_emit_error(ccx, span, "mutable references in const fn are unstable");
        }
    }

    #[derive(Debug)]
    pub struct FnPtr;
    impl NonConstOp for FnPtr {
        const STOPS_CONST_CHECKING: bool = true;

        fn status_in_item(&self, ccx: &ConstCx<'_, '_>) -> Status {
            // FIXME: This attribute a hack to allow the specialization of the `futures` API. See
            // #59739. We should have a proper feature gate for this.
            if ccx.tcx.has_attr(ccx.def_id.to_def_id(), sym::rustc_allow_const_fn_ptr) {
                Status::Allowed
            } else {
                mcf_status_in_item(ccx)
            }
        }

        fn emit_error(&self, ccx: &ConstCx<'_, '_>, span: Span) {
            mcf_emit_error(ccx, span, "function pointers in const fn are unstable");
        }
    }

    #[derive(Debug)]
    pub struct ImplTrait;
    impl NonConstOp for ImplTrait {
        const STOPS_CONST_CHECKING: bool = true;

        fn status_in_item(&self, ccx: &ConstCx<'_, '_>) -> Status {
            mcf_status_in_item(ccx)
        }

        fn emit_error(&self, ccx: &ConstCx<'_, '_>, span: Span) {
            mcf_emit_error(ccx, span, "`impl Trait` in const fn is unstable");
        }
    }

    #[derive(Debug)]
    pub struct TraitBound;
    impl NonConstOp for TraitBound {
        const STOPS_CONST_CHECKING: bool = true;

        fn status_in_item(&self, ccx: &ConstCx<'_, '_>) -> Status {
            mcf_status_in_item(ccx)
        }

        fn emit_error(&self, ccx: &ConstCx<'_, '_>, span: Span) {
            mcf_emit_error(
                ccx,
                span,
                "trait bounds other than `Sized` on const fn parameters are unstable",
            );
        }
    }

    /// A trait bound with the `?const Trait` opt-out
    #[derive(Debug)]
    pub struct TraitBoundNotConst;
    impl NonConstOp for TraitBoundNotConst {
        const STOPS_CONST_CHECKING: bool = true;

        fn status_in_item(&self, _: &ConstCx<'_, '_>) -> Status {
            Status::Unstable(sym::const_trait_bound_opt_out)
        }

        fn emit_error(&self, ccx: &ConstCx<'_, '_>, span: Span) {
            feature_err(
                &ccx.tcx.sess.parse_sess,
                sym::const_trait_bound_opt_out,
                span,
                "`?const Trait` syntax is unstable",
            )
            .emit()
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

fn mcf_emit_error(ccx: &ConstCx<'_, '_>, span: Span, msg: &str) {
    struct_span_err!(ccx.tcx.sess, span, E0723, "{}", msg)
        .note(
            "see issue #57563 <https://github.com/rust-lang/rust/issues/57563> \
             for more information",
        )
        .help("add `#![feature(const_fn)]` to the crate attributes to enable")
        .emit();
}
