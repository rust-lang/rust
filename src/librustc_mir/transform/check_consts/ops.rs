//! Concrete error types for all operations which may be invalid in a certain const context.

use rustc::hir::def_id::DefId;
use rustc::mir::BorrowKind;
use rustc::session::config::nightly_options;
use rustc::ty::TyCtxt;
use syntax::feature_gate::{emit_feature_err, GateIssue};
use syntax::symbol::sym;
use syntax_pos::{Span, Symbol};

use super::{ConstKind, Item};

use rustc_error_codes::*;

/// An operation that is not *always* allowed in a const context.
pub trait NonConstOp: std::fmt::Debug {
    /// Whether this operation can be evaluated by miri.
    const IS_SUPPORTED_IN_MIRI: bool = true;

    /// Returns a boolean indicating whether the feature gate that would allow this operation is
    /// enabled, or `None` if such a feature gate does not exist.
    fn feature_gate(_tcx: TyCtxt<'tcx>) -> Option<bool> {
        None
    }

    /// Returns `true` if this operation is allowed in the given item.
    ///
    /// This check should assume that we are not in a non-const `fn`, where all operations are
    /// legal.
    fn is_allowed_in_item(&self, item: &Item<'_, '_>) -> bool {
        Self::feature_gate(item.tcx).unwrap_or(false)
    }

    fn emit_error(&self, item: &Item<'_, '_>, span: Span) {
        let mut err = struct_span_err!(
            item.tcx.sess,
            span,
            E0019,
            "{} contains unimplemented expression type",
            item.const_kind()
        );
        if item.tcx.sess.teach(&err.get_code().unwrap()) {
            err.note("A function call isn't allowed in the const's initialization expression \
                      because the expression's value must be known at compile-time.");
            err.note("Remember: you can't use a function call inside a const's initialization \
                      expression! However, you can use it anywhere else.");
        }
        err.emit();
    }
}

/// A `Downcast` projection.
#[derive(Debug)]
pub struct Downcast;
impl NonConstOp for Downcast {}

/// A function call where the callee is a pointer.
#[derive(Debug)]
pub struct FnCallIndirect;
impl NonConstOp for FnCallIndirect {
    fn emit_error(&self, item: &Item<'_, '_>, span: Span) {
        let mut err = item.tcx.sess.struct_span_err(
            span,
            &format!("function pointers are not allowed in const fn"));
        err.emit();
    }
}

/// A function call where the callee is not marked as `const`.
#[derive(Debug)]
pub struct FnCallNonConst(pub DefId);
impl NonConstOp for FnCallNonConst {
    fn emit_error(&self, item: &Item<'_, '_>, span: Span) {
        let mut err = struct_span_err!(
            item.tcx.sess,
            span,
            E0015,
            "calls in {}s are limited to constant functions, \
             tuple structs and tuple variants",
            item.const_kind(),
        );
        err.emit();
    }
}

/// A function call where the callee is not a function definition or function pointer, e.g. a
/// closure.
///
/// This can be subdivided in the future to produce a better error message.
#[derive(Debug)]
pub struct FnCallOther;
impl NonConstOp for FnCallOther {
    const IS_SUPPORTED_IN_MIRI: bool = false;
}

/// A call to a `#[unstable]` const fn or `#[rustc_const_unstable]` function.
///
/// Contains the name of the feature that would allow the use of this function.
#[derive(Debug)]
pub struct FnCallUnstable(pub DefId, pub Symbol);
impl NonConstOp for FnCallUnstable {
    fn emit_error(&self, item: &Item<'_, '_>, span: Span) {
        let FnCallUnstable(def_id, feature) = *self;

        let mut err = item.tcx.sess.struct_span_err(span,
            &format!("`{}` is not yet stable as a const fn",
                    item.tcx.def_path_str(def_id)));
        if nightly_options::is_nightly_build() {
            help!(&mut err,
                  "add `#![feature({})]` to the \
                   crate attributes to enable",
                  feature);
        }
        err.emit();
    }
}

#[derive(Debug)]
pub struct HeapAllocation;
impl NonConstOp for HeapAllocation {
    const IS_SUPPORTED_IN_MIRI: bool = false;

    fn emit_error(&self, item: &Item<'_, '_>, span: Span) {
        let mut err = struct_span_err!(item.tcx.sess, span, E0010,
                                       "allocations are not allowed in {}s", item.const_kind());
        err.span_label(span, format!("allocation not allowed in {}s", item.const_kind()));
        if item.tcx.sess.teach(&err.get_code().unwrap()) {
            err.note(
                "The value of statics and constants must be known at compile time, \
                 and they live for the entire lifetime of a program. Creating a boxed \
                 value allocates memory on the heap at runtime, and therefore cannot \
                 be done at compile time."
            );
        }
        err.emit();
    }
}

#[derive(Debug)]
pub struct IfOrMatch;
impl NonConstOp for IfOrMatch {
    fn emit_error(&self, item: &Item<'_, '_>, span: Span) {
        // This should be caught by the HIR const-checker.
        item.tcx.sess.delay_span_bug(
            span,
            "complex control flow is forbidden in a const context",
        );
    }
}

#[derive(Debug)]
pub struct LiveDrop;
impl NonConstOp for LiveDrop {
    fn emit_error(&self, item: &Item<'_, '_>, span: Span) {
        struct_span_err!(item.tcx.sess, span, E0493,
                         "destructors cannot be evaluated at compile-time")
            .span_label(span, format!("{}s cannot evaluate destructors",
                                      item.const_kind()))
            .emit();
    }
}

#[derive(Debug)]
pub struct Loop;
impl NonConstOp for Loop {
    fn emit_error(&self, item: &Item<'_, '_>, span: Span) {
        // This should be caught by the HIR const-checker.
        item.tcx.sess.delay_span_bug(
            span,
            "complex control flow is forbidden in a const context",
        );
    }
}

#[derive(Debug)]
pub struct MutBorrow(pub BorrowKind);
impl NonConstOp for MutBorrow {
    fn emit_error(&self, item: &Item<'_, '_>, span: Span) {
        let kind = self.0;
        if let BorrowKind::Mut { .. } = kind {
            let mut err = struct_span_err!(item.tcx.sess, span, E0017,
                                           "references in {}s may only refer \
                                            to immutable values", item.const_kind());
            err.span_label(span, format!("{}s require immutable values",
                                                item.const_kind()));
            if item.tcx.sess.teach(&err.get_code().unwrap()) {
                err.note("References in statics and constants may only refer \
                          to immutable values.\n\n\
                          Statics are shared everywhere, and if they refer to \
                          mutable data one might violate memory safety since \
                          holding multiple mutable references to shared data \
                          is not allowed.\n\n\
                          If you really want global mutable state, try using \
                          static mut or a global UnsafeCell.");
            }
            err.emit();
        } else {
            span_err!(item.tcx.sess, span, E0492,
                      "cannot borrow a constant which may contain \
                       interior mutability, create a static instead");
        }
    }
}

#[derive(Debug)]
pub struct MutDeref;
impl NonConstOp for MutDeref {}

#[derive(Debug)]
pub struct Panic;
impl NonConstOp for Panic {
    fn feature_gate(tcx: TyCtxt<'_>) -> Option<bool> {
        Some(tcx.features().const_panic)
    }

    fn emit_error(&self, item: &Item<'_, '_>, span: Span) {
        emit_feature_err(
            &item.tcx.sess.parse_sess,
            sym::const_panic,
            span,
            GateIssue::Language,
            &format!("panicking in {}s is unstable", item.const_kind()),
        );
    }
}

#[derive(Debug)]
pub struct RawPtrComparison;
impl NonConstOp for RawPtrComparison {
    fn feature_gate(tcx: TyCtxt<'_>) -> Option<bool> {
        Some(tcx.features().const_compare_raw_pointers)
    }

    fn emit_error(&self, item: &Item<'_, '_>, span: Span) {
        emit_feature_err(
            &item.tcx.sess.parse_sess,
            sym::const_compare_raw_pointers,
            span,
            GateIssue::Language,
            &format!("comparing raw pointers inside {}", item.const_kind()),
        );
    }
}

#[derive(Debug)]
pub struct RawPtrDeref;
impl NonConstOp for RawPtrDeref {
    fn feature_gate(tcx: TyCtxt<'_>) -> Option<bool> {
        Some(tcx.features().const_raw_ptr_deref)
    }

    fn emit_error(&self, item: &Item<'_, '_>, span: Span) {
        emit_feature_err(
            &item.tcx.sess.parse_sess, sym::const_raw_ptr_deref,
            span, GateIssue::Language,
            &format!(
                "dereferencing raw pointers in {}s is unstable",
                item.const_kind(),
            ),
        );
    }
}

#[derive(Debug)]
pub struct RawPtrToIntCast;
impl NonConstOp for RawPtrToIntCast {
    fn feature_gate(tcx: TyCtxt<'_>) -> Option<bool> {
        Some(tcx.features().const_raw_ptr_to_usize_cast)
    }

    fn emit_error(&self, item: &Item<'_, '_>, span: Span) {
        emit_feature_err(
            &item.tcx.sess.parse_sess, sym::const_raw_ptr_to_usize_cast,
            span, GateIssue::Language,
            &format!(
                "casting pointers to integers in {}s is unstable",
                item.const_kind(),
            ),
        );
    }
}

/// An access to a (non-thread-local) `static`.
#[derive(Debug)]
pub struct StaticAccess;
impl NonConstOp for StaticAccess {
    fn is_allowed_in_item(&self, item: &Item<'_, '_>) -> bool {
        item.const_kind().is_static()
    }

    fn emit_error(&self, item: &Item<'_, '_>, span: Span) {
        let mut err = struct_span_err!(item.tcx.sess, span, E0013,
                                        "{}s cannot refer to statics, use \
                                        a constant instead", item.const_kind());
        if item.tcx.sess.teach(&err.get_code().unwrap()) {
            err.note(
                "Static and const variables can refer to other const variables. \
                    But a const variable cannot refer to a static variable."
            );
            err.help(
                "To fix this, the value can be extracted as a const and then used."
            );
        }
        err.emit();
    }
}

/// An access to a thread-local `static`.
#[derive(Debug)]
pub struct ThreadLocalAccess;
impl NonConstOp for ThreadLocalAccess {
    const IS_SUPPORTED_IN_MIRI: bool = false;

    fn emit_error(&self, item: &Item<'_, '_>, span: Span) {
        span_err!(item.tcx.sess, span, E0625,
            "thread-local statics cannot be \
            accessed at compile-time");
    }
}

#[derive(Debug)]
pub struct Transmute;
impl NonConstOp for Transmute {
    fn feature_gate(tcx: TyCtxt<'_>) -> Option<bool> {
        Some(tcx.features().const_transmute)
    }

    fn emit_error(&self, item: &Item<'_, '_>, span: Span) {
        emit_feature_err(
            &item.tcx.sess.parse_sess, sym::const_transmute,
            span, GateIssue::Language,
            &format!("The use of std::mem::transmute() \
            is gated in {}s", item.const_kind()));
    }
}

#[derive(Debug)]
pub struct UnionAccess;
impl NonConstOp for UnionAccess {
    fn is_allowed_in_item(&self, item: &Item<'_, '_>) -> bool {
        // Union accesses are stable in all contexts except `const fn`.
        item.const_kind() != ConstKind::ConstFn || Self::feature_gate(item.tcx).unwrap()
    }

    fn feature_gate(tcx: TyCtxt<'_>) -> Option<bool> {
        Some(tcx.features().const_fn_union)
    }

    fn emit_error(&self, item: &Item<'_, '_>, span: Span) {
        emit_feature_err(
            &item.tcx.sess.parse_sess, sym::const_fn_union,
            span, GateIssue::Language,
            "unions in const fn are unstable",
        );
    }
}
