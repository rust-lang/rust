//! Concrete error types for all operations which may be invalid in a certain const context.

use rustc::mir;
use rustc::session::config::nightly_options;
use rustc::session::parse::feature_err;
use rustc::ty::{Ty, TyCtxt};
use rustc_errors::struct_span_err;
use rustc_hir::def_id::DefId;
use rustc_hir::Constness;
use rustc_span::symbol::sym;
use rustc_span::{Span, Symbol};

use super::{ConstKind, Item};

/// An operation that is not *always* allowed in a const context.
pub trait NonConstOp: std::fmt::Debug {
    /// Whether this operation can be evaluated by miri.
    const IS_SUPPORTED_IN_MIRI: bool = true;

    /// Returns the `Symbol` for the feature gate that would allow this operation, or `None` if
    /// such a feature gate does not exist.
    fn feature_gate() -> Option<Symbol> {
        None
    }

    /// Returns `true` if this operation is allowed in the given item.
    ///
    /// This check should assume that we are not in a non-const `fn`, where all operations are
    /// legal.
    fn is_allowed_in_item(&self, item: &Item<'_, '_>) -> bool {
        Self::feature_gate().map_or(false, |gate| feature_allowed(item, gate))
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
impl NonConstOp for Abort {}

#[derive(Debug)]
pub enum ArithmeticOp {
    Binary(mir::BinOp),
    Unary(mir::UnOp),
}

impl From<mir::BinOp> for ArithmeticOp {
    fn from(op: mir::BinOp) -> Self {
        ArithmeticOp::Binary(op)
    }
}

impl From<mir::UnOp> for ArithmeticOp {
    fn from(op: mir::UnOp) -> Self {
        ArithmeticOp::Unary(op)
    }
}

#[derive(Debug)]
pub struct Arithmetic<'a>(pub ArithmeticOp, pub Ty<'a>);
impl NonConstOp for Arithmetic<'_> {
    fn is_allowed_in_item(&self, item: &Item<'_, '_>) -> bool {
        item.const_kind() != ConstKind::ConstFn || !min_const_fn_checks_enabled(item)
    }

    fn emit_error(&self, item: &Item<'_, '_>, span: Span) {
        let msg =
            format!("operations on `{:?}` are not allowed in a {}", self.1, item.const_kind());
        feature_err(&item.tcx.sess.parse_sess, sym::const_fn, span, &msg).emit()
    }
}

/// A `Downcast` projection.
#[derive(Debug)]
pub struct Downcast;
impl NonConstOp for Downcast {
    fn feature_gate() -> Option<Symbol> {
        Some(sym::const_if_match)
    }
}

/// A function call where the callee is a pointer.
#[derive(Debug)]
pub struct FnCallIndirect;
impl NonConstOp for FnCallIndirect {
    fn emit_error(&self, item: &Item<'_, '_>, span: Span) {
        item.tcx.sess.struct_span_err(span, "indirect function calls are not supported").emit()
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

        let mut err = item.tcx.sess.struct_span_err(
            span,
            &format!("`{}` is not yet stable as a const fn", item.tcx.def_path_str(def_id)),
        );
        if nightly_options::is_nightly_build() {
            err.help(&format!("add `#![feature({})]` to the crate attributes to enable", feature));
        }
        err.emit();
    }
}

#[derive(Debug)]
pub struct Generator;
impl NonConstOp for Generator {}

#[derive(Debug)]
pub struct HeapAllocation;
impl NonConstOp for HeapAllocation {
    const IS_SUPPORTED_IN_MIRI: bool = false;

    fn emit_error(&self, item: &Item<'_, '_>, span: Span) {
        let mut err = struct_span_err!(
            item.tcx.sess,
            span,
            E0010,
            "allocations are not allowed in {}s",
            item.const_kind()
        );
        err.span_label(span, format!("allocation not allowed in {}s", item.const_kind()));
        if item.tcx.sess.teach(&err.get_code().unwrap()) {
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
pub struct IfOrMatch;
impl NonConstOp for IfOrMatch {
    fn feature_gate() -> Option<Symbol> {
        Some(sym::const_if_match)
    }

    fn emit_error(&self, item: &Item<'_, '_>, span: Span) {
        // This should be caught by the HIR const-checker.
        item.tcx.sess.delay_span_bug(span, "complex control flow is forbidden in a const context");
    }
}

#[derive(Debug)]
pub struct InlineAsm;
impl NonConstOp for InlineAsm {}

#[derive(Debug)]
pub struct LiveDrop;
impl NonConstOp for LiveDrop {
    fn emit_error(&self, item: &Item<'_, '_>, span: Span) {
        struct_span_err!(
            item.tcx.sess,
            span,
            E0493,
            "destructors cannot be evaluated at compile-time"
        )
        .span_label(span, format!("{}s cannot evaluate destructors", item.const_kind()))
        .emit();
    }
}

#[derive(Debug)]
pub struct Loop;
impl NonConstOp for Loop {
    fn feature_gate() -> Option<Symbol> {
        Some(sym::const_loop)
    }

    fn emit_error(&self, item: &Item<'_, '_>, span: Span) {
        // This should be caught by the HIR const-checker.
        item.tcx.sess.delay_span_bug(span, "complex control flow is forbidden in a const context");
    }
}

#[derive(Debug)]
pub struct CellBorrow;
impl NonConstOp for CellBorrow {
    fn emit_error(&self, item: &Item<'_, '_>, span: Span) {
        struct_span_err!(
            item.tcx.sess,
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
    fn feature_gate() -> Option<Symbol> {
        Some(sym::const_mut_refs)
    }

    fn emit_error(&self, item: &Item<'_, '_>, span: Span) {
        let mut err = feature_err(
            &item.tcx.sess.parse_sess,
            sym::const_mut_refs,
            span,
            &format!("mutable references in a {} are unstable", item.const_kind()),
        );
        err.span_label(span, format!("{}s require immutable values", item.const_kind()));
        if item.tcx.sess.teach(&err.get_code().unwrap()) {
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

#[derive(Debug)]
pub struct MutDeref;
impl NonConstOp for MutDeref {
    fn feature_gate() -> Option<Symbol> {
        Some(sym::const_mut_refs)
    }
}

#[derive(Debug)]
pub struct Panic;
impl NonConstOp for Panic {
    fn feature_gate() -> Option<Symbol> {
        Some(sym::const_panic)
    }

    fn emit_error(&self, item: &Item<'_, '_>, span: Span) {
        feature_err(
            &item.tcx.sess.parse_sess,
            sym::const_panic,
            span,
            &format!("panicking in {}s is unstable", item.const_kind()),
        )
        .emit();
    }
}

#[derive(Debug)]
pub struct RawPtrComparison;
impl NonConstOp for RawPtrComparison {
    fn feature_gate() -> Option<Symbol> {
        Some(sym::const_compare_raw_pointers)
    }

    fn emit_error(&self, item: &Item<'_, '_>, span: Span) {
        feature_err(
            &item.tcx.sess.parse_sess,
            sym::const_compare_raw_pointers,
            span,
            &format!("comparing raw pointers inside {}", item.const_kind()),
        )
        .emit();
    }
}

#[derive(Debug)]
pub struct RawPtrDeref;
impl NonConstOp for RawPtrDeref {
    fn feature_gate() -> Option<Symbol> {
        Some(sym::const_raw_ptr_deref)
    }

    fn emit_error(&self, item: &Item<'_, '_>, span: Span) {
        feature_err(
            &item.tcx.sess.parse_sess,
            sym::const_raw_ptr_deref,
            span,
            &format!("dereferencing raw pointers in {}s is unstable", item.const_kind()),
        )
        .emit();
    }
}

#[derive(Debug)]
pub struct RawPtrToIntCast;
impl NonConstOp for RawPtrToIntCast {
    fn feature_gate() -> Option<Symbol> {
        Some(sym::const_raw_ptr_to_usize_cast)
    }

    fn emit_error(&self, item: &Item<'_, '_>, span: Span) {
        feature_err(
            &item.tcx.sess.parse_sess,
            sym::const_raw_ptr_to_usize_cast,
            span,
            &format!("casting pointers to integers in {}s is unstable", item.const_kind()),
        )
        .emit();
    }
}

#[derive(Debug)]
pub struct FnPtr;
impl NonConstOp for FnPtr {
    fn is_allowed_in_item(&self, item: &Item<'_, '_>) -> bool {
        item.const_kind() != ConstKind::ConstFn
            || !min_const_fn_checks_enabled(item)
            || item.tcx.has_attr(item.def_id, sym::rustc_allow_const_fn_ptr)
    }

    fn emit_error(&self, item: &Item<'_, '_>, span: Span) {
        feature_err(
            &item.tcx.sess.parse_sess,
            sym::const_fn,
            span,
            "function pointers in const fn are unstable",
        )
        .emit()
    }
}

/// See [#64992].
///
/// [#64992]: https://github.com/rust-lang/rust/issues/64992
#[derive(Debug)]
pub struct UnsizingCast;
impl NonConstOp for UnsizingCast {
    fn is_allowed_in_item(&self, item: &Item<'_, '_>) -> bool {
        item.const_kind() != ConstKind::ConstFn || !min_const_fn_checks_enabled(item)
    }

    fn emit_error(&self, item: &Item<'_, '_>, span: Span) {
        feature_err(
            &item.tcx.sess.parse_sess,
            sym::const_fn,
            span,
            "unsizing casts are not allowed in const fn",
        )
        .emit()
    }
}

#[derive(Debug)]
pub struct ImplTrait;
impl NonConstOp for ImplTrait {
    fn is_allowed_in_item(&self, item: &Item<'_, '_>) -> bool {
        item.const_kind() != ConstKind::ConstFn || !min_const_fn_checks_enabled(item)
    }

    fn emit_error(&self, item: &Item<'_, '_>, span: Span) {
        feature_err(
            &item.tcx.sess.parse_sess,
            sym::const_fn,
            span,
            "`impl Trait` in const fn is unstable",
        )
        .emit()
    }
}

#[derive(Debug)]
pub struct TraitBound(pub Constness);
impl NonConstOp for TraitBound {
    fn is_allowed_in_item(&self, item: &Item<'_, '_>) -> bool {
        if item.const_kind() != ConstKind::ConstFn || !min_const_fn_checks_enabled(item) {
            return true;
        }

        // Allow `T: ?const Trait`
        if self.0 == Constness::NotConst {
            feature_allowed(item, sym::const_trait_bound_opt_out)
        } else {
            false
        }
    }

    fn emit_error(&self, item: &Item<'_, '_>, span: Span) {
        feature_err(
            &item.tcx.sess.parse_sess,
            sym::const_fn,
            span,
            "trait bounds other than `Sized` on const fn parameters are unstable",
        )
        .emit()
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
        let mut err = struct_span_err!(
            item.tcx.sess,
            span,
            E0013,
            "{}s cannot refer to statics",
            item.const_kind()
        );
        err.help(
            "consider extracting the value of the `static` to a `const`, and referring to that",
        );
        if item.tcx.sess.teach(&err.get_code().unwrap()) {
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
    const IS_SUPPORTED_IN_MIRI: bool = false;

    fn emit_error(&self, item: &Item<'_, '_>, span: Span) {
        struct_span_err!(
            item.tcx.sess,
            span,
            E0625,
            "thread-local statics cannot be \
            accessed at compile-time"
        )
        .emit();
    }
}

#[derive(Debug)]
pub struct UnionAccess;
impl NonConstOp for UnionAccess {
    fn is_allowed_in_item(&self, item: &Item<'_, '_>) -> bool {
        // Union accesses are stable in all contexts except `const fn`.
        item.const_kind() != ConstKind::ConstFn || feature_allowed(item, sym::const_fn_union)
    }

    fn feature_gate() -> Option<Symbol> {
        Some(sym::const_fn_union)
    }

    fn emit_error(&self, item: &Item<'_, '_>, span: Span) {
        feature_err(
            &item.tcx.sess.parse_sess,
            sym::const_fn_union,
            span,
            "unions in const fn are unstable",
        )
        .emit();
    }
}

/// Returns `true` if the constness of this item is not stabilized, either because it is declared
/// with `#[rustc_const_unstable]` or because the item itself is unstable.
fn is_declared_const_unstable(tcx: TyCtxt<'tcx>, def_id: DefId) -> bool {
    tcx.lookup_const_stability(def_id).map_or(false, |stab| stab.level.is_unstable())
        || tcx.lookup_stability(def_id).map_or(false, |stab| stab.level.is_unstable())
}

/// Returns `true` if the feature with the given gate is allowed within this const context.
fn feature_allowed(item: &Item<'_, '_>, feature_gate: Symbol) -> bool {
    let Item { tcx, def_id, .. } = *item;

    // All features require that the corresponding gate be enabled,
    // even if the function has `#[allow_internal_unstable(the_gate)]`.
    if !tcx.features().enabled(feature_gate) {
        return false;
    }

    // If this crate is not using stability attributes, or this function is not claiming to be a
    // stable `const fn`, that is all that is required.
    if !tcx.features().staged_api {
        return true;
    }

    if is_declared_const_unstable(item.tcx, item.def_id) {
        return true;
    }

    // However, we cannot allow stable `const fn`s to use unstable features without an explicit
    // opt-in via `allow_internal_unstable`.
    rustc_attr::allow_internal_unstable(&tcx.get_attrs(def_id), &tcx.sess.diagnostic())
        .map_or(false, |mut features| features.any(|name| name == feature_gate))
}

fn min_const_fn_checks_enabled(item: &Item<'_, '_>) -> bool {
    assert_eq!(item.const_kind(), ConstKind::ConstFn);

    if item.tcx.features().staged_api {
        // All functions except for unstable ones need to pass the min const fn checks. This
        // includes private functions that are not marked unstable.
        !is_declared_const_unstable(item.tcx, item.def_id)
    } else {
        // Crates that are not using stability attributes can use `#![feature(const_fn)]` to opt out of
        // the min_const_fn checks.
        !item.tcx.features().const_fn
    }
}
