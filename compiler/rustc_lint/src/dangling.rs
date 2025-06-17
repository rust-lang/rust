use rustc_ast::visit::{visit_opt, walk_list};
use rustc_attr_data_structures::{AttributeKind, find_attr};
use rustc_hir::def_id::LocalDefId;
use rustc_hir::intravisit::{FnKind, Visitor, walk_expr};
use rustc_hir::{Block, Body, Expr, ExprKind, FnDecl, LangItem};
use rustc_middle::ty::{Ty, TyCtxt};
use rustc_session::{declare_lint, impl_lint_pass};
use rustc_span::{Span, sym};

use crate::lints::DanglingPointersFromTemporaries;
use crate::{LateContext, LateLintPass};

declare_lint! {
    /// The `dangling_pointers_from_temporaries` lint detects getting a pointer to data
    /// of a temporary that will immediately get dropped.
    ///
    /// ### Example
    ///
    /// ```rust
    /// # #![allow(unused)]
    /// # unsafe fn use_data(ptr: *const u8) { }
    /// fn gather_and_use(bytes: impl Iterator<Item = u8>) {
    ///     let x: *const u8 = bytes.collect::<Vec<u8>>().as_ptr();
    ///     unsafe { use_data(x) }
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Getting a pointer from a temporary value will not prolong its lifetime,
    /// which means that the value can be dropped and the allocation freed
    /// while the pointer still exists, making the pointer dangling.
    /// This is not an error (as far as the type system is concerned)
    /// but probably is not what the user intended either.
    ///
    /// If you need stronger guarantees, consider using references instead,
    /// as they are statically verified by the borrow-checker to never dangle.
    pub DANGLING_POINTERS_FROM_TEMPORARIES,
    Warn,
    "detects getting a pointer from a temporary"
}

/// FIXME: false negatives (i.e. the lint is not emitted when it should be)
/// 1. Ways to get a temporary that are not recognized:
///    - `owning_temporary.field`
///    - `owning_temporary[index]`
/// 2. No checks for ref-to-ptr conversions:
///    - `&raw [mut] temporary`
///    - `&temporary as *(const|mut) _`
///    - `ptr::from_ref(&temporary)` and friends
#[derive(Clone, Copy, Default)]
pub(crate) struct DanglingPointers;

impl_lint_pass!(DanglingPointers => [DANGLING_POINTERS_FROM_TEMPORARIES]);

// This skips over const blocks, but they cannot use or return a dangling pointer anyways.
impl<'tcx> LateLintPass<'tcx> for DanglingPointers {
    fn check_fn(
        &mut self,
        cx: &LateContext<'tcx>,
        _: FnKind<'tcx>,
        _: &'tcx FnDecl<'tcx>,
        body: &'tcx Body<'tcx>,
        _: Span,
        _: LocalDefId,
    ) {
        DanglingPointerSearcher { cx, inside_call_args: false }.visit_body(body)
    }
}

/// This produces a dangling pointer:
/// ```ignore (example)
/// let ptr = CString::new("hello").unwrap().as_ptr();
/// foo(ptr)
/// ```
///
/// But this does not:
/// ```ignore (example)
/// foo(CString::new("hello").unwrap().as_ptr())
/// ```
///
/// But this does:
/// ```ignore (example)
/// foo({ let ptr = CString::new("hello").unwrap().as_ptr(); ptr })
/// ```
///
/// So we have to keep track of when we are inside of a function/method call argument.
struct DanglingPointerSearcher<'lcx, 'tcx> {
    cx: &'lcx LateContext<'tcx>,
    /// Keeps track of whether we are inside of function/method call arguments,
    /// where this lint should not be emitted.
    ///
    /// See [the main doc][`Self`] for examples.
    inside_call_args: bool,
}

impl Visitor<'_> for DanglingPointerSearcher<'_, '_> {
    fn visit_expr(&mut self, expr: &Expr<'_>) -> Self::Result {
        if !self.inside_call_args {
            lint_expr(self.cx, expr)
        }
        match expr.kind {
            ExprKind::Call(lhs, args) | ExprKind::MethodCall(_, lhs, args, _) => {
                self.visit_expr(lhs);
                self.with_inside_call_args(true, |this| walk_list!(this, visit_expr, args))
            }
            ExprKind::Block(&Block { stmts, expr, .. }, _) => {
                self.with_inside_call_args(false, |this| walk_list!(this, visit_stmt, stmts));
                visit_opt!(self, visit_expr, expr)
            }
            _ => walk_expr(self, expr),
        }
    }
}

impl DanglingPointerSearcher<'_, '_> {
    fn with_inside_call_args<R>(
        &mut self,
        inside_call_args: bool,
        callback: impl FnOnce(&mut Self) -> R,
    ) -> R {
        let old = core::mem::replace(&mut self.inside_call_args, inside_call_args);
        let result = callback(self);
        self.inside_call_args = old;
        result
    }
}

fn lint_expr(cx: &LateContext<'_>, expr: &Expr<'_>) {
    if let ExprKind::MethodCall(method, receiver, _args, _span) = expr.kind
        && is_temporary_rvalue(receiver)
        && let ty = cx.typeck_results().expr_ty(receiver)
        && owns_allocation(cx.tcx, ty)
        && let Some(fn_id) = cx.typeck_results().type_dependent_def_id(expr.hir_id)
        && find_attr!(cx.tcx.get_all_attrs(fn_id), AttributeKind::AsPtr(_))
    {
        // FIXME: use `emit_node_lint` when `#[primary_span]` is added.
        cx.tcx.emit_node_span_lint(
            DANGLING_POINTERS_FROM_TEMPORARIES,
            expr.hir_id,
            method.ident.span,
            DanglingPointersFromTemporaries {
                callee: method.ident,
                ty,
                ptr_span: method.ident.span,
                temporary_span: receiver.span,
            },
        )
    }
}

fn is_temporary_rvalue(expr: &Expr<'_>) -> bool {
    match expr.kind {
        // Const is not temporary.
        ExprKind::ConstBlock(..) | ExprKind::Repeat(..) | ExprKind::Lit(..) => false,

        // This is literally lvalue.
        ExprKind::Path(..) => false,

        // Calls return rvalues.
        ExprKind::Call(..)
        | ExprKind::MethodCall(..)
        | ExprKind::Use(..)
        | ExprKind::Binary(..) => true,

        // Inner blocks are rvalues.
        ExprKind::If(..) | ExprKind::Loop(..) | ExprKind::Match(..) | ExprKind::Block(..) => true,

        // FIXME: these should probably recurse and typecheck along the way.
        //        Some false negatives are possible for now.
        ExprKind::Index(..) | ExprKind::Field(..) | ExprKind::Unary(..) => false,

        ExprKind::Struct(..) => true,

        // FIXME: this has false negatives, but I do not want to deal with 'static/const promotion just yet.
        ExprKind::Array(..) => false,

        // These typecheck to `!`
        ExprKind::Break(..) | ExprKind::Continue(..) | ExprKind::Ret(..) | ExprKind::Become(..) => {
            false
        }

        // These typecheck to `()`
        ExprKind::Assign(..) | ExprKind::AssignOp(..) | ExprKind::Yield(..) => false,

        // Compiler-magic macros
        ExprKind::AddrOf(..) | ExprKind::OffsetOf(..) | ExprKind::InlineAsm(..) => false,

        // We are not interested in these
        ExprKind::Cast(..)
        | ExprKind::Closure(..)
        | ExprKind::Tup(..)
        | ExprKind::DropTemps(..)
        | ExprKind::Let(..) => false,

        ExprKind::UnsafeBinderCast(..) => false,

        // Not applicable
        ExprKind::Type(..) | ExprKind::Err(..) => false,
    }
}

// Array, Vec, String, CString, MaybeUninit, Cell, Box<[_]>, Box<str>, Box<CStr>, UnsafeCell,
// SyncUnsafeCell, or any of the above in arbitrary many nested Box'es.
fn owns_allocation(tcx: TyCtxt<'_>, ty: Ty<'_>) -> bool {
    if ty.is_array() {
        true
    } else if let Some(inner) = ty.boxed_ty() {
        inner.is_slice()
            || inner.is_str()
            || inner.ty_adt_def().is_some_and(|def| tcx.is_lang_item(def.did(), LangItem::CStr))
            || owns_allocation(tcx, inner)
    } else if let Some(def) = ty.ty_adt_def() {
        for lang_item in [LangItem::String, LangItem::MaybeUninit, LangItem::UnsafeCell] {
            if tcx.is_lang_item(def.did(), lang_item) {
                return true;
            }
        }
        tcx.get_diagnostic_name(def.did()).is_some_and(|name| {
            matches!(name, sym::cstring_type | sym::Vec | sym::Cell | sym::SyncUnsafeCell)
        })
    } else {
        false
    }
}
