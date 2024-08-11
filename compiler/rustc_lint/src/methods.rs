use rustc_hir::{Expr, ExprKind, LangItem};
use rustc_middle::ty::Ty;
use rustc_session::{declare_lint, declare_lint_pass};
use rustc_span::symbol::{Ident, sym};

use crate::lints::TemporaryAsPtr;
use crate::{LateContext, LateLintPass, LintContext};

declare_lint! {
    /// The `temporary_cstring_as_ptr` lint detects getting the inner pointer of
    /// a temporary `CString`.
    ///
    /// ### Example
    ///
    /// ```rust
    /// # #![allow(unused)]
    /// # use std::ffi::CString;
    /// let c_str = CString::new("foo").unwrap().as_ptr();
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// The inner pointer of a `CString` lives only as long as the `CString` it
    /// points to. Getting the inner pointer of a *temporary* `CString` allows the `CString`
    /// to be dropped at the end of the statement, as it is not being referenced as far as the
    /// typesystem is concerned. This means outside of the statement the pointer will point to
    /// freed memory, which causes undefined behavior if the pointer is later dereferenced.
    pub TEMPORARY_CSTRING_AS_PTR,
    Warn,
    "detects getting the inner pointer of a temporary `CString`"
}

declare_lint! {
    /// TODO
    pub TEMPORARY_AS_PTR,
    Warn,
    "detects getting the inner pointer of a temporary container"
}

declare_lint_pass!(TemporaryCStringAsPtr => [TEMPORARY_CSTRING_AS_PTR, TEMPORARY_AS_PTR]);

impl<'tcx> LateLintPass<'tcx> for TemporaryCStringAsPtr {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        // We have a method call.
        let ExprKind::MethodCall(method, receiver, _args, _span) = expr.kind else {
            return;
        };
        let Ident { name: method_name, span: method_span } = method.ident;
        tracing::debug!(?method);

        // The method is `.as_ptr()` or `.as_mut_ptr`.
        if method_name != sym::as_ptr && method_name != sym::as_mut_ptr {
            return;
        }

        // It is called on a temporary rvalue.
        let is_temp = is_temporary_rvalue(receiver);
        tracing::debug!(receiver = ?receiver.kind, ?is_temp);
        if !is_temp {
            return;
        }

        // The temporary value's type is array, box, Vec, String, or CString
        let ty = cx.typeck_results().expr_ty(receiver);
        tracing::debug!(?ty);
        let Some(is_cstring) = as_container(cx, ty) else {
            return;
        };

        let span = method.ident.span;
        let decorator = TemporaryAsPtr {
            method: method_name,
            ty: ty.to_string(),
            as_ptr_span: method_span,
            temporary_span: receiver.span,
        };

        if is_cstring {
            cx.emit_span_lint(TEMPORARY_CSTRING_AS_PTR, span, decorator);
        } else {
            cx.emit_span_lint(TEMPORARY_AS_PTR, span, decorator);
        };
    }
}

fn is_temporary_rvalue(expr: &Expr<'_>) -> bool {
    match expr.kind {
        // We are not interested in these
        ExprKind::Cast(_, _) | ExprKind::Closure(_) | ExprKind::Tup(_) | ExprKind::Lit(_) => false,

        // Const is not temporary.
        ExprKind::ConstBlock(_) | ExprKind::Repeat(_, _) => false,

        // This is literally lvalue.
        ExprKind::Path(_) => false,

        // Calls return rvalues.
        ExprKind::Call(_, _)
        | ExprKind::MethodCall(_, _, _, _)
        | ExprKind::Index(_, _, _)
        | ExprKind::Binary(_, _, _) => true,

        // This is likely a dereference.
        ExprKind::Unary(_, _) => false,

        // Inner blocks are rvalues.
        ExprKind::If(_, _, _)
        | ExprKind::Loop(_, _, _, _)
        | ExprKind::Match(_, _, _)
        | ExprKind::Block(_, _) => true,

        ExprKind::DropTemps(inner) => is_temporary_rvalue(inner),
        ExprKind::Field(parent, _) => is_temporary_rvalue(parent),

        ExprKind::Struct(_, _, _) => true,
        // False negatives are possible, but arrays get promoted to 'static way too often.
        ExprKind::Array(_) => false,

        // These typecheck to `!`
        ExprKind::Break(_, _) | ExprKind::Continue(_) | ExprKind::Ret(_) | ExprKind::Become(_) => {
            false
        }

        // These typecheck to `()`
        ExprKind::Assign(_, _, _) | ExprKind::AssignOp(_, _, _) | ExprKind::Yield(_, _) => false,

        // Not applicable
        ExprKind::Type(_, _) | ExprKind::Err(_) | ExprKind::Let(_) => false,

        // These are compiler-magic macros
        ExprKind::AddrOf(_, _, _) | ExprKind::OffsetOf(_, _) | ExprKind::InlineAsm(_) => false,
    }
}

// None => not a container
// Some(true) => CString
// Some(false) => String, Vec, box, array
fn as_container(cx: &LateContext<'_>, ty: Ty<'_>) -> Option<bool> {
    if ty.is_array() {
        Some(false)
    } else if let Some(inner) = ty.boxed_ty() {
        // We only care about Box<[..]>, Box<str>, Box<CStr>,
        // or Box<T> iff T is another type we care about
        if inner.is_slice()
            || inner.is_str()
            || inner.ty_adt_def().is_some_and(|def| cx.tcx.is_lang_item(def.did(), LangItem::CStr))
            || as_container(cx, inner).is_some()
        {
            Some(false)
        } else {
            None
        }
    } else if let Some(def) = ty.ty_adt_def() {
        match cx.tcx.get_diagnostic_name(def.did()) {
            Some(sym::cstring_type) => Some(true),
            Some(sym::Vec) => Some(false),
            _ if cx.tcx.is_lang_item(def.did(), LangItem::String) => Some(false),
            _ => None,
        }
    } else {
        None
    }
}
