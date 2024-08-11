use rustc_hir::{Expr, ExprKind, LangItem};
use rustc_middle::ty;
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
    "TODO"
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
        tracing::debug!(?receiver, ?is_temp);
        if !is_temp {
            return;
        }

        // The temporary value's type is array, box, Vec, String, or CString
        let ty = cx.typeck_results().expr_ty(receiver);
        tracing::debug!(?ty);

        // None => not a container
        // Some(true) => CString
        // Some(false) => String, Vec, box, array
        let lint_is_cstring = match ty.kind() {
            ty::Array(_, _) => Some(false),
            ty::Adt(def, _) if def.is_box() => Some(false),
            ty::Adt(def, _) if cx.tcx.lang_items().get(LangItem::String) == Some(def.did()) => {
                Some(false)
            }
            ty::Adt(def, _) => match cx.tcx.get_diagnostic_name(def.did()) {
                Some(sym::Vec) => Some(false),
                Some(sym::cstring_type) => Some(true),
                _ => None,
            },
            _ => None,
        };
        tracing::debug!(?lint_is_cstring);
        let Some(is_cstring) = lint_is_cstring else {
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
        ExprKind::Cast(_, _) | ExprKind::Closure(_) | ExprKind::Tup(_) => false,

        // Const is not temporary.
        ExprKind::ConstBlock(_) => false,

        // This is literally lvalue.
        ExprKind::Path(_) => false,

        // Calls return rvalues.
        ExprKind::Call(_, _)
        | ExprKind::MethodCall(_, _, _, _)
        | ExprKind::Index(_, _, _)
        | ExprKind::Binary(_, _, _) => true,

        // TODO: Check if x: &String, *(x).as_ptr() gets triggered
        ExprKind::Unary(_, _) => true,

        // Inner blocks are rvalues.
        ExprKind::If(_, _, _)
        | ExprKind::Loop(_, _, _, _)
        | ExprKind::Match(_, _, _)
        | ExprKind::Block(_, _) => true,

        ExprKind::Field(parent, _) => is_temporary_rvalue(parent),

        // FIXME: some of these get promoted to const/'static ?
        ExprKind::Struct(_, _, _)
        | ExprKind::Array(_)
        | ExprKind::Repeat(_, _)
        | ExprKind::Lit(_) => true,

        // These typecheck to `!`
        ExprKind::Break(_, _) | ExprKind::Continue(_) | ExprKind::Ret(_) | ExprKind::Become(_) => {
            false
        }

        // These typecheck to `()`
        ExprKind::Assign(_, _, _) | ExprKind::AssignOp(_, _, _) => false,

        // Not applicable
        ExprKind::Type(_, _) | ExprKind::Err(_) | ExprKind::Let(_) => false,

        // These are compiler-magic macros
        ExprKind::AddrOf(_, _, _) | ExprKind::OffsetOf(_, _) | ExprKind::InlineAsm(_) => false,

        // TODO: WTF are these
        ExprKind::DropTemps(_) => todo!(),
        ExprKind::Yield(_, _) => todo!(),
    }
}
