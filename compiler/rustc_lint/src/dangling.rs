use rustc_hir::{Expr, ExprKind, LangItem};
use rustc_middle::ty::{self, Ty};
use rustc_session::{declare_lint, declare_lint_pass};
use rustc_span::symbol::sym;

use crate::lints::InstantlyDangling;
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

// FIXME: does not catch UnsafeCell::get
// FIXME: does not catch getting a ref to a temporary and then converting it to a ptr
declare_lint! {
    /// TODO
    pub INSTANTLY_DANGLING_POINTER,
    Warn,
    "detects getting a pointer that will immediately dangle"
}

declare_lint_pass!(DanglingPointers => [TEMPORARY_CSTRING_AS_PTR, INSTANTLY_DANGLING_POINTER]);

impl<'tcx> LateLintPass<'tcx> for DanglingPointers {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if let ExprKind::MethodCall(as_ptr_path, as_ptr_receiver, ..) = expr.kind
            && as_ptr_path.ident.name == sym::as_ptr
            && let ExprKind::MethodCall(unwrap_path, unwrap_receiver, ..) = as_ptr_receiver.kind
            && (unwrap_path.ident.name == sym::unwrap || unwrap_path.ident.name == sym::expect)
            && lint_cstring_as_ptr(cx, unwrap_receiver)
        {
            cx.emit_span_lint(
                TEMPORARY_CSTRING_AS_PTR,
                as_ptr_path.ident.span,
                InstantlyDangling {
                    callee: as_ptr_path.ident.name,
                    ty: "CString".into(),
                    ptr_span: as_ptr_path.ident.span,
                    temporary_span: as_ptr_receiver.span,
                },
            );
            return; // One lint is enough
        }

        if let ExprKind::MethodCall(method, receiver, _args, _span) = expr.kind
            && matches!(method.ident.name, sym::as_ptr | sym::as_mut_ptr)
            && is_temporary_rvalue(receiver)
            && let ty = cx.typeck_results().expr_ty(receiver)
            && is_interesting(cx, ty)
        {
            cx.emit_span_lint(INSTANTLY_DANGLING_POINTER, method.ident.span, InstantlyDangling {
                callee: method.ident.name,
                ty: ty.to_string(),
                ptr_span: method.ident.span,
                temporary_span: receiver.span,
            })
        }
    }
}

fn lint_cstring_as_ptr(cx: &LateContext<'_>, source: &rustc_hir::Expr<'_>) -> bool {
    let source_type = cx.typeck_results().expr_ty(source);
    if let ty::Adt(def, args) = source_type.kind() {
        if cx.tcx.is_diagnostic_item(sym::Result, def.did()) {
            if let ty::Adt(adt, _) = args.type_at(0).kind() {
                if cx.tcx.is_diagnostic_item(sym::cstring_type, adt.did()) {
                    return true;
                }
            }
        }
    }
    false
}

fn is_temporary_rvalue(expr: &Expr<'_>) -> bool {
    match expr.kind {
        // We are not interested in these
        ExprKind::Cast(_, _) | ExprKind::Closure(_) | ExprKind::Tup(_) => false,

        // Const is not temporary.
        ExprKind::ConstBlock(_) | ExprKind::Repeat(_, _) => false,

        // This is literally lvalue.
        ExprKind::Path(_) => false,

        // Calls return rvalues.
        ExprKind::Call(_, _) | ExprKind::MethodCall(_, _, _, _) | ExprKind::Binary(_, _, _) => true,

        // Produces lvalue.
        ExprKind::Unary(_, _) | ExprKind::Index(_, _, _) => false,

        // Inner blocks are rvalues.
        ExprKind::If(_, _, _)
        | ExprKind::Loop(_, _, _, _)
        | ExprKind::Match(_, _, _)
        | ExprKind::Block(_, _) => true,

        ExprKind::DropTemps(inner) => is_temporary_rvalue(inner),
        ExprKind::Field(parent, _) => is_temporary_rvalue(parent),

        ExprKind::Struct(_, _, _) => true,
        // These are 'static
        ExprKind::Lit(_) => false,
        // FIXME: False negatives are possible, but arrays get promoted to 'static way too often.
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

// Array, Vec, String, CString, MaybeUninit, Cell, Box<[_]>, Box<str>, Box<CStr>,
// or any of the above in arbitrary many nested Box'es.
fn is_interesting(cx: &LateContext<'_>, ty: Ty<'_>) -> bool {
    if ty.is_array() {
        true
    } else if let Some(inner) = ty.boxed_ty() {
        inner.is_slice()
            || inner.is_str()
            || inner.ty_adt_def().is_some_and(|def| cx.tcx.is_lang_item(def.did(), LangItem::CStr))
            || is_interesting(cx, inner)
    } else if let Some(def) = ty.ty_adt_def() {
        for lang_item in [LangItem::String, LangItem::MaybeUninit] {
            if cx.tcx.is_lang_item(def.did(), lang_item) {
                return true;
            }
        }
        cx.tcx
            .get_diagnostic_name(def.did())
            .is_some_and(|name| matches!(name, sym::cstring_type | sym::Vec | sym::Cell))
    } else {
        false
    }
}
