use clippy_utils::diagnostics::span_lint_and_help;
use clippy_utils::ty::is_c_void;
use clippy_utils::{path_def_id, sym};
use rustc_hir::def_id::DefId;
use rustc_hir::{Expr, ExprKind, QPath};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty;
use rustc_session::declare_lint_pass;

declare_clippy_lint! {
    /// ### What it does
    /// Checks if we're passing a `c_void` raw pointer to `{Box,Rc,Arc,Weak}::from_raw(_)`
    ///
    /// ### Why is this bad?
    /// When dealing with `c_void` raw pointers in FFI, it is easy to run into the pitfall of calling `from_raw` with the `c_void` pointer.
    /// The type signature of `Box::from_raw` is `fn from_raw(raw: *mut T) -> Box<T>`, so if you pass a `*mut c_void` you will get a `Box<c_void>` (and similarly for `Rc`, `Arc` and `Weak`).
    /// For this to be safe, `c_void` would need to have the same memory layout as the original type, which is often not the case.
    ///
    /// ### Example
    /// ```no_run
    /// # use std::ffi::c_void;
    /// let ptr = Box::into_raw(Box::new(42usize)) as *mut c_void;
    /// let _ = unsafe { Box::from_raw(ptr) };
    /// ```
    /// Use instead:
    /// ```no_run
    /// # use std::ffi::c_void;
    /// # let ptr = Box::into_raw(Box::new(42usize)) as *mut c_void;
    /// let _ = unsafe { Box::from_raw(ptr as *mut usize) };
    /// ```
    ///
    #[clippy::version = "1.67.0"]
    pub FROM_RAW_WITH_VOID_PTR,
    suspicious,
    "creating a `Box` from a void raw pointer"
}
declare_lint_pass!(FromRawWithVoidPtr => [FROM_RAW_WITH_VOID_PTR]);

impl LateLintPass<'_> for FromRawWithVoidPtr {
    fn check_expr(&mut self, cx: &LateContext<'_>, expr: &Expr<'_>) {
        if let ExprKind::Call(box_from_raw, [arg]) = expr.kind
            && let ExprKind::Path(QPath::TypeRelative(ty, seg)) = box_from_raw.kind
            && seg.ident.name == sym::from_raw
            && let Some(type_str) = path_def_id(cx, ty).and_then(|id| def_id_matches_type(cx, id))
            && let arg_kind = cx.typeck_results().expr_ty(arg).kind()
            && let ty::RawPtr(ty, _) = arg_kind
            && is_c_void(cx, *ty)
        {
            let msg = format!("creating a `{type_str}` from a void raw pointer");
            span_lint_and_help(
                cx,
                FROM_RAW_WITH_VOID_PTR,
                expr.span,
                msg,
                Some(arg.span),
                "cast this to a pointer of the appropriate type",
            );
        }
    }
}

/// Checks whether a `DefId` matches `Box`, `Rc`, `Arc`, or one of the `Weak` types.
/// Returns a static string slice with the name of the type, if one was found.
fn def_id_matches_type(cx: &LateContext<'_>, def_id: DefId) -> Option<&'static str> {
    // Box
    if Some(def_id) == cx.tcx.lang_items().owned_box() {
        return Some("Box");
    }

    if let Some(symbol) = cx.tcx.get_diagnostic_name(def_id) {
        if symbol == sym::Arc {
            return Some("Arc");
        } else if symbol == sym::Rc {
            return Some("Rc");
        }
    }

    if matches!(cx.tcx.get_diagnostic_name(def_id), Some(sym::RcWeak | sym::ArcWeak)) {
        Some("Weak")
    } else {
        None
    }
}
