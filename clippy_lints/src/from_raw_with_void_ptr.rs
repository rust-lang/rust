use clippy_utils::diagnostics::span_lint_and_help;
use clippy_utils::path_def_id;
use clippy_utils::ty::is_c_void;
use rustc_hir::{Expr, ExprKind, QPath};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::RawPtr;
use rustc_middle::ty::TypeAndMut;
use rustc_session::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// ### What it does
    /// Checks if we're passing a `c_void` raw pointer to `Box::from_raw(_)`
    ///
    /// ### Why is this bad?
    /// However, it is easy to run into the pitfall of calling from_raw with the c_void pointer.
    /// Note that the definition of, say, Box::from_raw is:
    ///
    /// `pub unsafe fn from_raw(raw: *mut T) -> Box<T>`
    ///
    /// meaning that if you pass a *mut c_void you will get a Box<c_void>.
    /// Per the safety requirements in the documentation, for this to be safe,
    /// c_void would need to have the same memory layout as the original type, which is often not the case.
    ///
    /// ### Example
    /// ```rust
    /// # use std::ffi::c_void;
    /// let ptr = Box::into_raw(Box::new(42usize)) as *mut c_void;
    /// let _ = unsafe { Box::from_raw(ptr) };
    /// ```
    /// Use instead:
    /// ```rust
    /// # use std::ffi::c_void;
    /// # let ptr = Box::into_raw(Box::new(42usize)) as *mut c_void;
    /// let _ = unsafe { Box::from_raw(ptr as *mut usize) };
    /// ```
    ///
    #[clippy::version = "1.66.0"]
    pub FROM_RAW_WITH_VOID_PTR,
    suspicious,
    "creating a `Box` from a raw void pointer"
}
declare_lint_pass!(FromRawWithVoidPtr => [FROM_RAW_WITH_VOID_PTR]);

impl LateLintPass<'_> for FromRawWithVoidPtr {
    fn check_expr(&mut self, cx: &LateContext<'_>, expr: &Expr<'_>) {
        if let ExprKind::Call(box_from_raw, [arg]) = expr.kind
        && let ExprKind::Path(QPath::TypeRelative(ty, seg)) = box_from_raw.kind
        && seg.ident.name == sym!(from_raw)
        // FIXME: This lint is also applicable to other types, like `Rc`, `Arc` and `Weak`.
        && path_def_id(cx, ty).map_or(false, |id| Some(id) == cx.tcx.lang_items().owned_box())
        && let arg_kind = cx.typeck_results().expr_ty(arg).kind()
        && let RawPtr(TypeAndMut { ty, .. }) = arg_kind
        && is_c_void(cx, *ty) {
            span_lint_and_help(cx, FROM_RAW_WITH_VOID_PTR, expr.span, "creating a `Box` from a raw void pointer", Some(arg.span), "cast this to a pointer of the actual type");
        }
    }
}
