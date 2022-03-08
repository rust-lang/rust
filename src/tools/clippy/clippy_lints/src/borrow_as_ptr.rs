use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::is_no_std_crate;
use clippy_utils::source::snippet_opt;
use clippy_utils::{meets_msrv, msrvs};
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir::{BorrowKind, Expr, ExprKind, Mutability, TyKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_semver::RustcVersion;
use rustc_session::{declare_tool_lint, impl_lint_pass};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for the usage of `&expr as *const T` or
    /// `&mut expr as *mut T`, and suggest using `ptr::addr_of` or
    /// `ptr::addr_of_mut` instead.
    ///
    /// ### Why is this bad?
    /// This would improve readability and avoid creating a reference
    /// that points to an uninitialized value or unaligned place.
    /// Read the `ptr::addr_of` docs for more information.
    ///
    /// ### Example
    /// ```rust
    /// let val = 1;
    /// let p = &val as *const i32;
    ///
    /// let mut val_mut = 1;
    /// let p_mut = &mut val_mut as *mut i32;
    /// ```
    /// Use instead:
    /// ```rust
    /// let val = 1;
    /// let p = std::ptr::addr_of!(val);
    ///
    /// let mut val_mut = 1;
    /// let p_mut = std::ptr::addr_of_mut!(val_mut);
    /// ```
    #[clippy::version = "1.60.0"]
    pub BORROW_AS_PTR,
    pedantic,
    "borrowing just to cast to a raw pointer"
}

impl_lint_pass!(BorrowAsPtr => [BORROW_AS_PTR]);

pub struct BorrowAsPtr {
    msrv: Option<RustcVersion>,
}

impl BorrowAsPtr {
    #[must_use]
    pub fn new(msrv: Option<RustcVersion>) -> Self {
        Self { msrv }
    }
}

impl<'tcx> LateLintPass<'tcx> for BorrowAsPtr {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if !meets_msrv(self.msrv.as_ref(), &msrvs::BORROW_AS_PTR) {
            return;
        }

        if expr.span.from_expansion() {
            return;
        }

        if_chain! {
            if let ExprKind::Cast(left_expr, ty) = &expr.kind;
            if let TyKind::Ptr(_) = ty.kind;
            if let ExprKind::AddrOf(BorrowKind::Ref, mutability, e) = &left_expr.kind;

            then {
                let core_or_std = if is_no_std_crate(cx) { "core" } else { "std" };
                let macro_name = match mutability {
                    Mutability::Not => "addr_of",
                    Mutability::Mut => "addr_of_mut",
                };

                span_lint_and_sugg(
                    cx,
                    BORROW_AS_PTR,
                    expr.span,
                    "borrow as raw pointer",
                    "try",
                    format!(
                        "{}::ptr::{}!({})",
                        core_or_std,
                        macro_name,
                        snippet_opt(cx, e.span).unwrap()
                    ),
                    Applicability::MachineApplicable,
                );
            }
        }
    }

    extract_msrv_attr!(LateContext);
}
