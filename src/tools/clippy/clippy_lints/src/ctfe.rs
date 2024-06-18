use rustc_hir::def_id::LocalDefId;
use rustc_hir::intravisit::FnKind;
use rustc_hir::{Body, FnDecl};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::declare_lint_pass;
use rustc_span::Span;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for comparisons where one side of the relation is
    /// either the minimum or maximum value for its type and warns if it involves a
    /// case that is always true or always false. Only integer and boolean types are
    /// checked.
    ///
    /// ### Why is this bad?
    /// An expression like `min <= x` may misleadingly imply
    /// that it is possible for `x` to be less than the minimum. Expressions like
    /// `max < x` are probably mistakes.
    ///
    /// ### Known problems
    /// For `usize` the size of the current compile target will
    /// be assumed (e.g., 64 bits on 64 bit systems). This means code that uses such
    /// a comparison to detect target pointer width will trigger this lint. One can
    /// use `mem::sizeof` and compare its value or conditional compilation
    /// attributes
    /// like `#[cfg(target_pointer_width = "64")] ..` instead.
    ///
    /// ### Example
    /// ```no_run
    /// let vec: Vec<isize> = Vec::new();
    /// if vec.len() <= 0 {}
    /// if 100 > i32::MAX {}
    /// ```
    #[clippy::version = "1.82.0"]
    pub CLIPPY_CTFE,
    correctness,
    "a comparison with a maximum or minimum value that is always true or false"
}

declare_lint_pass! { ClippyCtfe => [CLIPPY_CTFE] }

impl<'tcx> LateLintPass<'tcx> for ClippyCtfe {
    fn check_fn(
        &mut self,
        cx: &LateContext<'_>,
        _: FnKind<'tcx>,
        _: &'tcx FnDecl<'tcx>,
        _: &'tcx Body<'tcx>,
        _: Span,
        defid: LocalDefId,
    ) {
        cx.tcx.ensure().mir_drops_elaborated_and_const_checked(defid); // Lint
    }
}
