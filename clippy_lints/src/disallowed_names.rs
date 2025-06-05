use clippy_config::Conf;
use clippy_utils::diagnostics::span_lint;
use clippy_utils::{is_from_proc_macro, is_in_test};
use rustc_data_structures::fx::FxHashSet;
use rustc_hir::{Pat, PatKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::impl_lint_pass;
use rustc_span::Symbol;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of disallowed names for variables, such
    /// as `foo`.
    ///
    /// ### Why is this bad?
    /// These names are usually placeholder names and should be
    /// avoided.
    ///
    /// ### Example
    /// ```no_run
    /// let foo = 3.14;
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub DISALLOWED_NAMES,
    style,
    "usage of a disallowed/placeholder name"
}

pub struct DisallowedNames {
    disallow: FxHashSet<Symbol>,
}

impl DisallowedNames {
    pub fn new(conf: &'static Conf) -> Self {
        Self {
            disallow: conf.disallowed_names.iter().map(|x| Symbol::intern(x)).collect(),
        }
    }
}

impl_lint_pass!(DisallowedNames => [DISALLOWED_NAMES]);

impl<'tcx> LateLintPass<'tcx> for DisallowedNames {
    fn check_pat(&mut self, cx: &LateContext<'tcx>, pat: &'tcx Pat<'_>) {
        if let PatKind::Binding(.., ident, _) = pat.kind
            && !ident.span.from_expansion()
            && self.disallow.contains(&ident.name)
            && !is_in_test(cx.tcx, pat.hir_id)
            && !is_from_proc_macro(cx, &ident)
        {
            span_lint(
                cx,
                DISALLOWED_NAMES,
                ident.span,
                format!("use of a disallowed/placeholder name `{}`", ident.name),
            );
        }
    }
}
