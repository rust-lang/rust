use clippy_config::Conf;
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::is_from_proc_macro;
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::source::snippet_opt;
use rustc_errors::Applicability;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::{Item, ItemKind, UseKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::Visibility;
use rustc_session::impl_lint_pass;
use rustc_span::symbol::kw;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for `use Trait` where the Trait is only used for its methods and not referenced by a path directly.
    ///
    /// ### Why is this bad?
    /// Traits imported that aren't used directly can be imported anonymously with `use Trait as _`.
    /// It is more explicit, avoids polluting the current scope with unused names and can be useful to show which imports are required for traits.
    ///
    /// ### Example
    /// ```no_run
    /// use std::fmt::Write;
    ///
    /// fn main() {
    ///     let mut s = String::new();
    ///     let _ = write!(s, "hello, world!");
    ///     println!("{s}");
    /// }
    /// ```
    /// Use instead:
    /// ```no_run
    /// use std::fmt::Write as _;
    ///
    /// fn main() {
    ///     let mut s = String::new();
    ///     let _ = write!(s, "hello, world!");
    ///     println!("{s}");
    /// }
    /// ```
    #[clippy::version = "1.83.0"]
    pub UNUSED_TRAIT_NAMES,
    restriction,
    "use items that import a trait but only use it anonymously"
}

pub struct UnusedTraitNames {
    msrv: Msrv,
}

impl UnusedTraitNames {
    pub fn new(conf: &'static Conf) -> Self {
        Self { msrv: conf.msrv }
    }
}

impl_lint_pass!(UnusedTraitNames => [UNUSED_TRAIT_NAMES]);

impl<'tcx> LateLintPass<'tcx> for UnusedTraitNames {
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx Item<'tcx>) {
        if !item.span.from_expansion()
            && let ItemKind::Use(path, UseKind::Single(ident)) = item.kind
            // Ignore imports that already use Underscore
            && ident.name != kw::Underscore
            // Only check traits
            && let Some(Res::Def(DefKind::Trait, _)) = path.res.type_ns
            && cx.tcx.resolutions(()).maybe_unused_trait_imports.contains(&item.owner_id.def_id)
            // Only check this import if it is visible to its module only (no pub, pub(crate), ...)
            && let module = cx.tcx.parent_module_from_def_id(item.owner_id.def_id)
            && cx.tcx.visibility(item.owner_id.def_id) == Visibility::Restricted(module.to_def_id())
            && let Some(last_segment) = path.segments.last()
            && let Some(snip) = snippet_opt(cx, last_segment.ident.span)
            && self.msrv.meets(cx, msrvs::UNDERSCORE_IMPORTS)
            && !is_from_proc_macro(cx, &last_segment.ident)
        {
            let complete_span = last_segment.ident.span.to(ident.span);
            span_lint_and_sugg(
                cx,
                UNUSED_TRAIT_NAMES,
                complete_span,
                "importing trait that is only used anonymously",
                "use",
                format!("{snip} as _"),
                Applicability::MachineApplicable,
            );
        }
    }
}
