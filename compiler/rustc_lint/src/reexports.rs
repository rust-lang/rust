use crate::lints::UselessAnonymousReexportDiag;
use crate::{LateContext, LateLintPass, LintContext};
use rustc_hir::def::DefKind;
use rustc_hir::def_id::DefId;
use rustc_hir::{Item, ItemKind, UseKind};
use rustc_middle::ty::Visibility;
use rustc_span::symbol::kw;
use rustc_span::Span;

declare_lint! {
    /// The `useless_anonymous_reexport` lint checks if anonymous re-exports
    /// are re-exports of traits.
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// #![deny(useless_anonymous_reexport)]
    ///
    /// mod sub {
    ///     pub struct Bar;
    /// }
    ///
    /// pub use self::sub::Bar as _;
    /// # fn main() {}
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Anonymous re-exports are only useful if it's a re-export of a trait
    /// in case you want to give access to it. If you re-export any other kind,
    /// you won't be able to use it since its name won't be accessible.
    pub USELESS_ANONYMOUS_REEXPORT,
    Warn,
    "useless anonymous re-export"
}

declare_lint_pass!(UselessAnonymousReexport => [USELESS_ANONYMOUS_REEXPORT]);

fn emit_err(cx: &LateContext<'_>, span: Span, def_id: DefId) {
    let article = cx.tcx.def_descr_article(def_id);
    let desc = cx.tcx.def_descr(def_id);
    cx.emit_spanned_lint(
        USELESS_ANONYMOUS_REEXPORT,
        span,
        UselessAnonymousReexportDiag { article, desc },
    );
}

impl<'tcx> LateLintPass<'tcx> for UselessAnonymousReexport {
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx Item<'tcx>) {
        if let ItemKind::Use(path, kind) = item.kind &&
            !matches!(kind, UseKind::Glob) &&
            item.ident.name == kw::Underscore &&
            // We only want re-exports. If it's just a `use X;`, then we ignore it.
            match cx.tcx.local_visibility(item.owner_id.def_id) {
                Visibility::Public => true,
                Visibility::Restricted(level) => {
                    level != cx.tcx.parent_module_from_def_id(item.owner_id.def_id)
                }
            }
        {
            for def_id in path.res.iter().filter_map(|r| r.opt_def_id()) {
                match cx.tcx.def_kind(def_id) {
                    DefKind::Trait | DefKind::TraitAlias => {}
                    DefKind::TyAlias => {
                        let ty = cx.tcx.type_of(def_id);
                        if !ty.0.is_trait() {
                            emit_err(cx, item.span, def_id);
                            break;
                        }
                    }
                    _ => {
                        emit_err(cx, item.span, def_id);
                        break;
                    }
                }
            }
        }
    }
}
