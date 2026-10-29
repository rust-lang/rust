use rustc_attr_ir::find_attr;
use rustc_hir as hir;
use rustc_lint_defs::{declare_lint, declare_lint_pass, fcw};

use crate::LintContext as _;
use crate::diagnostics::UnmarkedUncheckedTypeAlias;

declare_lint! {
    /// The `unmarked_unchecked_type_aliases` lint suggests annotating unchecked type aliases with
    /// attribute `#[rustc_unchecked_type_alias]`.
    ///
    /// ### Example
    ///
    #[cfg_attr(bootstrap, doc = "```rust,ignore")]
    #[cfg_attr(not(bootstrap), doc = "```rust")]
    /// #![feature(marked_unchecked_type_aliases)]
    /// #![warn(unmarked_unchecked_type_aliases)]
    ///
    /// type Alias = [(); panic!()];
    #[cfg_attr(bootstrap, doc = "```")]
    #[cfg_attr(not(bootstrap), doc = "```")]
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// <!-- FIXME(checked_type_aliases): Write an explainer. -->
    pub UNMARKED_UNCHECKED_TYPE_ALIASES,
    Allow,
    "suggest adding `#[rustc_unchecked_type_aliases]` to unchecked type aliases",
    @feature_gate = marked_unchecked_type_aliases;
    @future_incompatible = FutureIncompatibleInfo {
        // FIXME(checked_type_aliases): The generated URL obviously points to nothing which is fine
        //     for now. However, once a proper edition has been assigned to us, actually write a
        //     chapter in the edition guide.
        reason: fcw!(EditionSemanticsChange future "checked-type-aliases"),
    };
}

declare_lint_pass!(UnmarkedUncheckedTypeAliases => [UNMARKED_UNCHECKED_TYPE_ALIASES]);

impl<'tcx> crate::LateLintPass<'tcx> for UnmarkedUncheckedTypeAliases {
    fn check_item(&mut self, cx: &crate::LateContext<'tcx>, item: &'tcx hir::Item<'tcx>) {
        let hir::ItemKind::TyAlias(..) = item.kind else { return };

        let def_id = item.owner_id.to_def_id();

        if cx.tcx.type_alias_is_checked(def_id) || find_attr!(cx.tcx, def_id, UncheckedTypeAlias) {
            return;
        }

        // FIXME(checked_type_aliases): Employ a lot more sophisticated checks to figure out if the
        // implicitly unchecked type alias can become a checked type alias without breaking the
        // definition site or any usage sites.

        let item_lo = item.span.shrink_to_lo();

        cx.emit_span_lint(
            UNMARKED_UNCHECKED_TYPE_ALIASES,
            cx.tcx.def_span(def_id),
            UnmarkedUncheckedTypeAlias {
                sugg: item_lo,
                padding: cx.tcx.sess.source_map().indentation_before(item_lo).unwrap_or_default(),
            },
        );
    }
}
