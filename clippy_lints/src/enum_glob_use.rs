//! lint on `use`ing all variants of an enum

use crate::utils::span_lint;
use rustc::hir::def::{DefKind, Res};
use rustc::hir::*;
use rustc::lint::{LateContext, LateLintPass, LintArray, LintPass};
use rustc::{declare_lint_pass, declare_tool_lint};
use syntax::source_map::Span;

declare_clippy_lint! {
    /// **What it does:** Checks for `use Enum::*`.
    ///
    /// **Why is this bad?** It is usually better style to use the prefixed name of
    /// an enumeration variant, rather than importing variants.
    ///
    /// **Known problems:** Old-style enumerations that prefix the variants are
    /// still around.
    ///
    /// **Example:**
    /// ```rust
    /// use std::cmp::Ordering::*;
    /// ```
    pub ENUM_GLOB_USE,
    pedantic,
    "use items that import all variants of an enum"
}

declare_lint_pass!(EnumGlobUse => [ENUM_GLOB_USE]);

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for EnumGlobUse {
    fn check_mod(&mut self, cx: &LateContext<'a, 'tcx>, m: &'tcx Mod, _: Span, _: HirId) {
        let map = cx.tcx.hir();
        // only check top level `use` statements
        for item in &m.item_ids {
            self.lint_item(cx, map.expect_item(item.id));
        }
    }
}

impl EnumGlobUse {
    fn lint_item(self, cx: &LateContext<'_, '_>, item: &Item) {
        if item.vis.node.is_pub() {
            return; // re-exports are fine
        }
        if let ItemKind::Use(ref path, UseKind::Glob) = item.node {
            if let Res::Def(DefKind::Enum, _) = path.res {
                span_lint(cx, ENUM_GLOB_USE, item.span, "don't use glob imports for enum variants");
            }
        }
    }
}
