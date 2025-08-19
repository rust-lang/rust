use rustc_hir::{self as hir};
use rustc_session::{declare_lint, declare_lint_pass};
use rustc_span::kw;

use crate::{LateContext, LateLintPass, LintContext, lints};

declare_lint! {
    /// The `unqualified_local_imports` lint checks for `use` items that import a local item using a
    /// path that does not start with `self::`, `super::`, or `crate::`.
    ///
    /// ### Example
    ///
    /// ```rust,edition2018
    /// #![feature(unqualified_local_imports)]
    /// #![warn(unqualified_local_imports)]
    ///
    /// mod localmod {
    ///     pub struct S;
    /// }
    ///
    /// use localmod::S;
    /// # // We have to actually use `S`, or else the `unused` warnings suppress the lint we care about.
    /// # pub fn main() {
    /// #     let _x = S;
    /// # }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// This lint is meant to be used with the (unstable) rustfmt setting `group_imports = "StdExternalCrate"`.
    /// That setting makes rustfmt group `self::`, `super::`, and `crate::` imports separately from those
    /// referring to other crates. However, rustfmt cannot know whether `use c::S;` refers to a local module `c`
    /// or an external crate `c`, so it always gets categorized as an import from another crate.
    /// To ensure consistent grouping of imports from the local crate, all local imports must
    /// start with `self::`, `super::`, or `crate::`. This lint can be used to enforce that style.
    pub UNQUALIFIED_LOCAL_IMPORTS,
    Allow,
    "`use` of a local item without leading `self::`, `super::`, or `crate::`",
    @feature_gate = unqualified_local_imports;
}

declare_lint_pass!(UnqualifiedLocalImports => [UNQUALIFIED_LOCAL_IMPORTS]);

impl<'tcx> LateLintPass<'tcx> for UnqualifiedLocalImports {
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx hir::Item<'tcx>) {
        let hir::ItemKind::Use(path, _kind) = item.kind else { return };
        // Check the type and value namespace resolutions for a local crate.
        let is_local_import = matches!(
            path.res.type_ns,
            Some(hir::def::Res::Def(_, def_id)) if def_id.is_local()
        ) || matches!(
            path.res.value_ns,
            Some(hir::def::Res::Def(_, def_id)) if def_id.is_local()
        );
        if !is_local_import {
            return;
        }
        // So this does refer to something local. Let's check whether it starts with `self`,
        // `super`, or `crate`. If the path is empty, that means we have a `use *`, which is
        // equivalent to `use crate::*` so we don't fire the lint in that case.
        let Some(first_seg) = path.segments.first() else { return };
        if matches!(first_seg.ident.name, kw::SelfLower | kw::Super | kw::Crate) {
            return;
        }

        let encl_item_id = cx.tcx.hir_get_parent_item(item.hir_id());
        let encl_item = cx.tcx.hir_node_by_def_id(encl_item_id.def_id);
        if encl_item.fn_kind().is_some() {
            // `use` in a method -- don't lint, that leads to too many undesirable lints
            // when a function imports all variants of an enum.
            return;
        }

        // This `use` qualifies for our lint!
        cx.emit_span_lint(
            UNQUALIFIED_LOCAL_IMPORTS,
            first_seg.ident.span,
            lints::UnqualifiedLocalImportsDiag {},
        );
    }
}
