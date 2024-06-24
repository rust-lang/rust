use rustc_hir as hir;
use rustc_session::{declare_lint, declare_lint_pass};
use rustc_span::symbol::kw;

use crate::{lints, LateContext, LateLintPass, LintContext};

declare_lint! {
    /// The `unclear_local_imports` lint checks for `use` items that import a local item using a
    /// path that does not start with `self::`, `super::`, or `crate::`.
    ///
    /// ### Example
    ///
    /// ```rust,edition2018
    /// #![warn(unclear_local_imports)]
    ///
    /// mod localmod {
    ///     pub struct S;
    /// }
    ///
    /// use localmod::S;
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// This lint is meant to be used with the (unstable) rustfmt setting `group_imports = "StdExternalCrate"`.
    /// That setting makes rustfmt group `self::`, `super::`, and `crate::` imports separately from those
    /// refering to other crates. However, rustfmt cannot know whether `use c::S;` refers to a local module `c`
    /// or an external crate `c`, so it always gets categorized as an import from another crate.
    /// To ensure consistent grouping of imports from the local crate, all local imports must
    /// start with `self::`, `super::`, or `crate::`. This lint can be used to enforce that style.
    pub UNCLEAR_LOCAL_IMPORTS,
    Allow,
    "`use` of a local item without leading `self::`, `super::`, or `crate::`"
}

declare_lint_pass!(UnclearLocalImports => [UNCLEAR_LOCAL_IMPORTS]);

impl<'tcx> LateLintPass<'tcx> for UnclearLocalImports {
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx hir::Item<'tcx>) {
        let hir::ItemKind::Use(path, _kind) = item.kind else { return };
        // `path` has three resolutions for the type, module, value namespaces.
        // However, it shouldn't be possible for those to be in different crates so we only check the first.
        let Some(hir::def::Res::Def(_def_kind, def_id)) = path.res.first() else { return };
        if !def_id.is_local() {
            return;
        }
        // So this does refer to something local. Let's check whether it starts with `self`,
        // `super`, or `crate`. If the path is empty, that means we have a `use *`, which is
        // equivalent to `use crate::*` so we don't fire the lint in that case.
        let Some(first_seg) = path.segments.first() else { return };
        if matches!(first_seg.ident.name, kw::SelfLower | kw::Super | kw::Crate) {
            return;
        }

        // This `use` qualifies for our lint!
        cx.emit_span_lint(
            UNCLEAR_LOCAL_IMPORTS,
            first_seg.ident.span,
            lints::UnclearLocalImportsDiag {},
        );
    }
}
