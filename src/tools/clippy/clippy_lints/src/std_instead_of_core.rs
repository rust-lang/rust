use clippy_utils::diagnostics::span_lint_and_help;
use rustc_hir::{def::Res, HirId, Path, PathSegment};
use rustc_lint::{LateContext, LateLintPass, Lint};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::{sym, symbol::kw, Symbol};

declare_clippy_lint! {
    /// ### What it does
    ///
    /// Finds items imported through `std` when available through `core`.
    ///
    /// ### Why is this bad?
    ///
    /// Crates which have `no_std` compatibility may wish to ensure types are imported from core to ensure
    /// disabling `std` does not cause the crate to fail to compile. This lint is also useful for crates
    /// migrating to become `no_std` compatible.
    ///
    /// ### Example
    /// ```rust
    /// use std::hash::Hasher;
    /// ```
    /// Use instead:
    /// ```rust
    /// use core::hash::Hasher;
    /// ```
    #[clippy::version = "1.64.0"]
    pub STD_INSTEAD_OF_CORE,
    restriction,
    "type is imported from std when available in core"
}

declare_clippy_lint! {
    /// ### What it does
    ///
    /// Finds items imported through `std` when available through `alloc`.
    ///
    /// ### Why is this bad?
    ///
    /// Crates which have `no_std` compatibility and require alloc may wish to ensure types are imported from
    /// alloc to ensure disabling `std` does not cause the crate to fail to compile. This lint is also useful
    /// for crates migrating to become `no_std` compatible.
    ///
    /// ### Example
    /// ```rust
    /// use std::vec::Vec;
    /// ```
    /// Use instead:
    /// ```rust
    /// # extern crate alloc;
    /// use alloc::vec::Vec;
    /// ```
    #[clippy::version = "1.64.0"]
    pub STD_INSTEAD_OF_ALLOC,
    restriction,
    "type is imported from std when available in alloc"
}

declare_clippy_lint! {
    /// ### What it does
    ///
    /// Finds items imported through `alloc` when available through `core`.
    ///
    /// ### Why is this bad?
    ///
    /// Crates which have `no_std` compatibility and may optionally require alloc may wish to ensure types are
    /// imported from alloc to ensure disabling `alloc` does not cause the crate to fail to compile. This lint
    /// is also useful for crates migrating to become `no_std` compatible.
    ///
    /// ### Example
    /// ```rust
    /// # extern crate alloc;
    /// use alloc::slice::from_ref;
    /// ```
    /// Use instead:
    /// ```rust
    /// use core::slice::from_ref;
    /// ```
    #[clippy::version = "1.64.0"]
    pub ALLOC_INSTEAD_OF_CORE,
    restriction,
    "type is imported from alloc when available in core"
}

declare_lint_pass!(StdReexports => [STD_INSTEAD_OF_CORE, STD_INSTEAD_OF_ALLOC, ALLOC_INSTEAD_OF_CORE]);

impl<'tcx> LateLintPass<'tcx> for StdReexports {
    fn check_path(&mut self, cx: &LateContext<'tcx>, path: &Path<'tcx>, _: HirId) {
        // std_instead_of_core
        check_path(cx, path, sym::std, sym::core, STD_INSTEAD_OF_CORE);
        // std_instead_of_alloc
        check_path(cx, path, sym::std, sym::alloc, STD_INSTEAD_OF_ALLOC);
        // alloc_instead_of_core
        check_path(cx, path, sym::alloc, sym::core, ALLOC_INSTEAD_OF_CORE);
    }
}

fn check_path(cx: &LateContext<'_>, path: &Path<'_>, krate: Symbol, suggested_crate: Symbol, lint: &'static Lint) {
    if_chain! {
        // check if path resolves to the suggested crate.
        if let Res::Def(_, def_id) = path.res;
        if suggested_crate == cx.tcx.crate_name(def_id.krate);

        // check if the first segment of the path is the crate we want to identify
        if let Some(path_root_segment) = get_first_segment(path);

        // check if the path matches the crate we want to suggest the other path for.
        if krate == path_root_segment.ident.name;
        then {
            span_lint_and_help(
                cx,
                lint,
                path.span,
                &format!("used import from `{}` instead of `{}`", krate, suggested_crate),
                None,
                &format!("consider importing the item from `{}`", suggested_crate),
            );
        }
    }
}

/// Returns the first named segment of a [`Path`].
///
/// If this is a global path (such as `::std::fmt::Debug`), then the segment after [`kw::PathRoot`]
/// is returned.
fn get_first_segment<'tcx>(path: &Path<'tcx>) -> Option<&'tcx PathSegment<'tcx>> {
    let segment = path.segments.first()?;

    // A global path will have PathRoot as the first segment. In this case, return the segment after.
    if segment.ident.name == kw::PathRoot {
        path.segments.get(1)
    } else {
        Some(segment)
    }
}
