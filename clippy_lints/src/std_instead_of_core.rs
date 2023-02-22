use clippy_utils::diagnostics::span_lint_and_help;
use rustc_hir::def_id::DefId;
use rustc_hir::{def::Res, HirId, Path, PathSegment};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::{sym, symbol::kw, Span};

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
    /// imported from core to ensure disabling `alloc` does not cause the crate to fail to compile. This lint
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

#[derive(Default)]
pub struct StdReexports {
    // Paths which can be either a module or a macro (e.g. `std::env`) will cause this check to happen
    // twice. First for the mod, second for the macro. This is used to avoid the lint reporting for the macro
    // when the path could be also be used to access the module.
    prev_span: Span,
}
impl_lint_pass!(StdReexports => [STD_INSTEAD_OF_CORE, STD_INSTEAD_OF_ALLOC, ALLOC_INSTEAD_OF_CORE]);

impl<'tcx> LateLintPass<'tcx> for StdReexports {
    fn check_path(&mut self, cx: &LateContext<'tcx>, path: &Path<'tcx>, _: HirId) {
        if let Res::Def(_, def_id) = path.res
            && let Some(first_segment) = get_first_segment(path)
            && is_stable(cx, def_id)
        {
            let (lint, msg, help) = match first_segment.ident.name {
                sym::std => match cx.tcx.crate_name(def_id.krate) {
                    sym::core => (
                        STD_INSTEAD_OF_CORE,
                        "used import from `std` instead of `core`",
                        "consider importing the item from `core`",
                    ),
                    sym::alloc => (
                        STD_INSTEAD_OF_ALLOC,
                        "used import from `std` instead of `alloc`",
                        "consider importing the item from `alloc`",
                    ),
                    _ => {
                        self.prev_span = path.span;
                        return;
                    },
                },
                sym::alloc => {
                    if cx.tcx.crate_name(def_id.krate) == sym::core {
                        (
                            ALLOC_INSTEAD_OF_CORE,
                            "used import from `alloc` instead of `core`",
                            "consider importing the item from `core`",
                        )
                    } else {
                        self.prev_span = path.span;
                        return;
                    }
                },
                _ => return,
            };
            if path.span != self.prev_span {
                span_lint_and_help(cx, lint, path.span, msg, None, help);
                self.prev_span = path.span;
            }
        }
    }
}

/// Returns the first named segment of a [`Path`].
///
/// If this is a global path (such as `::std::fmt::Debug`), then the segment after [`kw::PathRoot`]
/// is returned.
fn get_first_segment<'tcx>(path: &Path<'tcx>) -> Option<&'tcx PathSegment<'tcx>> {
    match path.segments {
        // A global path will have PathRoot as the first segment. In this case, return the segment after.
        [x, y, ..] if x.ident.name == kw::PathRoot => Some(y),
        [x, ..] => Some(x),
        _ => None,
    }
}

/// Checks if all ancestors of `def_id` are stable, to avoid linting
/// [unstable moves](https://github.com/rust-lang/rust/pull/95956)
fn is_stable(cx: &LateContext<'_>, mut def_id: DefId) -> bool {
    loop {
        if cx
            .tcx
            .lookup_stability(def_id)
            .map_or(false, |stability| stability.is_unstable())
        {
            return false;
        }

        match cx.tcx.opt_parent(def_id) {
            Some(parent) => def_id = parent,
            None => return true,
        }
    }
}
