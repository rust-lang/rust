use clippy_config::Conf;
use clippy_utils::diagnostics::span_lint;
use clippy_utils::is_from_proc_macro;
use rustc_data_structures::fx::FxHashSet;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::def_id::{CRATE_DEF_INDEX, DefId};
use rustc_hir::{HirId, ItemKind, Node, Path};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::impl_lint_pass;
use rustc_span::Symbol;
use rustc_span::symbol::kw;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of items through absolute paths, like `std::env::current_dir`.
    ///
    /// ### Why restrict this?
    /// Many codebases have their own style when it comes to importing, but one that is seldom used
    /// is using absolute paths *everywhere*. This is generally considered unidiomatic, and you
    /// should add a `use` statement.
    ///
    /// The default maximum segments (2) is pretty strict, you may want to increase this in
    /// `clippy.toml`.
    ///
    /// Note: One exception to this is code from macro expansion - this does not lint such cases, as
    /// using absolute paths is the proper way of referencing items in one.
    ///
    /// ### Known issues
    ///
    /// There are currently a few cases which are not caught by this lint:
    /// * Macro calls. e.g. `path::to::macro!()`
    /// * Derive macros. e.g. `#[derive(path::to::macro)]`
    /// * Attribute macros. e.g. `#[path::to::macro]`
    ///
    /// ### Example
    /// ```no_run
    /// let x = std::f64::consts::PI;
    /// ```
    /// Use any of the below instead, or anything else:
    /// ```no_run
    /// use std::f64;
    /// use std::f64::consts;
    /// use std::f64::consts::PI;
    /// let x = f64::consts::PI;
    /// let x = consts::PI;
    /// let x = PI;
    /// use std::f64::consts as f64_consts;
    /// let x = f64_consts::PI;
    /// ```
    #[clippy::version = "1.73.0"]
    pub ABSOLUTE_PATHS,
    restriction,
    "checks for usage of an item without a `use` statement"
}
impl_lint_pass!(AbsolutePaths => [ABSOLUTE_PATHS]);

pub struct AbsolutePaths {
    pub absolute_paths_max_segments: u64,
    pub absolute_paths_allowed_crates: FxHashSet<Symbol>,
}

impl AbsolutePaths {
    pub fn new(conf: &'static Conf) -> Self {
        Self {
            absolute_paths_max_segments: conf.absolute_paths_max_segments,
            absolute_paths_allowed_crates: conf
                .absolute_paths_allowed_crates
                .iter()
                .map(|x| Symbol::intern(x))
                .collect(),
        }
    }
}

impl<'tcx> LateLintPass<'tcx> for AbsolutePaths {
    // We should only lint `QPath::Resolved`s, but since `Path` is only used in `Resolved` and `UsePath`
    // we don't need to use a visitor or anything as we can just check if the `Node` for `hir_id` isn't
    // a `Use`
    fn check_path(&mut self, cx: &LateContext<'tcx>, path: &Path<'tcx>, hir_id: HirId) {
        let segments = match path.segments {
            [] | [_] => return,
            // Don't count enum variants and trait items as part of the length.
            [rest @ .., _]
                if let [.., s] = rest
                    && matches!(s.res, Res::Def(DefKind::Enum | DefKind::Trait | DefKind::TraitAlias, _)) =>
            {
                rest
            },
            path => path,
        };
        if let [s1, s2, ..] = segments
            && let has_root = s1.ident.name == kw::PathRoot
            && let first = if has_root { s2 } else { s1 }
            && let len = segments.len() - usize::from(has_root)
            && len as u64 > self.absolute_paths_max_segments
            && let crate_name = if let Res::Def(DefKind::Mod, DefId { index, .. }) = first.res
                && index == CRATE_DEF_INDEX
            {
                // `other_crate::foo` or `::other_crate::foo`
                first.ident.name
            } else if first.ident.name == kw::Crate || has_root {
                // `::foo` or `crate::foo`
                kw::Crate
            } else {
                return;
            }
            && !path.span.from_expansion()
            && let node = cx.tcx.hir_node(hir_id)
            && !matches!(node, Node::Item(item) if matches!(item.kind, ItemKind::Use(..)))
            && !self.absolute_paths_allowed_crates.contains(&crate_name)
            && !is_from_proc_macro(cx, path)
        {
            span_lint(
                cx,
                ABSOLUTE_PATHS,
                path.span,
                "consider bringing this path into scope with the `use` keyword",
            );
        }
    }
}
