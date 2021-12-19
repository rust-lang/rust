//! lint on multiple versions of a crate being used

use clippy_utils::diagnostics::span_lint;
use clippy_utils::is_lint_allowed;
use rustc_hir::def_id::LOCAL_CRATE;
use rustc_hir::CRATE_HIR_ID;
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::source_map::DUMMY_SP;

use cargo_metadata::{DependencyKind, Node, Package, PackageId};
use if_chain::if_chain;
use itertools::Itertools;

declare_clippy_lint! {
    /// ### What it does
    /// Checks to see if multiple versions of a crate are being
    /// used.
    ///
    /// ### Why is this bad?
    /// This bloats the size of targets, and can lead to
    /// confusing error messages when structs or traits are used interchangeably
    /// between different versions of a crate.
    ///
    /// ### Known problems
    /// Because this can be caused purely by the dependencies
    /// themselves, it's not always possible to fix this issue.
    ///
    /// ### Example
    /// ```toml
    /// # This will pull in both winapi v0.3.x and v0.2.x, triggering a warning.
    /// [dependencies]
    /// ctrlc = "=3.1.0"
    /// ansi_term = "=0.11.0"
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub MULTIPLE_CRATE_VERSIONS,
    cargo,
    "multiple versions of the same crate being used"
}

declare_lint_pass!(MultipleCrateVersions => [MULTIPLE_CRATE_VERSIONS]);

impl LateLintPass<'_> for MultipleCrateVersions {
    fn check_crate(&mut self, cx: &LateContext<'_>) {
        if is_lint_allowed(cx, MULTIPLE_CRATE_VERSIONS, CRATE_HIR_ID) {
            return;
        }

        let metadata = unwrap_cargo_metadata!(cx, MULTIPLE_CRATE_VERSIONS, true);
        let local_name = cx.tcx.crate_name(LOCAL_CRATE);
        let mut packages = metadata.packages;
        packages.sort_by(|a, b| a.name.cmp(&b.name));

        if_chain! {
            if let Some(resolve) = &metadata.resolve;
            if let Some(local_id) = packages
                .iter()
                .find_map(|p| if p.name == local_name.as_str() { Some(&p.id) } else { None });
            then {
                for (name, group) in &packages.iter().group_by(|p| p.name.clone()) {
                    let group: Vec<&Package> = group.collect();

                    if group.len() <= 1 {
                        continue;
                    }

                    if group.iter().all(|p| is_normal_dep(&resolve.nodes, local_id, &p.id)) {
                        let mut versions: Vec<_> = group.into_iter().map(|p| &p.version).collect();
                        versions.sort();
                        let versions = versions.iter().join(", ");

                        span_lint(
                            cx,
                            MULTIPLE_CRATE_VERSIONS,
                            DUMMY_SP,
                            &format!("multiple versions for dependency `{}`: {}", name, versions),
                        );
                    }
                }
            }
        }
    }
}

fn is_normal_dep(nodes: &[Node], local_id: &PackageId, dep_id: &PackageId) -> bool {
    fn depends_on(node: &Node, dep_id: &PackageId) -> bool {
        node.deps.iter().any(|dep| {
            dep.pkg == *dep_id
                && dep
                    .dep_kinds
                    .iter()
                    .any(|info| matches!(info.kind, DependencyKind::Normal))
        })
    }

    nodes
        .iter()
        .filter(|node| depends_on(node, dep_id))
        .any(|node| node.id == *local_id || is_normal_dep(nodes, local_id, &node.id))
}
