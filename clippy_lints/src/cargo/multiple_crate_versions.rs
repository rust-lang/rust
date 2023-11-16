//! lint on multiple versions of a crate being used

use cargo_metadata::{DependencyKind, Metadata, Node, Package, PackageId};
use clippy_utils::diagnostics::span_lint;
use itertools::Itertools;
use rustc_hir::def_id::LOCAL_CRATE;
use rustc_lint::LateContext;
use rustc_span::DUMMY_SP;

use super::MULTIPLE_CRATE_VERSIONS;

pub(super) fn check(cx: &LateContext<'_>, metadata: &Metadata) {
    let local_name = cx.tcx.crate_name(LOCAL_CRATE);
    let mut packages = metadata.packages.clone();
    packages.sort_by(|a, b| a.name.cmp(&b.name));

    if let Some(resolve) = &metadata.resolve
        && let Some(local_id) = packages.iter().find_map(|p| {
            if p.name == local_name.as_str() {
                Some(&p.id)
            } else {
                None
            }
        })
    {
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
                    &format!("multiple versions for dependency `{name}`: {versions}"),
                );
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
