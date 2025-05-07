use cargo_metadata::{DependencyKind, Metadata, Node, Package, PackageId};
use clippy_utils::diagnostics::span_lint;
use itertools::Itertools;
use rustc_data_structures::fx::FxHashSet;
use rustc_hir::def_id::LOCAL_CRATE;
use rustc_lint::LateContext;
use rustc_span::DUMMY_SP;

use super::MULTIPLE_CRATE_VERSIONS;

pub(super) fn check(cx: &LateContext<'_>, metadata: &Metadata, allowed_duplicate_crates: &FxHashSet<String>) {
    let local_name = cx.tcx.crate_name(LOCAL_CRATE);
    let mut packages = metadata.packages.clone();
    packages.sort_by(|a, b| a.name.cmp(&b.name));

    if let Some(resolve) = &metadata.resolve
        && let Some(local_id) = packages.iter().find_map(|p| {
            // p.name contains the original crate names with dashes intact
            // local_name contains the crate name as a namespace, with the dashes converted to underscores
            // the code below temporarily rectifies this discrepancy
            if p.name
                .as_bytes()
                .iter()
                .map(|b| if b == &b'-' { &b'_' } else { b })
                .eq(local_name.as_str().as_bytes())
            {
                Some(&p.id)
            } else {
                None
            }
        })
    {
        for (name, group) in &packages
            .iter()
            .filter(|p| !allowed_duplicate_crates.contains(&p.name))
            .group_by(|p| &p.name)
        {
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
                    format!("multiple versions for dependency `{name}`: {versions}"),
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
