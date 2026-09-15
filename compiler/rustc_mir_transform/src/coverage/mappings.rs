use rustc_middle::mir;
use rustc_middle::mir::coverage::{Mapping, MappingKind};
use rustc_middle::ty::TyCtxt;
use rustc_span::ExpnKind;

use crate::coverage::branch;
use crate::coverage::expansion::{self, ExpnTree};
use crate::coverage::graph::CoverageGraph;
use crate::coverage::hir_info::ExtractedHirInfo;
use crate::coverage::spans::extract_refined_covspans;

/// Indicates why mapping extraction failed, for debug-logging purposes.
#[derive(Debug)]
pub(crate) enum MappingsError {
    NoMappings,
    TreeSortFailure,
}

#[derive(Default)]
pub(crate) struct ExtractedMappings {
    pub(crate) mappings: Vec<Mapping>,
}

/// Extracts coverage-relevant spans from MIR, and uses them to create
/// coverage mapping data for inclusion in MIR.
pub(crate) fn extract_mappings_from_mir<'tcx>(
    tcx: TyCtxt<'tcx>,
    mir_body: &mir::Body<'tcx>,
    hir_info: &ExtractedHirInfo,
    graph: &CoverageGraph,
) -> Result<ExtractedMappings, MappingsError> {
    let expn_tree = expansion::build_expn_tree(tcx, mir_body, hir_info, graph)?;

    let mut mappings = vec![];

    // Extract ordinary code mappings from MIR statement/terminator spans.
    extract_refined_covspans(tcx, hir_info, graph, &expn_tree, &mut mappings);

    extract_branch_mappings(hir_info, &expn_tree, &mut mappings);

    if mappings.is_empty() {
        tracing::debug!("no mappings were extracted");
        return Err(MappingsError::NoMappings);
    }
    Ok(ExtractedMappings { mappings })
}

fn extract_branch_mappings(
    hir_info: &ExtractedHirInfo,
    expn_tree: &ExpnTree,
    mappings: &mut Vec<Mapping>,
) {
    // For now, ignore any branch span that was introduced by
    // expansion. This makes things like assert macros less noisy.
    let Some(node) = expn_tree.get(hir_info.body_span.ctxt()) else { return };
    if node.expn_kind != ExpnKind::Root {
        return;
    }

    mappings.extend(node.branch_spans.iter().map(
        |&branch::BranchSpan { span, true_bcb, false_bcb }| Mapping {
            span,
            kind: MappingKind::Branch { true_bcb, false_bcb },
        },
    ));
}
