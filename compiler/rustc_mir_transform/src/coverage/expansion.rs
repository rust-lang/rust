use itertools::Itertools;
use rustc_data_structures::fx::{FxIndexMap, FxIndexSet, IndexEntry};
use rustc_middle::mir;
use rustc_middle::mir::coverage::{BasicCoverageBlock, BranchSpan};
use rustc_span::{ExpnKind, Span, SyntaxContext};

use crate::coverage::from_mir;
use crate::coverage::graph::CoverageGraph;
use crate::coverage::hir_info::ExtractedHirInfo;
use crate::coverage::mappings::MappingsError;

#[derive(Clone, Copy, Debug)]
pub(crate) struct SpanWithBcb {
    pub(crate) span: Span,
    pub(crate) bcb: BasicCoverageBlock,
}

#[derive(Debug)]
pub(crate) struct ExpnTree {
    nodes: FxIndexMap<SyntaxContext, ExpnNode>,
}

impl ExpnTree {
    pub(crate) fn get(&self, context: SyntaxContext) -> Option<&ExpnNode> {
        self.nodes.get(&context)
    }
}

#[derive(Debug)]
pub(crate) struct ExpnNode {
    /// Storing the syntax context in its own node is not strictly necessary,
    /// but is helpful for debugging and might be useful later.
    #[expect(dead_code)]
    pub(crate) context: SyntaxContext,
    /// Index of this node in a depth-first traversal from the root.
    pub(crate) dfs_rank: usize,

    // Useful info extracted from `ExpnData`.
    pub(crate) expn_kind: ExpnKind,
    /// Non-dummy `ExpnData::call_site` span.
    pub(crate) call_site: Option<Span>,
    /// Syntax context of `call_site`, if present.
    /// This links an expansion node to its parent in the tree.
    pub(crate) call_site_context: Option<SyntaxContext>,

    /// Holds the function signature span, if it belongs to this expansion.
    /// Used by special-case code in span refinement.
    pub(crate) fn_sig_span: Option<Span>,
    /// Holds the function body span, if it belongs to this expansion.
    /// Used by special-case code in span refinement.
    pub(crate) body_span: Option<Span>,

    /// Spans (and their associated BCBs) belonging to this expansion.
    pub(crate) spans: Vec<SpanWithBcb>,
    /// Expansions whose call-site is in this expansion.
    pub(crate) child_contexts: FxIndexSet<SyntaxContext>,
    /// The "minimum" and "maximum" BCBs (in dominator order) of ordinary spans
    /// belonging to this tree node and all of its descendants. Used when
    /// creating a single code mapping representing an entire child expansion.
    pub(crate) minmax_bcbs: Option<MinMaxBcbs>,

    /// Branch spans (recorded during MIR building) belonging to this expansion.
    pub(crate) branch_spans: Vec<BranchSpan>,

    /// Hole spans belonging to this expansion, to be carved out from the
    /// code spans during span refinement.
    pub(crate) hole_spans: Vec<Span>,
}

impl ExpnNode {
    fn for_context(context: SyntaxContext) -> Self {
        let expn_data = context.outer_expn_data();

        let call_site = Some(expn_data.call_site).filter(|sp| !sp.is_dummy());
        let call_site_context = try { call_site?.ctxt() };

        Self {
            context,
            dfs_rank: usize::MAX,

            expn_kind: expn_data.kind,
            call_site,
            call_site_context,

            fn_sig_span: None,
            body_span: None,

            spans: vec![],
            child_contexts: FxIndexSet::default(),
            minmax_bcbs: None,

            branch_spans: vec![],

            hole_spans: vec![],
        }
    }
}

/// Extracts raw span/BCB pairs from potentially-different syntax contexts, and
/// arranges them into an "expansion tree" based on their expansion call-sites.
pub(crate) fn build_expn_tree(
    mir_body: &mir::Body<'_>,
    hir_info: &ExtractedHirInfo,
    graph: &CoverageGraph,
) -> Result<ExpnTree, MappingsError> {
    let raw_spans = from_mir::extract_raw_spans_from_mir(mir_body, graph);

    let mut nodes = FxIndexMap::default();
    let new_node = |&context: &SyntaxContext| ExpnNode::for_context(context);

    for from_mir::RawSpanFromMir { raw_span, bcb } in raw_spans {
        let span_with_bcb = SpanWithBcb { span: raw_span, bcb };

        // Create a node for this span's enclosing expansion, and add the span to it.
        let context = span_with_bcb.span.ctxt();
        let node = nodes.entry(context).or_insert_with_key(new_node);
        node.spans.push(span_with_bcb);

        // Now walk up the expansion call-site chain, creating nodes and registering children.
        let mut prev = context;
        let mut curr_context = node.call_site_context;
        while let Some(context) = curr_context {
            let entry = nodes.entry(context);
            let node_existed = matches!(entry, IndexEntry::Occupied(_));

            let node = entry.or_insert_with_key(new_node);
            node.child_contexts.insert(prev);

            if node_existed {
                break;
            }

            prev = context;
            curr_context = node.call_site_context;
        }
    }

    // Sort the tree nodes into depth-first order.
    sort_nodes_depth_first(&mut nodes)?;

    // For each node, determine its "minimum" and "maximum" BCBs, based on its
    // own spans and its immediate children. This relies on the nodes having
    // been sorted, so that each node's children are processed before the node
    // itself.
    for i in (0..nodes.len()).rev() {
        // Computing a node's min/max BCBs requires a shared ref to other nodes.
        let minmax_bcbs = minmax_bcbs_for_expn_tree_node(graph, &nodes, &nodes[i]);
        // Now we can mutate the current node to set its min/max BCBs.
        nodes[i].minmax_bcbs = minmax_bcbs;
    }

    // If we have a span for the function signature, associate it with the
    // corresponding expansion tree node.
    if let Some(fn_sig_span) = hir_info.fn_sig_span
        && let Some(node) = nodes.get_mut(&fn_sig_span.ctxt())
    {
        node.fn_sig_span = Some(fn_sig_span);
    }

    // Also associate the body span with its expansion tree node.
    let body_span = hir_info.body_span;
    if let Some(node) = nodes.get_mut(&body_span.ctxt()) {
        node.body_span = Some(body_span);
    }

    // Associate each hole span (extracted from HIR) with its corresponding
    // expansion tree node.
    for &hole_span in &hir_info.hole_spans {
        let context = hole_span.ctxt();
        let Some(node) = nodes.get_mut(&context) else { continue };
        node.hole_spans.push(hole_span);
    }

    // Associate each branch span (recorded during MIR building) with its
    // corresponding expansion tree node.
    if let Some(coverage_info_hi) = mir_body.coverage_info_hi.as_deref() {
        for branch_span in &coverage_info_hi.branch_spans {
            if let Some(node) = nodes.get_mut(&branch_span.span.ctxt()) {
                node.branch_spans.push(BranchSpan::clone(branch_span));
            }
        }
    }

    Ok(ExpnTree { nodes })
}

/// Sorts the tree nodes in the map into depth-first order.
///
/// This allows subsequent operations to iterate over all nodes, while assuming
/// that every node occurs before all of its descendants.
fn sort_nodes_depth_first(
    nodes: &mut FxIndexMap<SyntaxContext, ExpnNode>,
) -> Result<(), MappingsError> {
    let mut dfs_stack = vec![SyntaxContext::root()];
    let mut next_dfs_rank = 0usize;
    while let Some(context) = dfs_stack.pop() {
        if let Some(node) = nodes.get_mut(&context) {
            node.dfs_rank = next_dfs_rank;
            next_dfs_rank += 1;
            dfs_stack.extend(node.child_contexts.iter().rev().copied());
        }
    }
    nodes.sort_by_key(|_context, node| node.dfs_rank);

    // Verify that the depth-first search visited each node exactly once.
    for (i, &ExpnNode { dfs_rank, .. }) in nodes.values().enumerate() {
        if dfs_rank != i {
            tracing::debug!(dfs_rank, i, "expansion tree node's rank does not match its index");
            return Err(MappingsError::TreeSortFailure);
        }
    }

    Ok(())
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct MinMaxBcbs {
    pub(crate) min: BasicCoverageBlock,
    pub(crate) max: BasicCoverageBlock,
}

/// For a single node in the expansion tree, compute its "minimum" and "maximum"
/// BCBs (in dominator order), from among the BCBs of its immediate spans,
/// and the min/max of its immediate children.
fn minmax_bcbs_for_expn_tree_node(
    graph: &CoverageGraph,
    nodes: &FxIndexMap<SyntaxContext, ExpnNode>,
    node: &ExpnNode,
) -> Option<MinMaxBcbs> {
    let immediate_span_bcbs = node.spans.iter().map(|sp: &SpanWithBcb| sp.bcb);
    let child_minmax_bcbs = node
        .child_contexts
        .iter()
        .flat_map(|id| nodes.get(id))
        .flat_map(|child| child.minmax_bcbs)
        .flat_map(|MinMaxBcbs { min, max }| [min, max]);

    let (min, max) = Iterator::chain(immediate_span_bcbs, child_minmax_bcbs)
        .minmax_by(|&a, &b| graph.cmp_in_dominator_order(a, b))
        .into_option()?;
    Some(MinMaxBcbs { min, max })
}
