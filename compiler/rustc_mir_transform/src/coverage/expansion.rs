use rustc_data_structures::fx::{FxIndexMap, FxIndexSet, IndexEntry};
use rustc_middle::mir;
use rustc_middle::mir::coverage::BasicCoverageBlock;
use rustc_span::{ExpnId, ExpnKind, Span};

use crate::coverage::from_mir;
use crate::coverage::graph::CoverageGraph;
use crate::coverage::hir_info::ExtractedHirInfo;

#[derive(Clone, Copy, Debug)]
pub(crate) struct SpanWithBcb {
    pub(crate) span: Span,
    pub(crate) bcb: BasicCoverageBlock,
}

#[derive(Debug)]
pub(crate) struct ExpnTree {
    nodes: FxIndexMap<ExpnId, ExpnNode>,
}

impl ExpnTree {
    pub(crate) fn get(&self, expn_id: ExpnId) -> Option<&ExpnNode> {
        self.nodes.get(&expn_id)
    }

    /// Yields the tree node for the given expansion ID (if present), followed
    /// by the nodes of all of its descendants in depth-first order.
    pub(crate) fn iter_node_and_descendants(
        &self,
        root_expn_id: ExpnId,
    ) -> impl Iterator<Item = &ExpnNode> {
        gen move {
            let Some(root_node) = self.get(root_expn_id) else { return };
            yield root_node;

            // Stack of child-node-ID iterators that drives the depth-first traversal.
            let mut iter_stack = vec![root_node.child_expn_ids.iter()];

            while let Some(curr_iter) = iter_stack.last_mut() {
                // Pull the next ID from the top of the stack.
                let Some(&curr_id) = curr_iter.next() else {
                    iter_stack.pop();
                    continue;
                };

                // Yield this node.
                let Some(node) = self.get(curr_id) else { continue };
                yield node;

                // Push the node's children, to be traversed next.
                if !node.child_expn_ids.is_empty() {
                    iter_stack.push(node.child_expn_ids.iter());
                }
            }
        }
    }
}

#[derive(Debug)]
pub(crate) struct ExpnNode {
    /// Storing the expansion ID in its own node is not strictly necessary,
    /// but is helpful for debugging and might be useful later.
    #[expect(dead_code)]
    pub(crate) expn_id: ExpnId,

    // Useful info extracted from `ExpnData`.
    pub(crate) expn_kind: ExpnKind,
    /// Non-dummy `ExpnData::call_site` span.
    pub(crate) call_site: Option<Span>,
    /// Expansion ID of `call_site`, if present.
    /// This links an expansion node to its parent in the tree.
    pub(crate) call_site_expn_id: Option<ExpnId>,

    /// Spans (and their associated BCBs) belonging to this expansion.
    pub(crate) spans: Vec<SpanWithBcb>,
    /// Expansions whose call-site is in this expansion.
    pub(crate) child_expn_ids: FxIndexSet<ExpnId>,

    /// Hole spans belonging to this expansion, to be carved out from the
    /// code spans during span refinement.
    pub(crate) hole_spans: Vec<Span>,
}

impl ExpnNode {
    fn new(expn_id: ExpnId) -> Self {
        let expn_data = expn_id.expn_data();

        let call_site = Some(expn_data.call_site).filter(|sp| !sp.is_dummy());
        let call_site_expn_id = try { call_site?.ctxt().outer_expn() };

        Self {
            expn_id,

            expn_kind: expn_data.kind,
            call_site,
            call_site_expn_id,

            spans: vec![],
            child_expn_ids: FxIndexSet::default(),

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
) -> ExpnTree {
    let raw_spans = from_mir::extract_raw_spans_from_mir(mir_body, graph);

    let mut nodes = FxIndexMap::default();
    let new_node = |&expn_id: &ExpnId| ExpnNode::new(expn_id);

    for from_mir::RawSpanFromMir { raw_span, bcb } in raw_spans {
        let span_with_bcb = SpanWithBcb { span: raw_span, bcb };

        // Create a node for this span's enclosing expansion, and add the span to it.
        let expn_id = span_with_bcb.span.ctxt().outer_expn();
        let node = nodes.entry(expn_id).or_insert_with_key(new_node);
        node.spans.push(span_with_bcb);

        // Now walk up the expansion call-site chain, creating nodes and registering children.
        let mut prev = expn_id;
        let mut curr_expn_id = node.call_site_expn_id;
        while let Some(expn_id) = curr_expn_id {
            let entry = nodes.entry(expn_id);
            let node_existed = matches!(entry, IndexEntry::Occupied(_));

            let node = entry.or_insert_with_key(new_node);
            node.child_expn_ids.insert(prev);

            if node_existed {
                break;
            }

            prev = expn_id;
            curr_expn_id = node.call_site_expn_id;
        }
    }

    // Associate each hole span (extracted from HIR) with its corresponding
    // expansion tree node.
    for &hole_span in &hir_info.hole_spans {
        let expn_id = hole_span.ctxt().outer_expn();
        let Some(node) = nodes.get_mut(&expn_id) else { continue };
        node.hole_spans.push(hole_span);
    }

    ExpnTree { nodes }
}
