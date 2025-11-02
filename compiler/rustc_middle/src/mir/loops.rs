use rustc_index::bit_set::DenseBitSet;

use super::*;

/// Compute the set of loop headers in the given body. A loop header is usually defined as a block
/// which dominates one of its predecessors. This definition is only correct for reducible CFGs.
/// However, computing dominators is expensive, so we approximate according to the post-order
/// traversal order. A loop header for us is a block which is visited after its predecessor in
/// post-order. This is ok as we mostly need a heuristic.
pub fn maybe_loop_headers(body: &Body<'_>) -> DenseBitSet<BasicBlock> {
    let mut maybe_loop_headers = DenseBitSet::new_empty(body.basic_blocks.len());
    let mut visited = DenseBitSet::new_empty(body.basic_blocks.len());
    for (bb, bbdata) in traversal::postorder(body) {
        // Post-order means we visit successors before the block for acyclic CFGs.
        // If the successor is not visited yet, consider it a loop header.
        for succ in bbdata.terminator().successors() {
            if !visited.contains(succ) {
                maybe_loop_headers.insert(succ);
            }
        }

        // Only mark `bb` as visited after we checked the successors, in case we have a self-loop.
        //     bb1: goto -> bb1;
        let _new = visited.insert(bb);
        debug_assert!(_new);
    }

    maybe_loop_headers
}
