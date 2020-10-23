use rustc_index::bit_set::BitSet;
use rustc_index::vec::IndexVec;
use rustc_middle::mir::{self, BasicBlock, BasicBlockData, TerminatorKind};

const ID_SEPARATOR: &str = ",";

/// A BasicCoverageBlock (BCB) represents the maximal-length sequence of CFG (MIR) BasicBlocks
/// without conditional branches.
///
/// The BCB allows coverage analysis to be performed on a simplified projection of the underlying
/// MIR CFG, without altering the original CFG. Note that running the MIR `SimplifyCfg` transform,
/// is not sufficient, and therefore not necessary, since the BCB-based CFG projection is a more
/// aggressive simplification. For example:
///
///   * The BCB CFG projection ignores (trims) branches not relevant to coverage, such as unwind-
///     related code that is injected by the Rust compiler but has no physical source code to
///     count. This also means a BasicBlock with a `Call` terminator can be merged into its
///     primary successor target block, in the same BCB.
///   * Some BasicBlock terminators support Rust-specific concerns--like borrow-checking--that are
///     not relevant to coverage analysis. `FalseUnwind`, for example, can be treated the same as
///     a `Goto`, and merged with its successor into the same BCB.
///
/// Each BCB with at least one computed `CoverageSpan` will have no more than one `Counter`.
/// In some cases, a BCB's execution count can be computed by `CounterExpression`. Additional
/// disjoint `CoverageSpan`s in a BCB can also be counted by `CounterExpression` (by adding `ZERO`
/// to the BCB's primary counter or expression).
///
/// Dominator/dominated relationships (which are fundamental to the coverage analysis algorithm)
/// between two BCBs can be computed using the `mir::Body` `dominators()` with any `BasicBlock`
/// member of each BCB. (For consistency, BCB's use the first `BasicBlock`, also referred to as the
/// `bcb_leader_bb`.)
///
/// The BCB CFG projection is critical to simplifying the coverage analysis by ensuring graph
/// path-based queries (`is_dominated_by()`, `predecessors`, `successors`, etc.) have branch
/// (control flow) significance.
#[derive(Debug, Clone)]
pub(crate) struct BasicCoverageBlock {
    pub blocks: Vec<BasicBlock>,
}

impl BasicCoverageBlock {
    pub fn leader_bb(&self) -> BasicBlock {
        self.blocks[0]
    }

    pub fn id(&self) -> String {
        format!(
            "@{}",
            self.blocks
                .iter()
                .map(|bb| bb.index().to_string())
                .collect::<Vec<_>>()
                .join(ID_SEPARATOR)
        )
    }
}

pub(crate) struct BasicCoverageBlocks {
    vec: IndexVec<BasicBlock, Option<BasicCoverageBlock>>,
}

impl BasicCoverageBlocks {
    pub fn from_mir(mir_body: &mir::Body<'tcx>) -> Self {
        let mut basic_coverage_blocks =
            BasicCoverageBlocks { vec: IndexVec::from_elem_n(None, mir_body.basic_blocks().len()) };
        basic_coverage_blocks.extract_from_mir(mir_body);
        basic_coverage_blocks
    }

    pub fn iter(&self) -> impl Iterator<Item = &BasicCoverageBlock> {
        self.vec.iter().filter_map(|bcb| bcb.as_ref())
    }

    pub fn num_nodes(&self) -> usize {
        self.vec.len()
    }

    pub fn extract_from_mir(&mut self, mir_body: &mir::Body<'tcx>) {
        // Traverse the CFG but ignore anything following an `unwind`
        let cfg_without_unwind = ShortCircuitPreorder::new(&mir_body, |term_kind| {
            let mut successors = term_kind.successors();
            match &term_kind {
                // SwitchInt successors are never unwind, and all of them should be traversed.

                // NOTE: TerminatorKind::FalseEdge targets from SwitchInt don't appear to be
                // helpful in identifying unreachable code. I did test the theory, but the following
                // changes were not beneficial. (I assumed that replacing some constants with
                // non-deterministic variables might effect which blocks were targeted by a
                // `FalseEdge` `imaginary_target`. It did not.)
                //
                // Also note that, if there is a way to identify BasicBlocks that are part of the
                // MIR CFG, but not actually reachable, here are some other things to consider:
                //
                // Injecting unreachable code regions will probably require computing the set
                // difference between the basic blocks found without filtering out unreachable
                // blocks, and the basic blocks found with the filter; then computing the
                // `CoverageSpans` without the filter; and then injecting `Counter`s or
                // `CounterExpression`s for blocks that are not unreachable, or injecting
                // `Unreachable` code regions otherwise. This seems straightforward, but not
                // trivial.
                //
                // Alternatively, we might instead want to leave the unreachable blocks in
                // (bypass the filter here), and inject the counters. This will result in counter
                // values of zero (0) for unreachable code (and, notably, the code will be displayed
                // with a red background by `llvm-cov show`).
                //
                // TerminatorKind::SwitchInt { .. } => {
                //     let some_imaginary_target = successors.clone().find_map(|&successor| {
                //         let term = mir_body.basic_blocks()[successor].terminator();
                //         if let TerminatorKind::FalseEdge { imaginary_target, .. } = term.kind {
                //             if mir_body.predecessors()[imaginary_target].len() == 1 {
                //                 return Some(imaginary_target);
                //             }
                //         }
                //         None
                //     });
                //     if let Some(imaginary_target) = some_imaginary_target {
                //         box successors.filter(move |&&successor| successor != imaginary_target)
                //     } else {
                //         box successors
                //     }
                // }
                //
                // Note this also required changing the closure signature for the
                // `ShortCurcuitPreorder` to:
                //
                // F: Fn(&'tcx TerminatorKind<'tcx>) -> Box<dyn Iterator<Item = &BasicBlock> + 'a>,
                TerminatorKind::SwitchInt { .. } => successors,

                // For all other kinds, return only the first successor, if any, and ignore unwinds
                _ => successors.next().into_iter().chain(&[]),
            }
        });

        // Walk the CFG using a Preorder traversal, which starts from `START_BLOCK` and follows
        // each block terminator's `successors()`. Coverage spans must map to actual source code,
        // so compiler generated blocks and paths can be ignored. To that end the CFG traversal
        // intentionally omits unwind paths.
        let mut blocks = Vec::new();
        for (bb, data) in cfg_without_unwind {
            if let Some(last) = blocks.last() {
                let predecessors = &mir_body.predecessors()[bb];
                if predecessors.len() > 1 || !predecessors.contains(last) {
                    // The `bb` has more than one _incoming_ edge, and should start its own
                    // `BasicCoverageBlock`. (Note, the `blocks` vector does not yet include `bb`;
                    // it contains a sequence of one or more sequential blocks with no intermediate
                    // branches in or out. Save these as a new `BasicCoverageBlock` before starting
                    // the new one.)
                    self.add_basic_coverage_block(blocks.split_off(0));
                    debug!(
                        "  because {}",
                        if predecessors.len() > 1 {
                            "predecessors.len() > 1".to_owned()
                        } else {
                            format!("bb {} is not in precessors: {:?}", bb.index(), predecessors)
                        }
                    );
                }
            }
            blocks.push(bb);

            let term = data.terminator();

            match term.kind {
                TerminatorKind::Return { .. }
                | TerminatorKind::Abort
                | TerminatorKind::Assert { .. }
                | TerminatorKind::Yield { .. }
                | TerminatorKind::SwitchInt { .. } => {
                    // The `bb` has more than one _outgoing_ edge, or exits the function. Save the
                    // current sequence of `blocks` gathered to this point, as a new
                    // `BasicCoverageBlock`.
                    self.add_basic_coverage_block(blocks.split_off(0));
                    debug!("  because term.kind = {:?}", term.kind);
                    // Note that this condition is based on `TerminatorKind`, even though it
                    // theoretically boils down to `successors().len() != 1`; that is, either zero
                    // (e.g., `Return`, `Abort`) or multiple successors (e.g., `SwitchInt`), but
                    // since the Coverage graph (the BCB CFG projection) ignores things like unwind
                    // branches (which exist in the `Terminator`s `successors()` list) checking the
                    // number of successors won't work.
                }
                TerminatorKind::Goto { .. }
                | TerminatorKind::Resume
                | TerminatorKind::Unreachable
                | TerminatorKind::Drop { .. }
                | TerminatorKind::DropAndReplace { .. }
                | TerminatorKind::Call { .. }
                | TerminatorKind::GeneratorDrop
                | TerminatorKind::FalseEdge { .. }
                | TerminatorKind::FalseUnwind { .. }
                | TerminatorKind::InlineAsm { .. } => {}
            }
        }

        if !blocks.is_empty() {
            // process any remaining blocks into a final `BasicCoverageBlock`
            self.add_basic_coverage_block(blocks.split_off(0));
            debug!("  because the end of the CFG was reached while traversing");
        }
    }

    fn add_basic_coverage_block(&mut self, blocks: Vec<BasicBlock>) {
        let leader_bb = blocks[0];
        let bcb = BasicCoverageBlock { blocks };
        debug!("adding BCB: {:?}", bcb);
        self.vec[leader_bb] = Some(bcb);
    }
}

impl std::ops::Index<BasicBlock> for BasicCoverageBlocks {
    type Output = BasicCoverageBlock;

    fn index(&self, index: BasicBlock) -> &Self::Output {
        self.vec[index].as_ref().expect("is_some if BasicBlock is a BasicCoverageBlock leader")
    }
}

pub struct ShortCircuitPreorder<
    'a,
    'tcx,
    F: Fn(&'tcx TerminatorKind<'tcx>) -> mir::Successors<'tcx>,
> {
    body: &'a mir::Body<'tcx>,
    visited: BitSet<BasicBlock>,
    worklist: Vec<BasicBlock>,
    filtered_successors: F,
}

impl<'a, 'tcx, F: Fn(&'tcx TerminatorKind<'tcx>) -> mir::Successors<'tcx>>
    ShortCircuitPreorder<'a, 'tcx, F>
{
    pub fn new(
        body: &'a mir::Body<'tcx>,
        filtered_successors: F,
    ) -> ShortCircuitPreorder<'a, 'tcx, F> {
        let worklist = vec![mir::START_BLOCK];

        ShortCircuitPreorder {
            body,
            visited: BitSet::new_empty(body.basic_blocks().len()),
            worklist,
            filtered_successors,
        }
    }
}

impl<'a: 'tcx, 'tcx, F: Fn(&'tcx TerminatorKind<'tcx>) -> mir::Successors<'tcx>> Iterator
    for ShortCircuitPreorder<'a, 'tcx, F>
{
    type Item = (BasicBlock, &'a BasicBlockData<'tcx>);

    fn next(&mut self) -> Option<(BasicBlock, &'a BasicBlockData<'tcx>)> {
        while let Some(idx) = self.worklist.pop() {
            if !self.visited.insert(idx) {
                continue;
            }

            let data = &self.body[idx];

            if let Some(ref term) = data.terminator {
                self.worklist.extend((self.filtered_successors)(&term.kind));
            }

            return Some((idx, data));
        }

        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let size = self.body.basic_blocks().len() - self.visited.count();
        (size, Some(size))
    }
}
