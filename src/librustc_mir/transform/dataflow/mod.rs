use rustc::mir::repr as mir;
use rustc::mir::repr::{BasicBlock, BasicBlockData, Mir, START_BLOCK};
use rustc_data_structures::bitvec::BitVector;
use rustc_data_structures::indexed_vec::{IndexVec, Idx};
pub use self::lattice::*;
pub use self::combinators::*;

mod lattice;
mod combinators;

pub enum StatementChange<'tcx> {
    Remove,
    Statement(mir::Statement<'tcx>)
}
pub enum TerminatorChange<'tcx> {
    Terminator(mir::Terminator<'tcx>)
}

pub trait Transfer<'tcx> {
    type Lattice: Lattice;

    /// Return type of terminator transfer function.
    ///
    /// `Vec<Lattice>` for forward analysis, `Lattice` for backward analysis.
    type TerminatorReturn;

    /// The transfer function which given a statement and a fact produces another fact which is
    /// true after the statement.
    fn stmt(&self, stmt: &mir::Statement<'tcx>, fact: Self::Lattice) -> Self::Lattice;

    /// The transfer function which given a terminator and a fact produces another fact for each
    /// outgoing edge from the terminator.
    fn term(&self, term: &mir::Terminator<'tcx>, fact: Self::Lattice) -> Self::TerminatorReturn;
}

pub trait Rewrite<'tcx, T: Transfer<'tcx>> {
    /// Calculate changes to the statements.
    fn stmt(&self, stmt: &mir::Statement<'tcx>, fact: &T::Lattice)
    -> StatementChange<'tcx>;

    /// Calculate changes to the terminators.
    fn term(&self, term: &mir::Terminator<'tcx>, fact: &T::Lattice)
    -> TerminatorChange<'tcx>;

    /// Combine two rewrites using RewriteAndThen combinator.
    fn and_then<R2>(self, other: R2)
    -> RewriteAndThen<'tcx, T, Self, R2>
    where Self: Sized, R2: Rewrite<'tcx, T>
    {
        RewriteAndThen::new(self, other)
    }
}

#[derive(Clone)]
struct Knowledge<'tcx, F> {
    fact: F,
    new_block: Option<BasicBlockData<'tcx>>
}

pub struct Dataflow<'a, 'tcx: 'a, T, R>
where T: Transfer<'tcx>, R: Rewrite<'tcx, T>
{
    mir: &'a Mir<'tcx>,
    // FIXME: bitvector may not be the best choice
    queue: BitVector,
    knowledge: IndexVec<BasicBlock, Option<Knowledge<'tcx, T::Lattice>>>,
    rewrite: R,
    transfer: T
}

impl<'a, 'tcx, L, T, R> Dataflow<'a, 'tcx, T, R>
where L: Lattice,
      T: Transfer<'tcx, Lattice=L, TerminatorReturn=Vec<L>>,
      R: Rewrite<'tcx, T>
{
    pub fn forward(mir: &'a Mir<'tcx>, transfer: T, rewrite: R) -> Mir<'tcx> {
        let block_count = mir.basic_blocks().len();
        let mut queue = BitVector::new(block_count);
        queue.insert(START_BLOCK.index());
        let mut dataflow = Dataflow {
            mir: mir,
            queue: queue,
            knowledge: IndexVec::with_capacity(block_count),
            rewrite: rewrite,
            transfer: transfer,
        };
        dataflow.knowledge.extend(::std::iter::repeat(None).take(block_count));
        dataflow.fixpoint(Self::forward_block);
        dataflow.construct_mir()
    }

    fn forward_block(&mut self, bb: BasicBlock, mut fact: T::Lattice) {
        // debug!("forward dataflow analysing {:?}", block);
        let bb_data = &self.mir[bb];
        let mut new_stmts = Vec::with_capacity(bb_data.statements.len());
        for stmt in &bb_data.statements {
            match self.rewrite.stmt(stmt, &fact) {
                StatementChange::Remove => {}
                StatementChange::Statement(stmt) => {
                    fact = self.transfer.stmt(&stmt, fact);
                    new_stmts.push(stmt);
                }
            }
        }
        let terminator = bb_data.terminator();
        let (new_facts, new_term) = match self.rewrite.term(terminator, &fact) {
            TerminatorChange::Terminator(term) => (self.transfer.term(&term, fact), term),
        };
        let successors = new_term.successors().into_owned();
        assert!(successors.len() == new_facts.len(), "new_facts.len() must match successor count");
        self.replace_block(bb, BasicBlockData {
            statements: new_stmts,
            terminator: Some(new_term),
            is_cleanup: bb_data.is_cleanup,
        });
        for (fact, &target) in new_facts.into_iter().zip(successors.iter()) {
            if self.update_fact(target, fact) {
                self.queue.insert(target.index());
            }
        }
    }
}

impl<'a, 'tcx, L, T, R> Dataflow<'a, 'tcx, T, R>
where L: Lattice,
      T: Transfer<'tcx, Lattice=L, TerminatorReturn=L>,
      R: Rewrite<'tcx, T>
{
    pub fn backward(mir: &'a Mir<'tcx>, transfer: T, rewrite: R) -> Mir<'tcx> {
        let block_count = mir.basic_blocks().len();
        let mut queue = BitVector::new(block_count);
        mir_exits(mir, &mut queue);
        let mut dataflow = Dataflow {
            mir: mir,
            queue: queue,
            knowledge: IndexVec::with_capacity(block_count),
            rewrite: rewrite,
            transfer: transfer,
        };
        dataflow.knowledge.extend(::std::iter::repeat(None).take(block_count));
        dataflow.fixpoint(Self::backward_block);
        dataflow.construct_mir()
    }

    fn backward_block(&mut self, bb: BasicBlock, fact: T::Lattice) {
        let bb_data = &self.mir[bb];
        let terminator = bb_data.terminator();
        let (mut fact, new_term) = match self.rewrite.term(terminator, &fact) {
            TerminatorChange::Terminator(term) => (self.transfer.term(&term, fact), term),
        };
        let mut new_stmts = Vec::with_capacity(bb_data.statements.len());
        for stmt in bb_data.statements.iter().rev() {
            match self.rewrite.stmt(stmt, &fact) {
                StatementChange::Remove => {}
                StatementChange::Statement(stmt) => {
                    fact = self.transfer.stmt(&stmt, fact);
                    new_stmts.push(stmt);
                }
            }
        }
        new_stmts.reverse();
        self.replace_block(bb, BasicBlockData {
            statements: new_stmts,
            terminator: Some(new_term),
            is_cleanup: bb_data.is_cleanup,
        });
        for &target in self.mir.predecessors_for(bb).iter() {
            if self.update_fact(target, fact.clone()) {
                self.queue.insert(target.index());
            }
        }
    }
}

impl<'a, 'tcx, T, R> Dataflow<'a, 'tcx, T, R>
where T: Transfer<'tcx>,
      R: Rewrite<'tcx, T>
{
    fn fixpoint<BF>(&mut self, f: BF)
    where BF: Fn(&mut Self, BasicBlock, T::Lattice)
    {
        // In the fixpoint loop we never modify the incoming MIR and build up a new MIR in
        // order to avoid problems with speculative analysis. As to why speculative analysis is
        // problematic, consider a constant propagation pass for a loop.
        //
        // → --- { idx = 1 } ---
        // |    idx = idx + 1  # replaced with `idx = 2`
        // |   if(...) break;  # consider else branch taken
        // ↑ --- { idx = 2 } ---
        //
        // Here once we analyse the backedge the fact {idx = 1} is joined with fact {idx = 2}
        // producing a Top ({idx = ⊤}) and rendering our replacement of `idx = idx + 1` with `idx =
        // 2` invalid.
        //
        // If the input graph is immutable, we can always pass callback the original block with the
        // most recent facts for it, thus avoiding the problem altogether.
        while let Some(block) = self.queue.pop() {
            let block = BasicBlock::new(block);
            let fact: T::Lattice = self.recall_fact(block);
            f(self, block, fact);
        }
    }

    fn recall_fact(&self, bb: BasicBlock) -> T::Lattice {
        self.knowledge[bb].as_ref().map(|k| k.fact.clone()).unwrap_or_else(|| T::Lattice::bottom())
    }

    fn update_fact(&mut self, bb: BasicBlock, new_fact: T::Lattice) -> bool {
        match self.knowledge[bb] {
            ref mut val@None => {
                // In case of no prior knowledge about this block, we essentially introduce new
                // data and thus return true.
                *val = Some(Knowledge { fact: new_fact, new_block: None });
                true
            },
            Some(Knowledge { ref mut fact, ref mut new_block }) => {
                let join = T::Lattice::join(fact, new_fact);
                // In case of some prior knowledge and provided our knowledge changed, we must
                // invalidate any new_block that could exist.
                if join {
                    *new_block = None;
                }
                join
            }
        }
    }

    fn replace_block(&mut self, bb: BasicBlock, block_data: BasicBlockData<'tcx>) {
        match self.knowledge[bb] {
            ref mut v@None => *v = Some(Knowledge {
                fact: T::Lattice::bottom(),
                new_block: Some(block_data)
            }),
            Some(Knowledge { ref mut new_block, .. }) => *new_block = Some(block_data),
        }
    }

    fn construct_mir(mut self) -> Mir<'tcx> {
        let mut new_blocks = IndexVec::with_capacity(self.mir.basic_blocks().len());
        for (block, old_data) in self.mir.basic_blocks().iter_enumerated() {
            let new_block = ::std::mem::replace(&mut self.knowledge[block], None)
                            .and_then(|k| k.new_block);
            new_blocks.push(if let Some(new_data) = new_block {
                new_data
            } else {
                old_data.clone()
            });
        }
        Mir::new(
            new_blocks,
            self.mir.visibility_scopes.clone(),
            self.mir.promoted.clone(),
            self.mir.return_ty,
            self.mir.var_decls.clone(),
            self.mir.arg_decls.clone(),
            self.mir.temp_decls.clone(),
            self.mir.upvar_decls.clone(),
            self.mir.span
        )
    }
}

fn mir_exits<'tcx>(mir: &Mir<'tcx>, exits: &mut BitVector) {
    // Do this smartly. First of all, find all the nodes without successors (these are guaranteed
    // exit nodes).
    let mir_len = mir.basic_blocks().len();
    let mut lead_to_exit = BitVector::new(mir_len);
    for (block, block_data) in mir.basic_blocks().iter_enumerated() {
        if block_data.terminator().successors().len() == 0 {
            exits.insert(block.index());
            lead_to_exit.insert(block.index());
            // Then, all blocks which have a path to these nodes, are not exit nodes by definition.
            mark_all_paths_to_with(block, &mut lead_to_exit, mir, BitVector::insert);
        }
    }
    // All unmarked blocks, provided there’s no unreachable blocks in the graph, must be blocks
    // belonging to a `loop {}` of some sort and thus each of such structure should have one of
    // their block as an exit node. Optimal exit node for such a structure is the one containing a
    // backedge (i.e. the `}` of `loop {}`). Finding such backedge might appear to be easy, but
    // corner cases like
    //
    // ```
    // 'a: loop { loop { if x { continue 'a } } }
    // ```
    //
    // make it considerably more complex. In the end, it doesn’t matter very much which node we
    // pick here, as picking any node does not make analysis incorrect, and a bad choice will only
    // result in dataflow doing an extra pass over some of the blocks.
    lead_to_exit.invert();
    while let Some(exit) = lead_to_exit.pop() {
        if exit >= mir_len { continue }
        exits.insert(exit);
        mark_all_paths_to_with(BasicBlock::new(exit), &mut lead_to_exit, mir, BitVector::remove);
    }
}

fn mark_all_paths_to_with<'tcx, F>(block: BasicBlock, mask: &mut BitVector, mir: &Mir<'tcx>, f: F)
where F: Copy + Fn(&mut BitVector, usize) -> bool
{
    for &predecessor in mir.predecessors_for(block).iter() {
        if f(mask, predecessor.index()) {
            mark_all_paths_to_with(predecessor, mask, mir, f);
        }
    }
}
