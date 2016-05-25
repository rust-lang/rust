use mir::repr as mir;
use mir::cfg::CFG;
use mir::repr::{BasicBlock, START_BLOCK};
use rustc_data_structures::bitvec::BitVector;

use mir::transform::lattice::Lattice;

pub trait Rewrite<'tcx, L: Lattice> {
    /// The rewrite function which given a statement optionally produces an alternative graph to be
    /// placed in place of the original statement.
    ///
    /// The 2nd BasicBlock *MUST NOT* have the terminator set.
    ///
    /// Correctness precondition:
    /// * transfer_stmt(statement, fact) == transfer_stmt(rewrite_stmt(statement, fact))
    /// that is, given some fact `fact` true before both the statement and relacement graph, and
    /// a fact `fact2` which is true after the statement, the same `fact2` must be true after the
    /// replacement graph too.
    fn stmt(&self, &mir::Statement<'tcx>, &L, &mut CFG<'tcx>) -> StatementChange<'tcx>;

    /// The rewrite function which given a terminator optionally produces an alternative graph to
    /// be placed in place of the original statement.
    ///
    /// The 2nd BasicBlock *MUST* have the terminator set.
    ///
    /// Correctness precondition:
    /// * transfer_stmt(terminator, fact) == transfer_stmt(rewrite_term(terminator, fact))
    /// that is, given some fact `fact` true before both the terminator and relacement graph, and
    /// a fact `fact2` which is true after the statement, the same `fact2` must be true after the
    /// replacement graph too.
    fn term(&self, &mir::Terminator<'tcx>, &L, &mut CFG<'tcx>) -> TerminatorChange<'tcx>;

    fn and_then<R2>(self, other: R2) -> RewriteAndThen<Self, R2> where Self: Sized {
        RewriteAndThen(self, other)
    }
}

/// This combinator has the following behaviour:
///
/// * Rewrite the node with the first rewriter.
///   * if the first rewriter replaced the node, 2nd rewriter is used to rewrite the replacement.
///   * otherwise 2nd rewriter is used to rewrite the original node.
pub struct RewriteAndThen<R1, R2>(R1, R2);
impl<'tcx, L, R1, R2> Rewrite<'tcx, L> for RewriteAndThen<R1, R2>
where L: Lattice, R1: Rewrite<'tcx, L>, R2: Rewrite<'tcx, L> {
    fn stmt(&self, s: &mir::Statement<'tcx>, l: &L, c: &mut CFG<'tcx>) -> StatementChange<'tcx> {
        let rs = self.0.stmt(s, l, c);
        match rs {
            StatementChange::None => self.1.stmt(s, l, c),
            StatementChange::Remove => StatementChange::Remove,
            StatementChange::Statement(ns) =>
                match self.1.stmt(&ns, l, c) {
                    StatementChange::None => StatementChange::Statement(ns),
                    x => x
                }
        }
    }

    fn term(&self, t: &mir::Terminator<'tcx>, l: &L, c: &mut CFG<'tcx>) -> TerminatorChange<'tcx> {
        let rt = self.0.term(t, l, c);
        match rt {
            TerminatorChange::None => self.1.term(t, l, c),
            TerminatorChange::Terminator(nt) => match self.1.term(&nt, l, c) {
                TerminatorChange::None => TerminatorChange::Terminator(nt),
                x => x
            }
        }
    }
}

pub enum TerminatorChange<'tcx> {
    /// No change
    None,
    /// Replace with another terminator
    Terminator(mir::Terminator<'tcx>),
}

pub enum StatementChange<'tcx> {
    /// No change
    None,
    /// Remove the statement
    Remove,
    /// Replace with another single statement
    Statement(mir::Statement<'tcx>),
}

pub trait Transfer<'tcx> {
    type Lattice: Lattice;

    /// The transfer function which given a statement and a fact produces a fact which is true
    /// after the statement.
    fn stmt(&self, &mir::Statement<'tcx>, Self::Lattice) -> Self::Lattice;

    /// The transfer function which given a terminator and a fact produces a fact for each
    /// successor of the terminator.
    ///
    /// Corectness precondtition:
    /// * The list of facts produced should only contain the facts for blocks which are successors
    /// of the terminator being transfered.
    fn term(&self, &mir::Terminator<'tcx>, Self::Lattice) -> Vec<Self::Lattice>;
}


/// Facts is a mapping from basic block label (index) to the fact which is true about the first
/// statement in the block.
pub struct Facts<F>(Vec<F>);

impl<F: Lattice> Facts<F> {
    pub fn new() -> Facts<F> {
        Facts(vec![])
    }

    fn put(&mut self, index: BasicBlock, value: F) {
        let len = self.0.len();
        self.0.extend((len...index.index()).map(|_| <F as Lattice>::bottom()));
        self.0[index.index()] = value;
    }
}

impl<F: Lattice> ::std::ops::Index<BasicBlock> for Facts<F> {
    type Output = F;
    fn index(&self, index: BasicBlock) -> &F {
        &self.0.get(index.index()).expect("facts indexed immutably and the user is buggy!")
    }
}

impl<F: Lattice> ::std::ops::IndexMut<BasicBlock> for Facts<F> {
    fn index_mut(&mut self, index: BasicBlock) -> &mut F {
        if self.0.get(index.index()).is_none() {
            self.put(index, <F as Lattice>::bottom());
        }
        self.0.get_mut(index.index()).unwrap()
    }
}

/// Analyse and rewrite using dataflow in the forward direction
pub fn ar_forward<'tcx, T, R>(cfg: &CFG<'tcx>, fs: Facts<T::Lattice>, transfer: T, rewrite: R)
-> (CFG<'tcx>, Facts<T::Lattice>)
where T: Transfer<'tcx>,
      R: Rewrite<'tcx, T::Lattice>
{
    let mut queue = BitVector::new(cfg.len());
    queue.insert(START_BLOCK.index());

    fixpoint(cfg, Direction::Forward, |bb, fact, cfg| {
        let new_graph = cfg.start_new_block();
        let mut fact = fact.clone();
        let mut changed = false;
        // Swap out the vector of old statements for a duration of statement inspection.
        let old_statements = ::std::mem::replace(&mut cfg[bb].statements, Vec::new());
        for stmt in &old_statements {
            // Given a fact and statement produce a new fact and optionally a replacement
            // graph.
            match rewrite.stmt(&stmt, &fact, cfg) {
                StatementChange::None => {
                    fact = transfer.stmt(stmt, fact);
                    cfg.push(new_graph, stmt.clone());
                }
                StatementChange::Remove => changed = true,
                StatementChange::Statement(stmt) => {
                    changed = true;
                    fact = transfer.stmt(&stmt, fact);
                    cfg.push(new_graph, stmt);
                }
            }
        }
        // Swap the statements back in.
        cfg[bb].statements = old_statements;

        // Handle the terminator replacement and transfer.
        let terminator = cfg[bb].terminator.take().unwrap();
        let repl = rewrite.term(&terminator, &fact, cfg);
        match repl {
            TerminatorChange::None => {
                cfg[new_graph].terminator = Some(terminator.clone());
            }
            TerminatorChange::Terminator(t) => {
                changed = true;
                cfg[new_graph].terminator = Some(t);
            }
        }
        let new_facts = transfer.term(cfg[new_graph].terminator(), fact);
        cfg[bb].terminator = Some(terminator);

        (if changed { Some(new_graph) } else { None }, new_facts)
    }, &mut queue, fs)
}

// /// The implementation of forward dataflow.
// pub struct ForwardDataflow<F>(::std::marker::PhantomData<F>);
//
// impl<F> ForwardDataflow<F> {
//     pub fn new() -> ForwardDataflow<F> {
//         ForwardDataflow(::std::marker::PhantomData)
//     }
// }
//
// impl<F> Pass for ForwardDataflow<F> {}
//
// impl<'tcx, P> MirPass<'tcx> for ForwardDataflow<P>
// where P: DataflowPass<'tcx> {
//     fn run_pass<'a>(&mut self, _: TyCtxt<'a, 'tcx, 'tcx>, _: MirSource, mir: &mut mir::Mir<'tcx>) {
//         let facts: Facts<<P as DataflowPass<'tcx>>::Lattice> =
//             Facts::new(<P as DataflowPass<'tcx>>::input_fact());
//         let (new_cfg, _) = self.arf_body(&mir.cfg, facts, mir::START_BLOCK);
//         mir.cfg = new_cfg;
//     }
// }
//
// impl<'tcx, P> ForwardDataflow<P>
// where P: DataflowPass<'tcx> {
//
//
// /// The implementation of backward dataflow.
// pub struct BackwardDataflow<F>(::std::marker::PhantomData<F>);
//
// impl<F> BackwardDataflow<F> {
//     pub fn new() -> BackwardDataflow<F> {
//         BackwardDataflow(::std::marker::PhantomData)
//     }
// }
//
// impl<F> Pass for BackwardDataflow<F> {}
//
// impl<'tcx, P> MirPass<'tcx> for BackwardDataflow<P>
// where P: DataflowPass<'tcx> {
//     fn run_pass<'a>(&mut self, _: TyCtxt<'a, 'tcx, 'tcx>, _: MirSource, mir: &mut mir::Mir<'tcx>) {
//         let mut facts: Facts<<P as DataflowPass<'tcx>>::Lattice> =
//             Facts::new(<P as DataflowPass<'tcx>>::input_fact());
//         // The desired effect here is that we should begin flowing from the blocks which terminate
//         // the control flow (return, resume, calls of diverging functions, non-terminating loops
//         // etc), but finding them all is a pain, so we just get the list of graph nodes postorder
//         // and inspect them all! Perhaps not very effective, but certainly correct.
//         let start_at = postorder(mir).filter(|&(_, d)| !d.is_cleanup).map(|(bb, _)| {
//             facts.put(bb, <P as DataflowPass<'tcx>>::Lattice::bottom());
//             (bb, ())
//         }).collect();
//         let (new_cfg, _) = self.arb_body(&mir.cfg, facts, start_at);
//         mir.cfg = new_cfg;
//     }
// }
//
// impl<'tcx, P> BackwardDataflow<P>
// where P: DataflowPass<'tcx> {
//     fn arb_body(&self, cfg: &CFG<'tcx>,
//                 facts: Facts<<P as DataflowPass<'tcx>>::Lattice>, mut map: HashMap<BasicBlock, ()>)
//     -> (CFG<'tcx>, Facts<<P as DataflowPass<'tcx>>::Lattice>){
//         fixpoint(cfg, Direction::Backward, |bb, fact, cfg| {
//             let new_graph = cfg.start_new_block();
//             let mut fact = fact.clone();
//             // This is a reverse thing so we inspect the terminator first and statements in reverse
//             // order later.
//             //
//             // Handle the terminator replacement and transfer.
//             let terminator = ::std::mem::replace(&mut cfg[bb].terminator, None).unwrap();
//             let repl = P::rewrite_term(&terminator, &fact, cfg);
//             // TODO: this really needs to get factored out
//             let mut new_facts = match repl {
//                 TerminatorReplacement::Terminator(t) => {
//                     cfg[new_graph].terminator = Some(t);
//                     P::transfer_term(cfg[new_graph].terminator(), fact)
//                 }
//                 TerminatorReplacement::Graph(from, _) => {
//                     // FIXME: a more optimal approach would be to copy the from to the tail of our
//                     // new_graph. (1 less extra block). However there’s a problem with inspecting
//                     // the statements of the merged block, because we just did the statements
//                     // for this block already.
//                     cfg.terminate(new_graph, terminator.scope, terminator.span,
//                                   mir::TerminatorKind::Goto { target: from });
//                     P::transfer_term(&cfg[new_graph].terminator(), fact)
//                 }
//             };
//             // FIXME: this should just have a different API.
//             assert!(new_facts.len() == 1, "transfer_term function is incorrect");
//             fact = new_facts.pop().unwrap().1;
//             ::std::mem::replace(&mut cfg[bb].terminator, Some(terminator));
//
//             // Swap out the vector of old statements for a duration of statement inspection.
//             let old_statements = ::std::mem::replace(&mut cfg[bb].statements, Vec::new());
//             for stmt in old_statements.iter().rev() {
//                 // Given a fact and statement produce a new fact and optionally a replacement
//                 // graph.
//                 let mut new_repl = P::rewrite_stmt(&stmt, &fact, cfg);
//                 new_repl.normalise();
//                 match new_repl {
//                     StatementReplacement::None => {}
//                     StatementReplacement::Statement(nstmt) => {
//                         fact = P::transfer_stmt(&nstmt, fact);
//                         cfg.push(new_graph, nstmt)
//                     }
//                     StatementReplacement::Statements(stmts) => {
//                         for stmt in &stmts {
//                             fact = P::transfer_stmt(stmt, fact);
//                         }
//                         cfg[new_graph].statements.extend(stmts)
//                     }
//                     StatementReplacement::Graph(..) => unimplemented!(),
//                     // debug_assert!(cfg[replacement.1].terminator.is_none(),
//                     //               "buggy pass: replacement tail has a terminator set!");
//                 };
//             }
//             // Reverse the statements, because we were analysing bottom-top but pusshing
//             // top-bottom.
//             cfg[new_graph].statements.reverse();
//             ::std::mem::replace(&mut cfg[bb].statements, old_statements);
//             (Some(new_graph), vec![(mir::START_BLOCK, fact)])
//         }, &mut map, facts)
//     }
// }

enum Direction {
    Forward,
//    Backward
}

/// The fixpoint function is the engine of this whole thing. Important part of it is the `f: BF`
/// callback. This for each basic block and its facts has to produce a replacement graph and a
/// bunch of facts which are to be joined with the facts in the graph elsewhere.
fn fixpoint<'tcx, F: Lattice, BF>(cfg: &CFG<'tcx>,
                                  direction: Direction,
                                  f: BF,
                                  to_visit: &mut BitVector,
                                  mut init_facts: Facts<F>) -> (CFG<'tcx>, Facts<F>)
// TODO: we probably want to pass in a list of basicblocks as successors (predecessors in backward
// fixpoing) and let BF return just a list of F.
where BF: Fn(BasicBlock, &F, &mut CFG<'tcx>) -> (Option<BasicBlock>, Vec<F>),
      // ^~ This function given a single block and fact before it optionally produces a replacement
      // graph (if not, the original block is the “replacement graph”) for the block and a list of
      // facts for arbitrary blocks (most likely for the blocks in the replacement graph and blocks
      // into which data flows from the replacement graph)
      //
      // Invariant:
      // * None of the already existing blocks in CFG may be modified;
{
    let mut cfg = cfg.clone();

    while let Some(block) = to_visit.iter().next() {
        to_visit.remove(block);
        let block = BasicBlock::new(block);

        let (new_target, new_facts) = {
            let fact = &mut init_facts[block];
            f(block, fact, &mut cfg)
        };

        // First of all, we merge in the replacement graph, if any.
        if let Some(replacement_bb) = new_target {
            cfg.swap(replacement_bb, block);
        }

        // Then we record the facts in the correct direction.
        match direction {
            Direction::Forward => {
                for (f, &target) in new_facts.into_iter()
                                             .zip(cfg[block].terminator().successors().iter()) {
                    let facts_changed = Lattice::join(&mut init_facts[target], &f);
                    if facts_changed {
                        to_visit.insert(target.index());
                    }
                }
            }
            // Direction::Backward => unimplemented!()
            // let mut new_facts = new_facts;
            // let fact = new_facts.pop().unwrap().1;
            // for pred in cfg.predecessors(block) {
            //     if init_facts.exists(pred) {
            //         let old_fact = &mut init_facts[pred];
            //         let facts_changed = Lattice::join(old_fact, &fact);
            //         if facts_changed {
            //             to_visit.insert(pred, ());
            //         }
            //     } else {
            //         init_facts.put(pred, fact.clone());
            //         to_visit.insert(pred, ());
            //     }
            // }
        }
    }
    (cfg, init_facts)
}
