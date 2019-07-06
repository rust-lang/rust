use syntax::ast::{self, MetaItem};
use syntax::symbol::{Symbol, sym};

use rustc_data_structures::bit_set::{BitSet, HybridBitSet};
use rustc_data_structures::indexed_vec::Idx;
use rustc_data_structures::work_queue::WorkQueue;

use rustc::hir::def_id::DefId;
use rustc::ty::{self, TyCtxt};
use rustc::mir::{self, Body, BasicBlock, BasicBlockData, Location, Statement, Terminator};
use rustc::mir::traversal;
use rustc::session::Session;

use std::borrow::Borrow;
use std::fmt;
use std::io;
use std::path::PathBuf;
use std::usize;

pub use self::impls::{MaybeStorageLive, RequiresStorage};
pub use self::impls::{MaybeInitializedPlaces, MaybeUninitializedPlaces};
pub use self::impls::DefinitelyInitializedPlaces;
pub use self::impls::EverInitializedPlaces;
pub use self::impls::borrows::Borrows;
pub use self::impls::HaveBeenBorrowedLocals;
pub use self::at_location::{FlowAtLocation, FlowsAtLocation};
pub(crate) use self::drop_flag_effects::*;

use self::move_paths::MoveData;

mod at_location;
pub mod drop_flag_effects;
mod graphviz;
mod impls;
pub mod move_paths;

pub(crate) mod indexes {
    pub(crate) use super::{
        move_paths::{MovePathIndex, MoveOutIndex, InitIndex},
        impls::borrows::BorrowIndex,
    };
}

pub(crate) struct DataflowBuilder<'a, 'tcx, BD>
where
    BD: BitDenotation<'tcx>,
{
    def_id: DefId,
    flow_state: DataflowAnalysis<'a, 'tcx, BD>,
    print_preflow_to: Option<String>,
    print_postflow_to: Option<String>,
}

/// `DebugFormatted` encapsulates the "{:?}" rendering of some
/// arbitrary value. This way: you pay cost of allocating an extra
/// string (as well as that of rendering up-front); in exchange, you
/// don't have to hand over ownership of your value or deal with
/// borrowing it.
pub(crate) struct DebugFormatted(String);

impl DebugFormatted {
    pub fn new(input: &dyn fmt::Debug) -> DebugFormatted {
        DebugFormatted(format!("{:?}", input))
    }
}

impl fmt::Debug for DebugFormatted {
    fn fmt(&self, w: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(w, "{}", self.0)
    }
}

pub(crate) trait Dataflow<'tcx, BD: BitDenotation<'tcx>> {
    /// Sets up and runs the dataflow problem, using `p` to render results if
    /// implementation so chooses.
    fn dataflow<P>(&mut self, p: P) where P: Fn(&BD, BD::Idx) -> DebugFormatted {
        let _ = p; // default implementation does not instrument process.
        self.build_sets();
        self.propagate();
    }

    /// Sets up the entry, gen, and kill sets for this instance of a dataflow problem.
    fn build_sets(&mut self);

    /// Finds a fixed-point solution to this instance of a dataflow problem.
    fn propagate(&mut self);
}

impl<'a, 'tcx, BD> Dataflow<'tcx, BD> for DataflowBuilder<'a, 'tcx, BD>
where
    BD: BitDenotation<'tcx>,
{
    fn dataflow<P>(&mut self, p: P) where P: Fn(&BD, BD::Idx) -> DebugFormatted {
        self.flow_state.build_sets();
        self.pre_dataflow_instrumentation(|c,i| p(c,i)).unwrap();
        self.flow_state.propagate();
        self.post_dataflow_instrumentation(|c,i| p(c,i)).unwrap();
    }

    fn build_sets(&mut self) { self.flow_state.build_sets(); }
    fn propagate(&mut self) { self.flow_state.propagate(); }
}

pub(crate) fn has_rustc_mir_with(attrs: &[ast::Attribute], name: Symbol) -> Option<MetaItem> {
    for attr in attrs {
        if attr.check_name(sym::rustc_mir) {
            let items = attr.meta_item_list();
            for item in items.iter().flat_map(|l| l.iter()) {
                match item.meta_item() {
                    Some(mi) if mi.check_name(name) => return Some(mi.clone()),
                    _ => continue
                }
            }
        }
    }
    return None;
}

pub struct MoveDataParamEnv<'tcx> {
    pub(crate) move_data: MoveData<'tcx>,
    pub(crate) param_env: ty::ParamEnv<'tcx>,
}

pub(crate) fn do_dataflow<'a, 'tcx, BD, P>(
    tcx: TyCtxt<'tcx>,
    body: &'a Body<'tcx>,
    def_id: DefId,
    attributes: &[ast::Attribute],
    dead_unwinds: &BitSet<BasicBlock>,
    bd: BD,
    p: P,
) -> DataflowResults<'tcx, BD>
where
    BD: BitDenotation<'tcx>,
    P: Fn(&BD, BD::Idx) -> DebugFormatted,
{
    let flow_state = DataflowAnalysis::new(body, dead_unwinds, bd);
    flow_state.run(tcx, def_id, attributes, p)
}

impl<'a, 'tcx, BD> DataflowAnalysis<'a, 'tcx, BD>
where
    BD: BitDenotation<'tcx>,
{
    pub(crate) fn run<P>(
        self,
        tcx: TyCtxt<'tcx>,
        def_id: DefId,
        attributes: &[ast::Attribute],
        p: P,
    ) -> DataflowResults<'tcx, BD>
    where
        P: Fn(&BD, BD::Idx) -> DebugFormatted,
    {
        let name_found = |sess: &Session, attrs: &[ast::Attribute], name| -> Option<String> {
            if let Some(item) = has_rustc_mir_with(attrs, name) {
                if let Some(s) = item.value_str() {
                    return Some(s.to_string())
                } else {
                    sess.span_err(
                        item.span,
                        &format!("{} attribute requires a path", item.path));
                    return None;
                }
            }
            return None;
        };

        let print_preflow_to = name_found(tcx.sess, attributes, sym::borrowck_graphviz_preflow);
        let print_postflow_to = name_found(tcx.sess, attributes, sym::borrowck_graphviz_postflow);

        let mut mbcx = DataflowBuilder {
            def_id,
            print_preflow_to, print_postflow_to, flow_state: self,
        };

        mbcx.dataflow(p);
        mbcx.flow_state.results()
    }
}

struct PropagationContext<'b, 'a, 'tcx, O>
where
    O: BitDenotation<'tcx>,
{
    builder: &'b mut DataflowAnalysis<'a, 'tcx, O>,
}

impl<'a, 'tcx, BD> DataflowAnalysis<'a, 'tcx, BD>
where
    BD: BitDenotation<'tcx>,
{
    fn propagate(&mut self) {
        let mut temp = BitSet::new_empty(self.flow_state.sets.bits_per_block);
        let mut propcx = PropagationContext {
            builder: self,
        };
        propcx.walk_cfg(&mut temp);
    }

    fn build_sets(&mut self) {
        // Build the transfer function for each block.
        for (bb, data) in self.body.basic_blocks().iter_enumerated() {
            let &mir::BasicBlockData { ref statements, ref terminator, is_cleanup: _ } = data;

            let trans = self.flow_state.sets.trans_mut_for(bb.index());
            for j_stmt in 0..statements.len() {
                let location = Location { block: bb, statement_index: j_stmt };
                self.flow_state.operator.before_statement_effect(trans, location);
                self.flow_state.operator.statement_effect(trans, location);
            }

            if terminator.is_some() {
                let location = Location { block: bb, statement_index: statements.len() };
                self.flow_state.operator.before_terminator_effect(trans, location);
                self.flow_state.operator.terminator_effect(trans, location);
            }
        }

        // Initialize the flow state at entry to the start block.
        let on_entry = self.flow_state.sets.entry_set_mut_for(mir::START_BLOCK.index());
        self.flow_state.operator.start_block_effect(on_entry);
    }
}

impl<'b, 'a, 'tcx, BD> PropagationContext<'b, 'a, 'tcx, BD>
where
    BD: BitDenotation<'tcx>,
{
    fn walk_cfg(&mut self, in_out: &mut BitSet<BD::Idx>) {
        let body = self.builder.body;

        // Initialize the dirty queue in reverse post-order. This makes it more likely that the
        // entry state for each basic block will have the effects of its predecessors applied
        // before it is processed. In fact, for CFGs without back edges, this guarantees that
        // dataflow will converge in exactly `N` iterations, where `N` is the number of basic
        // blocks.
        let mut dirty_queue: WorkQueue<mir::BasicBlock> =
            WorkQueue::with_none(body.basic_blocks().len());
        for (bb, _) in traversal::reverse_postorder(body) {
            dirty_queue.insert(bb);
        }

        // Add blocks which are not reachable from START_BLOCK to the work queue. These blocks will
        // be processed after the ones added above.
        for bb in body.basic_blocks().indices() {
            dirty_queue.insert(bb);
        }

        while let Some(bb) = dirty_queue.pop() {
            let (on_entry, trans) = self.builder.flow_state.sets.get_mut(bb.index());
            debug_assert!(in_out.words().len() == on_entry.words().len());
            in_out.overwrite(on_entry);
            trans.apply(in_out);

            let bb_data = &body[bb];
            self.builder.propagate_bits_into_graph_successors_of(
                in_out, (bb, bb_data), &mut dirty_queue);
        }
    }
}

fn dataflow_path(context: &str, path: &str) -> PathBuf {
    let mut path = PathBuf::from(path);
    let new_file_name = {
        let orig_file_name = path.file_name().unwrap().to_str().unwrap();
        format!("{}_{}", context, orig_file_name)
    };
    path.set_file_name(new_file_name);
    path
}

impl<'a, 'tcx, BD> DataflowBuilder<'a, 'tcx, BD>
where
    BD: BitDenotation<'tcx>,
{
    fn pre_dataflow_instrumentation<P>(&self, p: P) -> io::Result<()>
        where P: Fn(&BD, BD::Idx) -> DebugFormatted
    {
        if let Some(ref path_str) = self.print_preflow_to {
            let path = dataflow_path(BD::name(), path_str);
            graphviz::print_borrowck_graph_to(self, &path, p)
        } else {
            Ok(())
        }
    }

    fn post_dataflow_instrumentation<P>(&self, p: P) -> io::Result<()>
        where P: Fn(&BD, BD::Idx) -> DebugFormatted
    {
        if let Some(ref path_str) = self.print_postflow_to {
            let path = dataflow_path(BD::name(), path_str);
            graphviz::print_borrowck_graph_to(self, &path, p)
        } else {
            Ok(())
        }
    }
}

/// DataflowResultsConsumer abstracts over walking the MIR with some
/// already constructed dataflow results.
///
/// It abstracts over the FlowState and also completely hides the
/// underlying flow analysis results, because it needs to handle cases
/// where we are combining the results of *multiple* flow analyses
/// (e.g., borrows + inits + uninits).
pub(crate) trait DataflowResultsConsumer<'a, 'tcx: 'a> {
    type FlowState: FlowsAtLocation;

    // Observation Hooks: override (at least one of) these to get analysis feedback.
    fn visit_block_entry(&mut self,
                         _bb: BasicBlock,
                         _flow_state: &Self::FlowState) {}

    fn visit_statement_entry(&mut self,
                             _loc: Location,
                             _stmt: &Statement<'tcx>,
                             _flow_state: &Self::FlowState) {}

    fn visit_terminator_entry(&mut self,
                              _loc: Location,
                              _term: &Terminator<'tcx>,
                              _flow_state: &Self::FlowState) {}

    // Main entry point: this drives the processing of results.

    fn analyze_results(&mut self, flow_uninit: &mut Self::FlowState) {
        let flow = flow_uninit;
        for (bb, _) in traversal::reverse_postorder(self.body()) {
            flow.reset_to_entry_of(bb);
            self.process_basic_block(bb, flow);
        }
    }

    fn process_basic_block(&mut self, bb: BasicBlock, flow_state: &mut Self::FlowState) {
        let BasicBlockData { ref statements, ref terminator, is_cleanup: _ } =
            self.body()[bb];
        let mut location = Location { block: bb, statement_index: 0 };
        for stmt in statements.iter() {
            flow_state.reconstruct_statement_effect(location);
            self.visit_statement_entry(location, stmt, flow_state);
            flow_state.apply_local_effect(location);
            location.statement_index += 1;
        }

        if let Some(ref term) = *terminator {
            flow_state.reconstruct_terminator_effect(location);
            self.visit_terminator_entry(location, term, flow_state);

            // We don't need to apply the effect of the terminator,
            // since we are only visiting dataflow state on control
            // flow entry to the various nodes. (But we still need to
            // reconstruct the effect, because the visit method might
            // inspect it.)
        }
    }

    // Delegated Hooks: Provide access to the MIR and process the flow state.

    fn body(&self) -> &'a Body<'tcx>;
}

/// Allows iterating dataflow results in a flexible and reasonably fast way.
pub struct DataflowResultsCursor<'mir, 'tcx, BD, DR = DataflowResults<'tcx, BD>>
where
    BD: BitDenotation<'tcx>,
    DR: Borrow<DataflowResults<'tcx, BD>>,
{
    flow_state: FlowAtLocation<'tcx, BD, DR>,

    // The statement (or terminator) whose effect has been reconstructed in
    // flow_state.
    curr_loc: Option<Location>,

    body: &'mir Body<'tcx>,
}

pub type DataflowResultsRefCursor<'mir, 'tcx, BD> =
    DataflowResultsCursor<'mir, 'tcx, BD, &'mir DataflowResults<'tcx, BD>>;

impl<'mir, 'tcx, BD, DR> DataflowResultsCursor<'mir, 'tcx, BD, DR>
where
    BD: BitDenotation<'tcx>,
    DR: Borrow<DataflowResults<'tcx, BD>>,
{
    pub fn new(result: DR, body: &'mir Body<'tcx>) -> Self {
        DataflowResultsCursor {
            flow_state: FlowAtLocation::new(result),
            curr_loc: None,
            body,
        }
    }

    /// Seek to the given location in MIR. This method is fast if you are
    /// traversing your MIR statements in order.
    ///
    /// After calling `seek`, the current state will reflect all effects up to
    /// and including the `before_statement_effect` of the statement at location
    /// `loc`. The `statement_effect` of the statement at `loc` will be
    /// available as the current effect (see e.g. `each_gen_bit`).
    ///
    /// If `loc.statement_index` equals the number of statements in the block,
    /// we will reconstruct the terminator effect in the same way as described
    /// above.
    pub fn seek(&mut self, loc: Location) {
        if self.curr_loc.map(|cur| loc == cur).unwrap_or(false) {
            return;
        }

        let start_index;
        let should_reset = match self.curr_loc {
            None => true,
            Some(cur)
                if loc.block != cur.block || loc.statement_index < cur.statement_index => true,
            _ => false,
        };
        if should_reset {
            self.flow_state.reset_to_entry_of(loc.block);
            start_index = 0;
        } else {
            let curr_loc = self.curr_loc.unwrap();
            start_index = curr_loc.statement_index;
            // Apply the effect from the last seek to the current state.
            self.flow_state.apply_local_effect(curr_loc);
        }

        for stmt in start_index..loc.statement_index {
            let mut stmt_loc = loc;
            stmt_loc.statement_index = stmt;
            self.flow_state.reconstruct_statement_effect(stmt_loc);
            self.flow_state.apply_local_effect(stmt_loc);
        }

        if loc.statement_index == self.body[loc.block].statements.len() {
            self.flow_state.reconstruct_terminator_effect(loc);
        } else {
            self.flow_state.reconstruct_statement_effect(loc);
        }
        self.curr_loc = Some(loc);
    }

    /// Return whether the current state contains bit `x`.
    pub fn contains(&self, x: BD::Idx) -> bool {
        self.flow_state.contains(x)
    }

    /// Iterate over each `gen` bit in the current effect (invoke `seek` first).
    pub fn each_gen_bit<F>(&self, f: F)
    where
        F: FnMut(BD::Idx),
    {
        self.flow_state.each_gen_bit(f)
    }
}

pub fn state_for_location<'tcx, T: BitDenotation<'tcx>>(loc: Location,
                                                        analysis: &T,
                                                        result: &DataflowResults<'tcx, T>,
                                                        body: &Body<'tcx>)
    -> BitSet<T::Idx> {
    let mut trans = GenKill::from_elem(HybridBitSet::new_empty(analysis.bits_per_block()));

    for stmt in 0..loc.statement_index {
        let mut stmt_loc = loc;
        stmt_loc.statement_index = stmt;
        analysis.before_statement_effect(&mut trans, stmt_loc);
        analysis.statement_effect(&mut trans, stmt_loc);
    }

    // Apply the pre-statement effect of the statement we're evaluating.
    if loc.statement_index == body[loc.block].statements.len() {
        analysis.before_terminator_effect(&mut trans, loc);
    } else {
        analysis.before_statement_effect(&mut trans, loc);
    }

    // Apply the transfer function for all preceding statements to the fixpoint
    // at the start of the block.
    let mut state = result.sets().entry_set_for(loc.block.index()).to_owned();
    trans.apply(&mut state);
    state
}

pub struct DataflowAnalysis<'a, 'tcx, O>
where
    O: BitDenotation<'tcx>,
{
    flow_state: DataflowState<'tcx, O>,
    dead_unwinds: &'a BitSet<mir::BasicBlock>,
    body: &'a Body<'tcx>,
}

impl<'a, 'tcx, O> DataflowAnalysis<'a, 'tcx, O>
where
    O: BitDenotation<'tcx>,
{
    pub fn results(self) -> DataflowResults<'tcx, O> {
        DataflowResults(self.flow_state)
    }

    pub fn body(&self) -> &'a Body<'tcx> { self.body }
}

pub struct DataflowResults<'tcx, O>(pub(crate) DataflowState<'tcx, O>) where O: BitDenotation<'tcx>;

impl<'tcx, O: BitDenotation<'tcx>> DataflowResults<'tcx, O> {
    pub fn sets(&self) -> &AllSets<O::Idx> {
        &self.0.sets
    }

    pub fn operator(&self) -> &O {
        &self.0.operator
    }
}

/// State of a dataflow analysis; couples a collection of bit sets
/// with operator used to initialize and merge bits during analysis.
pub struct DataflowState<'tcx, O: BitDenotation<'tcx>>
{
    /// All the sets for the analysis. (Factored into its
    /// own structure so that we can borrow it mutably
    /// on its own separate from other fields.)
    pub sets: AllSets<O::Idx>,

    /// operator used to initialize, combine, and interpret bits.
    pub(crate) operator: O,
}

impl<'tcx, O: BitDenotation<'tcx>> DataflowState<'tcx, O> {
    pub(crate) fn interpret_set<'c, P>(&self,
                                       o: &'c O,
                                       set: &BitSet<O::Idx>,
                                       render_idx: &P)
                                       -> Vec<DebugFormatted>
        where P: Fn(&O, O::Idx) -> DebugFormatted
    {
        set.iter().map(|i| render_idx(o, i)).collect()
    }

    pub(crate) fn interpret_hybrid_set<'c, P>(&self,
                                              o: &'c O,
                                              set: &HybridBitSet<O::Idx>,
                                              render_idx: &P)
                                              -> Vec<DebugFormatted>
        where P: Fn(&O, O::Idx) -> DebugFormatted
    {
        set.iter().map(|i| render_idx(o, i)).collect()
    }
}

/// A 2-tuple representing the "gen" and "kill" bitsets during
/// dataflow analysis.
///
/// It is best to ensure that the intersection of `gen_set` and
/// `kill_set` is empty; otherwise the results of the dataflow will
/// have a hidden dependency on what order the bits are generated and
/// killed during the iteration. (This is such a good idea that the
/// `fn gen` and `fn kill` methods that set their state enforce this
/// for you.)
#[derive(Debug, Clone, Copy)]
pub struct GenKill<T> {
    pub(crate) gen_set: T,
    pub(crate) kill_set: T,
}

type GenKillSet<T> = GenKill<HybridBitSet<T>>;

impl<T> GenKill<T> {
    /// Creates a new tuple where `gen_set == kill_set == elem`.
    pub(crate) fn from_elem(elem: T) -> Self
        where T: Clone
    {
        GenKill {
            gen_set: elem.clone(),
            kill_set: elem,
        }
    }
}

impl<E:Idx> GenKillSet<E> {
    pub(crate) fn clear(&mut self) {
        self.gen_set.clear();
        self.kill_set.clear();
    }

    fn gen(&mut self, e: E) {
        self.gen_set.insert(e);
        self.kill_set.remove(e);
    }
    fn gen_all<I>(&mut self, i: I)
        where I: IntoIterator,
              I::Item: Borrow<E>
    {
        for j in i {
            self.gen(*j.borrow());
        }
    }

    fn kill(&mut self, e: E) {
        self.gen_set.remove(e);
        self.kill_set.insert(e);
    }

    fn kill_all<I>(&mut self, i: I)
        where I: IntoIterator,
              I::Item: Borrow<E>
    {
        for j in i {
            self.kill(*j.borrow());
        }
    }

    /// Computes `(set ∪ gen) - kill` and assigns the result to `set`.
    pub(crate) fn apply(&self, set: &mut BitSet<E>) {
        set.union(&self.gen_set);
        set.subtract(&self.kill_set);
    }
}

#[derive(Debug)]
pub struct AllSets<E: Idx> {
    /// Analysis bitwidth for each block.
    bits_per_block: usize,

    /// For each block, bits valid on entry to the block.
    on_entry: Vec<BitSet<E>>,

    /// The transfer function of each block expressed as the set of bits
    /// generated and killed by executing the statements + terminator in the
    /// block -- with one caveat. In particular, for *call terminators*, the
    /// effect of storing the destination is not included, since that only takes
    /// effect on the **success** edge (and not the unwind edge).
    trans: Vec<GenKillSet<E>>,
}

impl<E:Idx> AllSets<E> {
    pub fn bits_per_block(&self) -> usize { self.bits_per_block }

    pub fn get_mut(&mut self, block_idx: usize) -> (&mut BitSet<E>, &mut GenKillSet<E>) {
        (&mut self.on_entry[block_idx], &mut self.trans[block_idx])
    }

    pub fn trans_for(&self, block_idx: usize) -> &GenKillSet<E> {
        &self.trans[block_idx]
    }
    pub fn trans_mut_for(&mut self, block_idx: usize) -> &mut GenKillSet<E> {
        &mut self.trans[block_idx]
    }
    pub fn entry_set_for(&self, block_idx: usize) -> &BitSet<E> {
        &self.on_entry[block_idx]
    }
    pub fn entry_set_mut_for(&mut self, block_idx: usize) -> &mut BitSet<E> {
        &mut self.on_entry[block_idx]
    }
    pub fn gen_set_for(&self, block_idx: usize) -> &HybridBitSet<E> {
        &self.trans_for(block_idx).gen_set
    }
    pub fn kill_set_for(&self, block_idx: usize) -> &HybridBitSet<E> {
        &self.trans_for(block_idx).kill_set
    }
}

/// Parameterization for the precise form of data flow that is used.
///
/// `BottomValue` determines whether the initial entry set for each basic block is empty or full.
/// This also determines the semantics of the lattice `join` operator used to merge dataflow
/// results, since dataflow works by starting at the bottom and moving monotonically to a fixed
/// point.
///
/// This means, for propagation across the graph, that you either want to start at all-zeroes and
/// then use Union as your merge when propagating, or you start at all-ones and then use Intersect
/// as your merge when propagating.
pub trait BottomValue {
    /// Specifies the initial value for each bit in the entry set for each basic block.
    const BOTTOM_VALUE: bool;

    /// Merges `in_set` into `inout_set`, returning `true` if `inout_set` changed.
    #[inline]
    fn join<T: Idx>(&self, inout_set: &mut BitSet<T>, in_set: &BitSet<T>) -> bool {
        if Self::BOTTOM_VALUE == false {
            inout_set.union(in_set)
        } else {
            inout_set.intersect(in_set)
        }
    }
}

/// A specific flavor of dataflow analysis.
///
/// To run a dataflow analysis, one sets up an initial state for the
/// `START_BLOCK` via `start_block_effect` and a transfer function (`trans`)
/// for each block individually. The entry set for all other basic blocks is
/// initialized to `Self::BOTTOM_VALUE`. The dataflow analysis then
/// iteratively modifies the various entry sets (but leaves the the transfer
/// function unchanged).
pub trait BitDenotation<'tcx>: BottomValue {
    /// Specifies what index type is used to access the bitvector.
    type Idx: Idx;

    /// A name describing the dataflow analysis that this
    /// `BitDenotation` is supporting. The name should be something
    /// suitable for plugging in as part of a filename (i.e., avoid
    /// space-characters or other things that tend to look bad on a
    /// file system, like slashes or periods). It is also better for
    /// the name to be reasonably short, again because it will be
    /// plugged into a filename.
    fn name() -> &'static str;

    /// Size of each bitvector allocated for each block in the analysis.
    fn bits_per_block(&self) -> usize;

    /// Mutates the entry set according to the effects that
    /// have been established *prior* to entering the start
    /// block. This can't access the gen/kill sets, because
    /// these won't be accounted for correctly.
    ///
    /// (For example, establishing the call arguments.)
    fn start_block_effect(&self, entry_set: &mut BitSet<Self::Idx>);

    /// Similar to `statement_effect`, except it applies
    /// *just before* the statement rather than *just after* it.
    ///
    /// This matters for "dataflow at location" APIs, because the
    /// before-statement effect is visible while visiting the
    /// statement, while the after-statement effect only becomes
    /// visible at the next statement.
    ///
    /// Both the before-statement and after-statement effects are
    /// applied, in that order, before moving for the next
    /// statement.
    fn before_statement_effect(&self,
                               _trans: &mut GenKillSet<Self::Idx>,
                               _location: Location) {}

    /// Mutates the block-sets (the flow sets for the given
    /// basic block) according to the effects of evaluating statement.
    ///
    /// This is used, in particular, for building up the
    /// "transfer-function" representing the overall-effect of the
    /// block, represented via GEN and KILL sets.
    ///
    /// The statement is identified as `bb_data[idx_stmt]`, where
    /// `bb_data` is the sequence of statements identified by `bb` in
    /// the MIR.
    fn statement_effect(&self,
                        trans: &mut GenKillSet<Self::Idx>,
                        location: Location);

    /// Similar to `terminator_effect`, except it applies
    /// *just before* the terminator rather than *just after* it.
    ///
    /// This matters for "dataflow at location" APIs, because the
    /// before-terminator effect is visible while visiting the
    /// terminator, while the after-terminator effect only becomes
    /// visible at the terminator's successors.
    ///
    /// Both the before-terminator and after-terminator effects are
    /// applied, in that order, before moving for the next
    /// terminator.
    fn before_terminator_effect(&self,
                                _trans: &mut GenKillSet<Self::Idx>,
                                _location: Location) {}

    /// Mutates the block-sets (the flow sets for the given
    /// basic block) according to the effects of evaluating
    /// the terminator.
    ///
    /// This is used, in particular, for building up the
    /// "transfer-function" representing the overall-effect of the
    /// block, represented via GEN and KILL sets.
    ///
    /// The effects applied here cannot depend on which branch the
    /// terminator took.
    fn terminator_effect(&self,
                         trans: &mut GenKillSet<Self::Idx>,
                         location: Location);

    /// Mutates the block-sets according to the (flow-dependent)
    /// effect of a successful return from a Call terminator.
    ///
    /// If basic-block BB_x ends with a call-instruction that, upon
    /// successful return, flows to BB_y, then this method will be
    /// called on the exit flow-state of BB_x in order to set up the
    /// entry flow-state of BB_y.
    ///
    /// This is used, in particular, as a special case during the
    /// "propagate" loop where all of the basic blocks are repeatedly
    /// visited. Since the effects of a Call terminator are
    /// flow-dependent, the current MIR cannot encode them via just
    /// GEN and KILL sets attached to the block, and so instead we add
    /// this extra machinery to represent the flow-dependent effect.
    //
    // FIXME: right now this is a bit of a wart in the API. It might
    // be better to represent this as an additional gen- and
    // kill-sets associated with each edge coming out of the basic
    // block.
    fn propagate_call_return(
        &self,
        in_out: &mut BitSet<Self::Idx>,
        call_bb: mir::BasicBlock,
        dest_bb: mir::BasicBlock,
        dest_place: &mir::Place<'tcx>,
    );
}

impl<'a, 'tcx, D> DataflowAnalysis<'a, 'tcx, D> where D: BitDenotation<'tcx>
{
    pub fn new(body: &'a Body<'tcx>,
               dead_unwinds: &'a BitSet<mir::BasicBlock>,
               denotation: D) -> Self {
        let bits_per_block = denotation.bits_per_block();
        let num_blocks = body.basic_blocks().len();

        let on_entry = if D::BOTTOM_VALUE == true {
            vec![BitSet::new_filled(bits_per_block); num_blocks]
        } else {
            vec![BitSet::new_empty(bits_per_block); num_blocks]
        };
        let nop = GenKill::from_elem(HybridBitSet::new_empty(bits_per_block));

        DataflowAnalysis {
            body,
            dead_unwinds,
            flow_state: DataflowState {
                sets: AllSets {
                    bits_per_block,
                    on_entry,
                    trans: vec![nop; num_blocks],
                },
                operator: denotation,
            }
        }
    }
}

impl<'a, 'tcx, D> DataflowAnalysis<'a, 'tcx, D>
where
    D: BitDenotation<'tcx>,
{
    /// Propagates the bits of `in_out` into all the successors of `bb`,
    /// using bitwise operator denoted by `self.operator`.
    ///
    /// For most blocks, this is entirely uniform. However, for blocks
    /// that end with a call terminator, the effect of the call on the
    /// dataflow state may depend on whether the call returned
    /// successfully or unwound.
    ///
    /// To reflect this, the `propagate_call_return` method of the
    /// `BitDenotation` mutates `in_out` when propagating `in_out` via
    /// a call terminator; such mutation is performed *last*, to
    /// ensure its side-effects do not leak elsewhere (e.g., into
    /// unwind target).
    fn propagate_bits_into_graph_successors_of(
        &mut self,
        in_out: &mut BitSet<D::Idx>,
        (bb, bb_data): (mir::BasicBlock, &mir::BasicBlockData<'tcx>),
        dirty_list: &mut WorkQueue<mir::BasicBlock>)
    {
        match bb_data.terminator().kind {
            mir::TerminatorKind::Return |
            mir::TerminatorKind::Resume |
            mir::TerminatorKind::Abort |
            mir::TerminatorKind::GeneratorDrop |
            mir::TerminatorKind::Unreachable => {}
            mir::TerminatorKind::Goto { target } |
            mir::TerminatorKind::Assert { target, cleanup: None, .. } |
            mir::TerminatorKind::Yield { resume: target, drop: None, .. } |
            mir::TerminatorKind::Drop { target, location: _, unwind: None } |
            mir::TerminatorKind::DropAndReplace {
                target, value: _, location: _, unwind: None
            } => {
                self.propagate_bits_into_entry_set_for(in_out, target, dirty_list);
            }
            mir::TerminatorKind::Yield { resume: target, drop: Some(drop), .. } => {
                self.propagate_bits_into_entry_set_for(in_out, target, dirty_list);
                self.propagate_bits_into_entry_set_for(in_out, drop, dirty_list);
            }
            mir::TerminatorKind::Assert { target, cleanup: Some(unwind), .. } |
            mir::TerminatorKind::Drop { target, location: _, unwind: Some(unwind) } |
            mir::TerminatorKind::DropAndReplace {
                target, value: _, location: _, unwind: Some(unwind)
            } => {
                self.propagate_bits_into_entry_set_for(in_out, target, dirty_list);
                if !self.dead_unwinds.contains(bb) {
                    self.propagate_bits_into_entry_set_for(in_out, unwind, dirty_list);
                }
            }
            mir::TerminatorKind::SwitchInt { ref targets, .. } => {
                for target in targets {
                    self.propagate_bits_into_entry_set_for(in_out, *target, dirty_list);
                }
            }
            mir::TerminatorKind::Call { cleanup, ref destination, .. } => {
                if let Some(unwind) = cleanup {
                    if !self.dead_unwinds.contains(bb) {
                        self.propagate_bits_into_entry_set_for(in_out, unwind, dirty_list);
                    }
                }
                if let Some((ref dest_place, dest_bb)) = *destination {
                    // N.B.: This must be done *last*, after all other
                    // propagation, as documented in comment above.
                    self.flow_state.operator.propagate_call_return(
                        in_out, bb, dest_bb, dest_place);
                    self.propagate_bits_into_entry_set_for(in_out, dest_bb, dirty_list);
                }
            }
            mir::TerminatorKind::FalseEdges { real_target, imaginary_target } => {
                self.propagate_bits_into_entry_set_for(in_out, real_target, dirty_list);
                self.propagate_bits_into_entry_set_for(in_out, imaginary_target, dirty_list);
            }
            mir::TerminatorKind::FalseUnwind { real_target, unwind } => {
                self.propagate_bits_into_entry_set_for(in_out, real_target, dirty_list);
                if let Some(unwind) = unwind {
                    if !self.dead_unwinds.contains(bb) {
                        self.propagate_bits_into_entry_set_for(in_out, unwind, dirty_list);
                    }
                }
            }
        }
    }

    fn propagate_bits_into_entry_set_for(&mut self,
                                         in_out: &BitSet<D::Idx>,
                                         bb: mir::BasicBlock,
                                         dirty_queue: &mut WorkQueue<mir::BasicBlock>) {
        let entry_set = self.flow_state.sets.entry_set_mut_for(bb.index());
        let set_changed = self.flow_state.operator.join(entry_set, &in_out);
        if set_changed {
            dirty_queue.insert(bb);
        }
    }
}
