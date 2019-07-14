use std::collections::BTreeMap;
use std::ops;

use rustc::mir::{self, BasicBlock, Local, Location, Place};
use rustc::mir::visit::{Visitor, MirVisitable};
use rustc::util::captures::Captures;
use rustc_data_structures::bit_set::BitSet;
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::indexed_vec::{Idx, IndexVec};

use crate::dataflow::{
    DataflowResults, DataflowResultsCursor,
    impls::HaveBeenBorrowedLocals,
    impls::reaching_defs::{DefIndex, DefKind, Definition, ReachingDefinitions, UseVisitor},
};

type LocationMap<T> = FxHashMap<Location, T>;
type DefsForLocalDense = IndexVec<Local, BitSet<DefIndex>>;
type DefsForLocalSparse = BTreeMap<Local, BitSet<DefIndex>>;
type DefsForUseAtLocation = LocationMap<DefsForLocalSparse>;

/// A data structure, built using the results of a `ReachingDefinitions` analysis, that maps each
/// use of a variable to the definitions of that variable that could reach it.
///
/// While `ReachingDefinitions` maps each point in the program to every definition which reaches
/// that point, a `UseDefChain` maps each use of a variable to every definition which could define
/// that variable. See below for an example where this distinction is meaningful.
///
/// ```rust,no_run
/// fn five() -> i32 {
///     let mut x = 0;  // (1)
///
///     let p = &mut x; // (2)
///     *p = 1;         // (3)
///
///     x = 2;          // (4)
///
///     x + 3
///     // At this point, a `UseDefChain` knows that (4) is the only possible definition of `x`,
///     // even though (2), (3) and (4) reach here. A reaching definitions analysis can kill (1)
///     // since it is superseded by (4), but it cannot kill (3) since `p` may be pointing
///     // to a local besides `x`. However, a `UseDefChain` is only interested in definitions of
///     // `x`, so it can kill *all* other definitions when (4) is encountered.
/// }
/// ```
///
/// This precision comes at the cost of increased memory usage; A `UseDefChain` stores a `BitSet`
/// for every use in the program. This allows consumers to traverse the graph of data dependencies
/// efficiently. For example, a `UseDefChain` could be used for const-propagation by iterating over
/// the definitions which could possibly reach `y` at the exit of the following function and
/// realizing that all of them are the constant `6`.
///
/// ```rust,no_run
/// fn six(condition: bool) -> i32 {
///     let mut y;
///     let a = 2 * 3;
///     let b = 10 - 4;
///
///     if condition {
///         y = a;
///     } else {
///         y = b;
///     }
///
///     y
/// }
/// ```
pub struct UseDefChain<'me, 'tcx> {
    reaching_defs: &'me DataflowResults<'tcx, ReachingDefinitions>,

    /// Maps each local that is used at a given location to the set of definitions which can reach
    /// it.
    defs_for_use: DefsForUseAtLocation,
}

impl<'me, 'tcx> UseDefChain<'me, 'tcx> {
    pub fn new(
        body: &'me mir::Body<'tcx>,
        reaching_defs: &'me DataflowResults<'tcx, ReachingDefinitions>,
        observed_addresses: &DataflowResults<'tcx, HaveBeenBorrowedLocals<'_, 'tcx>>,
    ) -> Self {
        let defs = reaching_defs.operator();
        let all_defs_for_local = all_defs_for_local(body, defs, &observed_addresses);
        let locals_used_in_block = LocalsUsedInBlock::new(body);
        let curr_defs_for_local =
            IndexVec::from_elem(BitSet::new_empty(defs.len()), &body.local_decls);

        let mut defs_for_use = DefsForUseAtLocation::default();

        {
            let mut builder = UseDefBuilder {
                body,
                ud_chain: &mut defs_for_use,
                reaching_defs,
                locals_used_in_block,
                curr_defs_for_local,
                all_defs_for_local,
            };

            builder.visit_body(body);
        }

        UseDefChain {
            defs_for_use,
            reaching_defs,
        }
    }

    pub fn defs_for_use(
        &self,
        local: Local,
        location: Location,
    ) -> impl Iterator<Item = &'_ Definition> + Captures<'tcx> + '_ {
        let UseDefChain { defs_for_use, reaching_defs, .. } = self;

        defs_for_use
            .get(&location)
            .and_then(|for_local| for_local.get(&local))
            .into_iter()
            .flat_map(|defs| defs.iter())
            .map(move |def| reaching_defs.operator().get(def))
    }
}

struct UseDefBuilder<'a, 'me, 'tcx> {
    ud_chain: &'a mut DefsForUseAtLocation,

    body: &'me mir::Body<'tcx>,

    reaching_defs: &'me DataflowResults<'tcx, ReachingDefinitions>,

    /// It's expensive to clone the entry set at the start of each block for every local and apply
    /// the appropriate mask. However, we only need to keep `curr_defs_for_local` up-to-date for
    /// the `Local`s which are actually used in a given `BasicBlock`.
    locals_used_in_block: LocalsUsedInBlock,

    /// The set of definitions which could have reached a given local at the program point we are
    /// currently processing.
    curr_defs_for_local: DefsForLocalDense,

    /// The set of all definitions which *might* define the given local. These definitions may or
    /// may not be live at the current program point.
    all_defs_for_local: DefsForLocalDense,
}

impl<'me, 'tcx> UseDefBuilder<'_, 'me, 'tcx> {
    fn visit_location(&mut self, location: Location, stmt_or_term: &impl MirVisitable<'tcx>) {
        let UseDefBuilder {
            all_defs_for_local,
            curr_defs_for_local,
            locals_used_in_block,
            reaching_defs,
            ud_chain,
            ..
        } = self;

        // If this is the start of a `BasicBlock`, reset `curr_defs_for_local` for each (tracked)
        // local to the set of definitions which reach the start of this block and can modify that
        // local.
        if location.statement_index == 0 {
            for local in locals_used_in_block[location.block].iter() {
                let entry_set = &reaching_defs.sets().entry_set_for(location.block.index());
                curr_defs_for_local[local].overwrite(entry_set);
                curr_defs_for_local[local].intersect(&all_defs_for_local[local]);
            }
        }

        // Save the current set of reaching definitions for any local which is used at this
        // location.
        let record_use = |place: &Place<'tcx>, location| {
            if let Some(local) = place.base_local() {
                ud_chain.entry(location)
                    .or_default()
                    .entry(local)
                    .or_insert_with(|| {
                        debug_assert!(locals_used_in_block[location.block].contains(local));
                        curr_defs_for_local[local].clone()
                    });
            }
        };

        stmt_or_term.apply(location, &mut UseVisitor(record_use));

        // Update `curr_defs_for_local` with any new definitions from this location.
        //
        // This needs to take place after processing uses, otherwise a def like `x =
        // x+1` will be marked as reaching its right hand side.
        let reaching_defs = reaching_defs.operator();
        let tracked_in_block = &locals_used_in_block[location.block];
        for def in reaching_defs.at_location(location) {
            match reaching_defs.get(def).kind {
                // No need to process defs for a local that is untracked
                DefKind::DirectWhole(local) | DefKind::DirectPart(local)
                    if !tracked_in_block.contains(local)
                    => continue,

                DefKind::DirectWhole(local) => {
                    let defs = &mut curr_defs_for_local[local];
                    defs.clear();
                    defs.insert(def);
                }

                DefKind::DirectPart(local) => {
                    curr_defs_for_local[local].insert(def);
                }

                DefKind::Indirect => {
                    self.body
                        .local_decls
                        .indices()
                        .filter(|&local| all_defs_for_local[local].contains(def))
                        .for_each(|local| { curr_defs_for_local[local].insert(def); });
                }
            }
        }
    }
}

impl<'tcx> Visitor<'tcx> for UseDefBuilder<'_, '_, 'tcx> {
    fn visit_statement(&mut self, statement: &mir::Statement<'tcx>, location: Location) {
        // Don't recurse: We only care about statements and terminators
        // self.super_statement(statement, location);

        self.visit_location(location, statement)
    }

    fn visit_terminator(&mut self,
                        terminator: &mir::Terminator<'tcx>,
                        location: Location) {
        // Don't recurse: We only care about statements and terminators
        // self.super_terminator(terminator, location);

        self.visit_location(location, terminator)
    }
}

/// The set of `Local`s which are used at least once in each basic block.
struct LocalsUsedInBlock(IndexVec<BasicBlock, BitSet<Local>>);

impl LocalsUsedInBlock {
    fn new(body: &mir::Body<'_>) -> Self {
        let each_block =
            IndexVec::from_elem(BitSet::new_empty(body.local_decls.len()), body.basic_blocks());
        let mut ret = LocalsUsedInBlock(each_block);

        UseVisitor(|place: &Place<'_>, loc| ret.visit_use(place, loc))
            .visit_body(body);

        ret
    }

    fn visit_use(&mut self, place: &mir::Place<'tcx>, loc: Location) {
        if let Some(local) = place.base_local() {
            self.0[loc.block].insert(local);
        }
    }
}

impl ops::Deref for LocalsUsedInBlock {
    type Target = IndexVec<BasicBlock, BitSet<Local>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// `BitSet` cannot efficiently `FromIterator` since it needs to know its size ahead of time.
fn bitset_from_iter<I: Idx>(size: usize, it: impl IntoIterator<Item = I>) -> BitSet<I> {
    let mut ret = BitSet::new_empty(size);
    for item in it {
        ret.insert(item);
    }

    ret
}

fn all_defs_for_local<'tcx>(
    body: &mir::Body<'tcx>,
    defs: &ReachingDefinitions,
    observed_addresses: &DataflowResults<'tcx, HaveBeenBorrowedLocals<'_, 'tcx>>,
) -> DefsForLocalDense {
    // Initialize the return value with the direct definitions of each local.
    let mut defs_for_local: DefsForLocalDense = body
        .local_decls
        .iter_enumerated()
        .map(|(local, _)| bitset_from_iter(defs.len(), defs.for_local_direct(local)))
        .collect();

    // Add each indirect definition to the set of definitions for each local whose address has been
    // observed at that point.
    let mut observed_addresses = DataflowResultsCursor::new(observed_addresses, body);
    for def_id in defs.indirect() {
        let def = defs.get(def_id);
        observed_addresses.seek(def.location.expect("Indirect def without location"));

        for local in observed_addresses.get().iter() {
            defs_for_local[local].insert(def_id);
        }
    }

    defs_for_local
}
