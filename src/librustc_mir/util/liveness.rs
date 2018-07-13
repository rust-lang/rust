// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Liveness analysis which computes liveness of MIR local variables at the boundary of basic blocks
//!
//! This analysis considers references as being used only at the point of the
//! borrow. This means that this does not track uses because of references that
//! already exist:
//!
//! ```Rust
//!     fn foo() {
//!         x = 0;
//!         // `x` is live here
//!         GLOBAL = &x: *const u32;
//!         // but not here, even while it can be accessed through `GLOBAL`.
//!         foo();
//!         x = 1;
//!         // `x` is live again here, because it is assigned to `OTHER_GLOBAL`
//!         OTHER_GLOBAL = &x: *const u32;
//!         // ...
//!     }
//! ```
//!
//! This means that users of this analysis still have to check whether
//! pre-existing references can be used to access the value (e.g. at movable
//! generator yield points, all pre-existing references are invalidated, so this
//! doesn't matter).

use rustc::mir::*;
use rustc::mir::visit::{PlaceContext, Visitor};
use rustc_data_structures::indexed_vec::{Idx, IndexVec};
use rustc_data_structures::indexed_set::IdxSetBuf;
use rustc_data_structures::work_queue::WorkQueue;
use util::pretty::{dump_enabled, write_basic_block, write_mir_intro};
use rustc::ty::item_path;
use rustc::mir::Local;
use rustc::mir::visit::MirVisitable;
use std::path::{Path, PathBuf};
use std::fs;
use rustc::ty::TyCtxt;
use std::io::{self, Write};
use transform::MirSource;

pub type LocalSet<V: LiveVariableMap> = IdxSetBuf<V::LiveVar>;

/// This gives the result of the liveness analysis at the boundary of
/// basic blocks. You can use `simulate_block` to obtain the
/// intra-block results.
pub struct LivenessResult<V: LiveVariableMap> {
    /// Liveness mode in use when these results were computed.
    pub mode: LivenessMode,

    /// Live variables on exit to each basic block. This is equal to
    /// the union of the `ins` for each successor.
    pub outs: IndexVec<BasicBlock, LocalSet<V::LiveVar>>,
}

pub(crate) trait LiveVariableMap {
    type LiveVar;

    fn from_local(&self, local: Local) -> Option<Self::LiveVar>;
    fn from_live_var(&self, local: Self::LiveVar) -> Local;
}

#[derive(Eq, PartialEq, Clone)]
struct IdentityMap;

impl LiveVariableMap for IdentityMap {
    type LiveVar = Local;

    fn from_local(&self, local: Local) -> Option<Self::LiveVar> {
        Some(local)
    }

    fn from_live_var(&self, local: Self::LiveVar) -> Local {
        local
    }
}

#[derive(Copy, Clone, Debug)]
pub struct LivenessMode {
    /// If true, then we will consider "regular uses" of a variable to be live.
    /// For example, if the user writes `foo(x)`, then this is a regular use of
    /// the variable `x`.
    pub include_regular_use: bool,

    /// If true, then we will consider (implicit) drops of a variable
    /// to be live.  For example, if the user writes `{ let x =
    /// vec![...]; .. }`, then the drop at the end of the block is an
    /// implicit drop.
    ///
    /// NB. Despite its name, a call like `::std::mem::drop(x)` is
    /// **not** considered a drop for this purposes, but rather a
    /// regular use.
    pub include_drops: bool,
}

/// A combination of liveness results, used in NLL.
pub struct LivenessResults<V> {
    /// Liveness results where a regular use makes a variable X live,
    /// but not a drop.
    pub regular: LivenessResult<V>,

    /// Liveness results where a drop makes a variable X live,
    /// but not a regular use.
    pub drop: LivenessResult<V>,
}

impl<V, M: LiveVariableMap<LiveVar = V>> LivenessResults<V> {
    pub fn compute<'tcx>(mir: &Mir<'tcx>, map: &M) -> LivenessResults<V> {
        LivenessResults {
            regular: liveness_of_locals(
                &mir,
                LivenessMode {
                    include_regular_use: true,
                    include_drops: false,
                },
            ),

            drop: liveness_of_locals(
                &mir,
                LivenessMode {
                    include_regular_use: false,
                    include_drops: true,
                },
            ),
        }
    }
}

/// Compute which local variables are live within the given function
/// `mir`. The liveness mode `mode` determines what sorts of uses are
/// considered to make a variable live (e.g., do drops count?).
pub fn liveness_of_locals<'tcx, V>(mir: &Mir<'tcx>, mode: LivenessMode) -> LivenessResult<V> {
    let locals = mir.local_decls.len();
    let def_use: IndexVec<_, _> = mir.basic_blocks()
        .iter()
        .map(|b| block(mode, b, locals))
        .collect();

    let mut outs: IndexVec<_, _> = mir.basic_blocks()
        .indices()
        .map(|_| LocalSet::new_empty(locals))
        .collect();

    let mut bits = LocalSet::new_empty(locals);

    // queue of things that need to be re-processed, and a set containing
    // the things currently in the queue
    let mut dirty_queue: WorkQueue<BasicBlock> = WorkQueue::with_all(mir.basic_blocks().len());

    let predecessors = mir.predecessors();

    while let Some(bb) = dirty_queue.pop() {
        // bits = use ∪ (bits - def)
        bits.overwrite(&outs[bb]);
        def_use[bb].apply(&mut bits);

        // `bits` now contains the live variables on entry. Therefore,
        // add `bits` to the `out` set for each predecessor; if those
        // bits were not already present, then enqueue the predecessor
        // as dirty.
        //
        // (note that `union` returns true if the `self` set changed)
        for &pred_bb in &predecessors[bb] {
            if outs[pred_bb].union(&bits) {
                dirty_queue.insert(pred_bb);
            }
        }
    }

    LivenessResult { mode, outs }
}

impl<V: LiveVariableMap> LivenessResult<V>
{
    /// Walks backwards through the statements/terminator in the given
    /// basic block `block`.  At each point within `block`, invokes
    /// the callback `op` with the current location and the set of
    /// variables that are live on entry to that location.
    pub fn simulate_block<'tcx, OP>(&self, mir: &Mir<'tcx>, block: BasicBlock, mut callback: OP)
    where
        OP: FnMut(Location, &LocalSet<V>),
    {
        let data = &mir[block];

        // Get a copy of the bits on exit from the block.
        let mut bits = self.outs[block].clone();

        // Start with the maximal statement index -- i.e., right before
        // the terminator executes.
        let mut statement_index = data.statements.len();

        // Compute liveness right before terminator and invoke callback.
        let terminator_location = Location {
            block,
            statement_index,
        };
        let locals = mir.local_decls.len();
        let mut visitor = DefsUsesVisitor {
            mode: self.mode,
            defs_uses: DefsUses {
                defs: LocalSet::new_empty(locals),
                uses: LocalSet::new_empty(locals),
                map: &IdentityMap {},
            },
        };
        // Visit the various parts of the basic block in reverse. If we go
        // forward, the logic in `add_def` and `add_use` would be wrong.
        visitor.update_bits_and_do_callback(terminator_location, &data.terminator, &mut bits,
                                            &mut callback);

        // Compute liveness before each statement (in rev order) and invoke callback.
        for statement in data.statements.iter().rev() {
            statement_index -= 1;
            let statement_location = Location {
                block,
                statement_index,
            };
            visitor.defs_uses.clear();
            visitor.update_bits_and_do_callback(statement_location, statement, &mut bits,
                                                &mut callback);
        }
    }
}

#[derive(Eq, PartialEq, Clone)]
pub enum DefUse {
    Def,
    Use,
}

pub fn categorize<'tcx>(context: PlaceContext<'tcx>, mode: LivenessMode) -> Option<DefUse> {
    match context {
        ///////////////////////////////////////////////////////////////////////////
        // DEFS

        PlaceContext::Store |

        // This is potentially both a def and a use...
        PlaceContext::AsmOutput |

        // We let Call define the result in both the success and
        // unwind cases. This is not really correct, however it
        // does not seem to be observable due to the way that we
        // generate MIR. See the test case
        // `mir-opt/nll/liveness-call-subtlety.rs`. To do things
        // properly, we would apply the def in call only to the
        // input from the success path and not the unwind
        // path. -nmatsakis
        PlaceContext::Call |

        // Storage live and storage dead aren't proper defines, but we can ignore
        // values that come before them.
        PlaceContext::StorageLive |
        PlaceContext::StorageDead => Some(DefUse::Def),

        ///////////////////////////////////////////////////////////////////////////
        // REGULAR USES
        //
        // These are uses that occur *outside* of a drop. For the
        // purposes of NLL, these are special in that **all** the
        // lifetimes appearing in the variable must be live for each regular use.

        PlaceContext::Projection(..) |

        // Borrows only consider their local used at the point of the borrow.
        // This won't affect the results since we use this analysis for generators
        // and we only care about the result at suspension points. Borrows cannot
        // cross suspension points so this behavior is unproblematic.
        PlaceContext::Borrow { .. } |

        PlaceContext::Inspect |
        PlaceContext::Copy |
        PlaceContext::Move |
        PlaceContext::Validate => {
            if mode.include_regular_use {
                Some(DefUse::Use)
            } else {
                None
            }
        }

        ///////////////////////////////////////////////////////////////////////////
        // DROP USES
        //
        // These are uses that occur in a DROP (a MIR drop, not a
        // call to `std::mem::drop()`). For the purposes of NLL,
        // uses in drop are special because `#[may_dangle]`
        // attributes can affect whether lifetimes must be live.

        PlaceContext::Drop => {
            if mode.include_drops {
                Some(DefUse::Use)
            } else {
                None
            }
        }
    }
}

struct DefsUsesVisitor<'lv> {
    mode: LivenessMode,
    defs_uses: DefsUses<'lv>,
}

#[derive(Eq, PartialEq, Clone)]
struct DefsUses<'lv>
{
    defs: LocalSet<Local>,
    uses: LocalSet<Local>,
    map: &'lv dyn LiveVariableMap<LiveVar=Local>,
}

impl<'lv> DefsUses<'lv> {
    fn clear(&mut self) {
        self.uses.clear();
        self.defs.clear();
    }

    fn apply(&self, bits: &mut LocalSet<Local>) -> bool {
        bits.subtract(&self.defs) | bits.union(&self.uses)
    }

    fn add_def(&mut self, index: Local) {
        // If it was used already in the block, remove that use
        // now that we found a definition.
        //
        // Example:
        //
        //     // Defs = {X}, Uses = {}
        //     X = 5
        //     // Defs = {}, Uses = {X}
        //     use(X)
        if let Some(v_index) = self.map.from_local(index) {
            self.uses.remove(&v_index);
            self.defs.add(&v_index);
        }
    }

    fn add_use(&mut self, index: Local) {
        // Inverse of above.
        //
        // Example:
        //
        //     // Defs = {}, Uses = {X}
        //     use(X)
        //     // Defs = {X}, Uses = {}
        //     X = 5
        //     // Defs = {}, Uses = {X}
        //     use(X)
        if let Some(v_index) = self.map.from_local(index) {
            self.defs.remove(&v_index);
            self.uses.add(&v_index);
        }
    }
}

impl<'lv> DefsUsesVisitor<'lv>
{
    /// Update `bits` with the effects of `value` and call `callback`. We
    /// should always visit in reverse order. This method assumes that we have
    /// not visited anything before; if you have, clear `bits` first.
    fn update_bits_and_do_callback<'tcx, OP>(&mut self, location: Location,
                                             value: &impl MirVisitable<'tcx>, bits: &mut LocalSet<Local>,
                                             callback: &mut OP)
    where
        OP: FnMut(Location, &LocalSet<Local>),
    {
        value.apply(location, self);
        self.defs_uses.apply(bits);
        callback(location, bits);
    }
}

impl<'tcx, 'lv> Visitor<'tcx> for DefsUsesVisitor<'lv> {
    fn visit_local(&mut self, &local: &Local, context: PlaceContext<'tcx>, _: Location) {
        match categorize(context, self.mode) {
            Some(DefUse::Def) => {
                self.defs_uses.add_def(local);
            }

            Some(DefUse::Use) => {
                self.defs_uses.add_use(local);
            }

            None => {}
        }
    }
}

fn block<'tcx, 'lv>(mode: LivenessMode, b: &BasicBlockData<'tcx>, locals: usize) -> DefsUses<'lv> {
    let mut visitor = DefsUsesVisitor {
        mode,
        defs_uses: DefsUses {
            defs: LocalSet::new_empty(locals),
            uses: LocalSet::new_empty(locals),
            map: &IdentityMap {},
        },
    };

    let dummy_location = Location {
        block: BasicBlock::new(0),
        statement_index: 0,
    };

    // Visit the various parts of the basic block in reverse. If we go
    // forward, the logic in `add_def` and `add_use` would be wrong.
    visitor.visit_terminator(BasicBlock::new(0), b.terminator(), dummy_location);
    for statement in b.statements.iter().rev() {
        visitor.visit_statement(BasicBlock::new(0), statement, dummy_location);
    }

    visitor.defs_uses
}

pub fn dump_mir<'a, 'tcx, V: LiveVariableMap>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    pass_name: &str,
    source: MirSource,
    mir: &Mir<'tcx>,
    result: &LivenessResult<Local>,
    map: &impl LiveVariableMap<LiveVar = V>
) {
    if !dump_enabled(tcx, pass_name, source) {
        return;
    }
    let node_path = item_path::with_forced_impl_filename_line(|| {
        // see notes on #41697 below
        tcx.item_path_str(source.def_id)
    });
    dump_matched_mir_node(tcx, pass_name, &node_path, source, mir, result);
}

fn dump_matched_mir_node<'a, 'tcx, V: LiveVariableMap>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    pass_name: &str,
    node_path: &str,
    source: MirSource,
    mir: &Mir<'tcx>,
    result: &LivenessResult<V::LiveVar>,
) {
    let mut file_path = PathBuf::new();
    file_path.push(Path::new(&tcx.sess.opts.debugging_opts.dump_mir_dir));
    let item_id = tcx.hir.as_local_node_id(source.def_id).unwrap();
    let file_name = format!("rustc.node{}{}-liveness.mir", item_id, pass_name);
    file_path.push(&file_name);
    let _ = fs::File::create(&file_path).and_then(|mut file| {
        writeln!(file, "// MIR local liveness analysis for `{}`", node_path)?;
        writeln!(file, "// source = {:?}", source)?;
        writeln!(file, "// pass_name = {}", pass_name)?;
        writeln!(file, "")?;
        write_mir_fn(tcx, source, mir, &mut file, result)?;
        Ok(())
    });
}

pub fn write_mir_fn<'a, 'tcx, V: LiveVariableMap>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    src: MirSource,
    mir: &Mir<'tcx>,
    w: &mut dyn Write,
    result: &LivenessResult<V::LiveVar>,
) -> io::Result<()> {
    write_mir_intro(tcx, src, mir, w)?;
    for block in mir.basic_blocks().indices() {
        let print = |w: &mut dyn Write, prefix, result: &IndexVec<BasicBlock, LocalSet<Local>>| {
            let live: Vec<String> = mir.local_decls
                .indices()
                .filter(|i| result[block].contains(i))
                .map(|i| format!("{:?}", i))
                .collect();
            writeln!(w, "{} {{{}}}", prefix, live.join(", "))
        };
        write_basic_block(tcx, block, mir, &mut |_, _| Ok(()), w)?;
        print(w, "   ", &result.outs)?;
        if block.index() + 1 != mir.basic_blocks().len() {
            writeln!(w, "")?;
        }
    }

    writeln!(w, "}}")?;
    Ok(())
}
