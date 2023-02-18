//! Propagates assignment destinations backwards in the CFG to eliminate redundant assignments.
//!
//! # Motivation
//!
//! MIR building can insert a lot of redundant copies, and Rust code in general often tends to move
//! values around a lot. The result is a lot of assignments of the form `dest = {move} src;` in MIR.
//! MIR building for constants in particular tends to create additional locals that are only used
//! inside a single block to shuffle a value around unnecessarily.
//!
//! LLVM by itself is not good enough at eliminating these redundant copies (eg. see
//! <https://github.com/rust-lang/rust/issues/32966>), so this leaves some performance on the table
//! that we can regain by implementing an optimization for removing these assign statements in rustc
//! itself. When this optimization runs fast enough, it can also speed up the constant evaluation
//! and code generation phases of rustc due to the reduced number of statements and locals.
//!
//! # The Optimization
//!
//! Conceptually, this optimization is "destination propagation". It is similar to the Named Return
//! Value Optimization, or NRVO, known from the C++ world, except that it isn't limited to return
//! values or the return place `_0`. On a very high level, independent of the actual implementation
//! details, it does the following:
//!
//! 1) Identify `dest = src;` statements with values for `dest` and `src` whose storage can soundly
//!    be merged.
//! 2) Replace all mentions of `src` with `dest` ("unifying" them and propagating the destination
//!    backwards).
//! 3) Delete the `dest = src;` statement (by making it a `nop`).
//!
//! Step 1) is by far the hardest, so it is explained in more detail below.
//!
//! ## Soundness
//!
//! We have a pair of places `p` and `q`, whose memory we would like to merge. In order for this to
//! be sound, we need to check a number of conditions:
//!
//! * `p` and `q` must both be *constant* - it does not make much sense to talk about merging them
//!   if they do not consistently refer to the same place in memory. This is satisfied if they do
//!   not contain any indirection through a pointer or any indexing projections.
//!
//! * We need to make sure that the goal of "merging the memory" is actually structurally possible
//!   in MIR. For example, even if all the other conditions are satisfied, there is no way to
//!   "merge" `_5.foo` and `_6.bar`. For now, we ensure this by requiring that both `p` and `q` are
//!   locals with no further projections. Future iterations of this pass should improve on this.
//!
//! * Finally, we want `p` and `q` to use the same memory - however, we still need to make sure that
//!   each of them has enough "ownership" of that memory to continue "doing its job." More
//!   precisely, what we will check is that whenever the program performs a write to `p`, then it
//!   does not currently care about what the value in `q` is (and vice versa). We formalize the
//!   notion of "does not care what the value in `q` is" by checking the *liveness* of `q`.
//!
//!   Because of the difficulty of computing liveness of places that have their address taken, we do
//!   not even attempt to do it. Any places that are in a local that has its address taken is
//!   excluded from the optimization.
//!
//! The first two conditions are simple structural requirements on the `Assign` statements that can
//! be trivially checked. The third requirement however is more difficult and costly to check.
//!
//! ## Future Improvements
//!
//! There are a number of ways in which this pass could be improved in the future:
//!
//! * Merging storage liveness ranges instead of removing storage statements completely. This may
//!   improve stack usage.
//!
//! * Allow merging locals into places with projections, eg `_5` into `_6.foo`.
//!
//! * Liveness analysis with more precision than whole locals at a time. The smaller benefit of this
//!   is that it would allow us to dest prop at "sub-local" levels in some cases. The bigger benefit
//!   of this is that such liveness analysis can report more accurate results about whole locals at
//!   a time. For example, consider:
//!
//!   ```ignore (syntax-highliting-only)
//!   _1 = u;
//!   // unrelated code
//!   _1.f1 = v;
//!   _2 = _1.f1;
//!   ```
//!
//!   Because the current analysis only thinks in terms of locals, it does not have enough
//!   information to report that `_1` is dead in the "unrelated code" section.
//!
//! * Liveness analysis enabled by alias analysis. This would allow us to not just bail on locals
//!   that ever have their address taken. Of course that requires actually having alias analysis
//!   (and a model to build it on), so this might be a bit of a ways off.
//!
//! * Various perf improvents. There are a bunch of comments in here marked `PERF` with ideas for
//!   how to do things more efficiently. However, the complexity of the pass as a whole should be
//!   kept in mind.
//!
//! ## Previous Work
//!
//! A [previous attempt][attempt 1] at implementing an optimization like this turned out to be a
//! significant regression in compiler performance. Fixing the regressions introduced a lot of
//! undesirable complexity to the implementation.
//!
//! A [subsequent approach][attempt 2] tried to avoid the costly computation by limiting itself to
//! acyclic CFGs, but still turned out to be far too costly to run due to suboptimal performance
//! within individual basic blocks, requiring a walk across the entire block for every assignment
//! found within the block. For the `tuple-stress` benchmark, which has 458745 statements in a
//! single block, this proved to be far too costly.
//!
//! [Another approach after that][attempt 3] was much closer to correct, but had some soundness
//! issues - it was failing to consider stores outside live ranges, and failed to uphold some of the
//! requirements that MIR has for non-overlapping places within statements. However, it also had
//! performance issues caused by `O(l² * s)` runtime, where `l` is the number of locals and `s` is
//! the number of statements and terminators.
//!
//! Since the first attempt at this, the compiler has improved dramatically, and new analysis
//! frameworks have been added that should make this approach viable without requiring a limited
//! approach that only works for some classes of CFGs:
//! - rustc now has a powerful dataflow analysis framework that can handle forwards and backwards
//!   analyses efficiently.
//! - Layout optimizations for generators have been added to improve code generation for
//!   async/await, which are very similar in spirit to what this optimization does.
//!
//! Also, rustc now has a simple NRVO pass (see `nrvo.rs`), which handles a subset of the cases that
//! this destination propagation pass handles, proving that similar optimizations can be performed
//! on MIR.
//!
//! ## Pre/Post Optimization
//!
//! It is recommended to run `SimplifyCfg` and then `SimplifyLocals` some time after this pass, as
//! it replaces the eliminated assign statements with `nop`s and leaves unused locals behind.
//!
//! [liveness]: https://en.wikipedia.org/wiki/Live_variable_analysis
//! [attempt 1]: https://github.com/rust-lang/rust/pull/47954
//! [attempt 2]: https://github.com/rust-lang/rust/pull/71003
//! [attempt 3]: https://github.com/rust-lang/rust/pull/72632

use std::collections::hash_map::{Entry, OccupiedEntry};

use crate::simplify::remove_dead_blocks;
use crate::MirPass;
use rustc_data_structures::fx::FxHashMap;
use rustc_index::bit_set::BitSet;
use rustc_middle::mir::visit::{MutVisitor, PlaceContext, Visitor};
use rustc_middle::mir::{dump_mir, PassWhere};
use rustc_middle::mir::{
    traversal, Body, InlineAsmOperand, Local, LocalKind, Location, Operand, Place, Rvalue,
    Statement, StatementKind, TerminatorKind,
};
use rustc_middle::ty::TyCtxt;
use rustc_mir_dataflow::impls::MaybeLiveLocals;
use rustc_mir_dataflow::{Analysis, ResultsCursor};

pub struct DestinationPropagation;

impl<'tcx> MirPass<'tcx> for DestinationPropagation {
    fn is_enabled(&self, sess: &rustc_session::Session) -> bool {
        // For now, only run at MIR opt level 3. Two things need to be changed before this can be
        // turned on by default:
        //  1. Because of the overeager removal of storage statements, this can cause stack space
        //     regressions. This opt is not the place to fix this though, it's a more general
        //     problem in MIR.
        //  2. Despite being an overall perf improvement, this still causes a 30% regression in
        //     keccak. We can temporarily fix this by bounding function size, but in the long term
        //     we should fix this by being smarter about invalidating analysis results.
        sess.mir_opt_level() >= 3
    }

    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        let def_id = body.source.def_id();
        let mut allocations = Allocations::default();
        trace!(func = ?tcx.def_path_str(def_id));

        let borrowed = rustc_mir_dataflow::impls::borrowed_locals(body);

        // In order to avoid having to collect data for every single pair of locals in the body, we
        // do not allow doing more than one merge for places that are derived from the same local at
        // once. To avoid missed opportunities, we instead iterate to a fixed point - we'll refer to
        // each of these iterations as a "round."
        //
        // Reaching a fixed point could in theory take up to `min(l, s)` rounds - however, we do not
        // expect to see MIR like that. To verify this, a test was run against `[rust-lang/regex]` -
        // the average MIR body saw 1.32 full iterations of this loop. The most that was hit were 30
        // for a single function. Only 80/2801 (2.9%) of functions saw at least 5.
        //
        // [rust-lang/regex]:
        //     https://github.com/rust-lang/regex/tree/b5372864e2df6a2f5e543a556a62197f50ca3650
        let mut round_count = 0;
        loop {
            // PERF: Can we do something smarter than recalculating the candidates and liveness
            // results?
            let mut candidates = find_candidates(
                body,
                &borrowed,
                &mut allocations.candidates,
                &mut allocations.candidates_reverse,
            );
            trace!(?candidates);
            let mut live = MaybeLiveLocals
                .into_engine(tcx, body)
                .iterate_to_fixpoint()
                .into_results_cursor(body);
            dest_prop_mir_dump(tcx, body, &mut live, round_count);

            FilterInformation::filter_liveness(
                &mut candidates,
                &mut live,
                &mut allocations.write_info,
                body,
            );

            // Because we do not update liveness information, it is unsound to use a local for more
            // than one merge operation within a single round of optimizations. We store here which
            // ones we have already used.
            let mut merged_locals: BitSet<Local> = BitSet::new_empty(body.local_decls.len());

            // This is the set of merges we will apply this round. It is a subset of the candidates.
            let mut merges = FxHashMap::default();

            for (src, candidates) in candidates.c.iter() {
                if merged_locals.contains(*src) {
                    continue;
                }
                let Some(dest) =
                    candidates.iter().find(|dest| !merged_locals.contains(**dest)) else {
                        continue;
                };
                if !tcx.consider_optimizing(|| {
                    format!("{} round {}", tcx.def_path_str(def_id), round_count)
                }) {
                    break;
                }
                merges.insert(*src, *dest);
                merged_locals.insert(*src);
                merged_locals.insert(*dest);
            }
            trace!(merging = ?merges);

            if merges.is_empty() {
                break;
            }
            round_count += 1;

            apply_merges(body, tcx, &merges, &merged_locals);
        }

        if round_count != 0 {
            // Merging can introduce overlap between moved arguments and/or call destination in an
            // unreachable code, which validator considers to be ill-formed.
            remove_dead_blocks(tcx, body);
        }

        trace!(round_count);
    }
}

/// Container for the various allocations that we need.
///
/// We store these here and hand out `&mut` access to them, instead of dropping and recreating them
/// frequently. Everything with a `&'alloc` lifetime points into here.
#[derive(Default)]
struct Allocations {
    candidates: FxHashMap<Local, Vec<Local>>,
    candidates_reverse: FxHashMap<Local, Vec<Local>>,
    write_info: WriteInfo,
    // PERF: Do this for `MaybeLiveLocals` allocations too.
}

#[derive(Debug)]
struct Candidates<'alloc> {
    /// The set of candidates we are considering in this optimization.
    ///
    /// We will always merge the key into at most one of its values.
    ///
    /// Whether a place ends up in the key or the value does not correspond to whether it appears as
    /// the lhs or rhs of any assignment. As a matter of fact, the places in here might never appear
    /// in an assignment at all. This happens because if we see an assignment like this:
    ///
    /// ```ignore (syntax-highlighting-only)
    /// _1.0 = _2.0
    /// ```
    ///
    /// We will still report that we would like to merge `_1` and `_2` in an attempt to allow us to
    /// remove that assignment.
    c: &'alloc mut FxHashMap<Local, Vec<Local>>,
    /// A reverse index of the `c` set; if the `c` set contains `a => Place { local: b, proj }`,
    /// then this contains `b => a`.
    // PERF: Possibly these should be `SmallVec`s?
    reverse: &'alloc mut FxHashMap<Local, Vec<Local>>,
}

//////////////////////////////////////////////////////////
// Merging
//
// Applies the actual optimization

fn apply_merges<'tcx>(
    body: &mut Body<'tcx>,
    tcx: TyCtxt<'tcx>,
    merges: &FxHashMap<Local, Local>,
    merged_locals: &BitSet<Local>,
) {
    let mut merger = Merger { tcx, merges, merged_locals };
    merger.visit_body_preserves_cfg(body);
}

struct Merger<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    merges: &'a FxHashMap<Local, Local>,
    merged_locals: &'a BitSet<Local>,
}

impl<'a, 'tcx> MutVisitor<'tcx> for Merger<'a, 'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn visit_local(&mut self, local: &mut Local, _: PlaceContext, _location: Location) {
        if let Some(dest) = self.merges.get(local) {
            *local = *dest;
        }
    }

    fn visit_statement(&mut self, statement: &mut Statement<'tcx>, location: Location) {
        match &statement.kind {
            // FIXME: Don't delete storage statements, but "merge" the storage ranges instead.
            StatementKind::StorageDead(local) | StatementKind::StorageLive(local)
                if self.merged_locals.contains(*local) =>
            {
                statement.make_nop();
                return;
            }
            _ => (),
        };
        self.super_statement(statement, location);
        match &statement.kind {
            StatementKind::Assign(box (dest, rvalue)) => {
                match rvalue {
                    Rvalue::CopyForDeref(place)
                    | Rvalue::Use(Operand::Copy(place) | Operand::Move(place)) => {
                        // These might've been turned into self-assignments by the replacement
                        // (this includes the original statement we wanted to eliminate).
                        if dest == place {
                            debug!("{:?} turned into self-assignment, deleting", location);
                            statement.make_nop();
                        }
                    }
                    _ => {}
                }
            }

            _ => {}
        }
    }
}

//////////////////////////////////////////////////////////
// Liveness filtering
//
// This section enforces bullet point 2

struct FilterInformation<'a, 'body, 'alloc, 'tcx> {
    body: &'body Body<'tcx>,
    live: &'a mut ResultsCursor<'body, 'tcx, MaybeLiveLocals>,
    candidates: &'a mut Candidates<'alloc>,
    write_info: &'alloc mut WriteInfo,
    at: Location,
}

// We first implement some utility functions which we will expose removing candidates according to
// different needs. Throughout the livenss filtering, the `candidates` are only ever accessed
// through these methods, and not directly.
impl<'alloc> Candidates<'alloc> {
    /// Just `Vec::retain`, but the condition is inverted and we add debugging output
    fn vec_filter_candidates(
        src: Local,
        v: &mut Vec<Local>,
        mut f: impl FnMut(Local) -> CandidateFilter,
        at: Location,
    ) {
        v.retain(|dest| {
            let remove = f(*dest);
            if remove == CandidateFilter::Remove {
                trace!("eliminating {:?} => {:?} due to conflict at {:?}", src, dest, at);
            }
            remove == CandidateFilter::Keep
        });
    }

    /// `vec_filter_candidates` but for an `Entry`
    fn entry_filter_candidates(
        mut entry: OccupiedEntry<'_, Local, Vec<Local>>,
        p: Local,
        f: impl FnMut(Local) -> CandidateFilter,
        at: Location,
    ) {
        let candidates = entry.get_mut();
        Self::vec_filter_candidates(p, candidates, f, at);
        if candidates.len() == 0 {
            entry.remove();
        }
    }

    /// For all candidates `(p, q)` or `(q, p)` removes the candidate if `f(q)` says to do so
    fn filter_candidates_by(
        &mut self,
        p: Local,
        mut f: impl FnMut(Local) -> CandidateFilter,
        at: Location,
    ) {
        // Cover the cases where `p` appears as a `src`
        if let Entry::Occupied(entry) = self.c.entry(p) {
            Self::entry_filter_candidates(entry, p, &mut f, at);
        }
        // And the cases where `p` appears as a `dest`
        let Some(srcs) = self.reverse.get_mut(&p) else {
            return;
        };
        // We use `retain` here to remove the elements from the reverse set if we've removed the
        // matching candidate in the forward set.
        srcs.retain(|src| {
            if f(*src) == CandidateFilter::Keep {
                return true;
            }
            let Entry::Occupied(entry) = self.c.entry(*src) else {
                return false;
            };
            Self::entry_filter_candidates(
                entry,
                *src,
                |dest| {
                    if dest == p { CandidateFilter::Remove } else { CandidateFilter::Keep }
                },
                at,
            );
            false
        });
    }
}

#[derive(Copy, Clone, PartialEq, Eq)]
enum CandidateFilter {
    Keep,
    Remove,
}

impl<'a, 'body, 'alloc, 'tcx> FilterInformation<'a, 'body, 'alloc, 'tcx> {
    /// Filters the set of candidates to remove those that conflict.
    ///
    /// The steps we take are exactly those that are outlined at the top of the file. For each
    /// statement/terminator, we collect the set of locals that are written to in that
    /// statement/terminator, and then we remove all pairs of candidates that contain one such local
    /// and another one that is live.
    ///
    /// We need to be careful about the ordering of operations within each statement/terminator
    /// here. Many statements might write and read from more than one place, and we need to consider
    /// them all. The strategy for doing this is as follows: We first gather all the places that are
    /// written to within the statement/terminator via `WriteInfo`. Then, we use the liveness
    /// analysis from *before* the statement/terminator (in the control flow sense) to eliminate
    /// candidates - this is because we want to conservatively treat a pair of locals that is both
    /// read and written in the statement/terminator to be conflicting, and the liveness analysis
    /// before the statement/terminator will correctly report locals that are read in the
    /// statement/terminator to be live. We are additionally conservative by treating all written to
    /// locals as also being read from.
    fn filter_liveness<'b>(
        candidates: &mut Candidates<'alloc>,
        live: &mut ResultsCursor<'b, 'tcx, MaybeLiveLocals>,
        write_info_alloc: &'alloc mut WriteInfo,
        body: &'b Body<'tcx>,
    ) {
        let mut this = FilterInformation {
            body,
            live,
            candidates,
            // We don't actually store anything at this scope, we just keep things here to be able
            // to reuse the allocation.
            write_info: write_info_alloc,
            // Doesn't matter what we put here, will be overwritten before being used
            at: Location::START,
        };
        this.internal_filter_liveness();
    }

    fn internal_filter_liveness(&mut self) {
        for (block, data) in traversal::preorder(self.body) {
            self.at = Location { block, statement_index: data.statements.len() };
            self.live.seek_after_primary_effect(self.at);
            self.write_info.for_terminator(&data.terminator().kind);
            self.apply_conflicts();

            for (i, statement) in data.statements.iter().enumerate().rev() {
                self.at = Location { block, statement_index: i };
                self.live.seek_after_primary_effect(self.at);
                self.write_info.for_statement(&statement.kind, self.body);
                self.apply_conflicts();
            }
        }
    }

    fn apply_conflicts(&mut self) {
        let writes = &self.write_info.writes;
        for p in writes {
            let other_skip = self.write_info.skip_pair.and_then(|(a, b)| {
                if a == *p {
                    Some(b)
                } else if b == *p {
                    Some(a)
                } else {
                    None
                }
            });
            self.candidates.filter_candidates_by(
                *p,
                |q| {
                    if Some(q) == other_skip {
                        return CandidateFilter::Keep;
                    }
                    // It is possible that a local may be live for less than the
                    // duration of a statement This happens in the case of function
                    // calls or inline asm. Because of this, we also mark locals as
                    // conflicting when both of them are written to in the same
                    // statement.
                    if self.live.contains(q) || writes.contains(&q) {
                        CandidateFilter::Remove
                    } else {
                        CandidateFilter::Keep
                    }
                },
                self.at,
            );
        }
    }
}

/// Describes where a statement/terminator writes to
#[derive(Default, Debug)]
struct WriteInfo {
    writes: Vec<Local>,
    /// If this pair of locals is a candidate pair, completely skip processing it during this
    /// statement. All other candidates are unaffected.
    skip_pair: Option<(Local, Local)>,
}

impl WriteInfo {
    fn for_statement<'tcx>(&mut self, statement: &StatementKind<'tcx>, body: &Body<'tcx>) {
        self.reset();
        match statement {
            StatementKind::Assign(box (lhs, rhs)) => {
                self.add_place(*lhs);
                match rhs {
                    Rvalue::Use(op) => {
                        self.add_operand(op);
                        self.consider_skipping_for_assign_use(*lhs, op, body);
                    }
                    Rvalue::Repeat(op, _) => {
                        self.add_operand(op);
                    }
                    Rvalue::Cast(_, op, _)
                    | Rvalue::UnaryOp(_, op)
                    | Rvalue::ShallowInitBox(op, _) => {
                        self.add_operand(op);
                    }
                    Rvalue::BinaryOp(_, ops) | Rvalue::CheckedBinaryOp(_, ops) => {
                        for op in [&ops.0, &ops.1] {
                            self.add_operand(op);
                        }
                    }
                    Rvalue::Aggregate(_, ops) => {
                        for op in ops {
                            self.add_operand(op);
                        }
                    }
                    Rvalue::ThreadLocalRef(_)
                    | Rvalue::NullaryOp(_, _)
                    | Rvalue::Ref(_, _, _)
                    | Rvalue::AddressOf(_, _)
                    | Rvalue::Len(_)
                    | Rvalue::Discriminant(_)
                    | Rvalue::CopyForDeref(_) => (),
                }
            }
            // Retags are technically also reads, but reporting them as a write suffices
            StatementKind::SetDiscriminant { place, .. }
            | StatementKind::Deinit(place)
            | StatementKind::Retag(_, place) => {
                self.add_place(**place);
            }
            StatementKind::Intrinsic(_)
            | StatementKind::ConstEvalCounter
            | StatementKind::Nop
            | StatementKind::Coverage(_)
            | StatementKind::StorageLive(_)
            | StatementKind::StorageDead(_) => (),
            StatementKind::FakeRead(_) | StatementKind::AscribeUserType(_, _) => {
                bug!("{:?} not found in this MIR phase", statement)
            }
        }
    }

    fn consider_skipping_for_assign_use<'tcx>(
        &mut self,
        lhs: Place<'tcx>,
        rhs: &Operand<'tcx>,
        body: &Body<'tcx>,
    ) {
        let Some(rhs) = rhs.place() else {
            return
        };
        if let Some(pair) = places_to_candidate_pair(lhs, rhs, body) {
            self.skip_pair = Some(pair);
        }
    }

    fn for_terminator<'tcx>(&mut self, terminator: &TerminatorKind<'tcx>) {
        self.reset();
        match terminator {
            TerminatorKind::SwitchInt { discr: op, .. }
            | TerminatorKind::Assert { cond: op, .. } => {
                self.add_operand(op);
            }
            TerminatorKind::Call { destination, func, args, .. } => {
                self.add_place(*destination);
                self.add_operand(func);
                for arg in args {
                    self.add_operand(arg);
                }
            }
            TerminatorKind::InlineAsm { operands, .. } => {
                for asm_operand in operands {
                    match asm_operand {
                        InlineAsmOperand::In { value, .. } => {
                            self.add_operand(value);
                        }
                        InlineAsmOperand::Out { place, .. } => {
                            if let Some(place) = place {
                                self.add_place(*place);
                            }
                        }
                        // Note that the `late` field in `InOut` is about whether the registers used
                        // for these things overlap, and is of absolutely no interest to us.
                        InlineAsmOperand::InOut { in_value, out_place, .. } => {
                            if let Some(place) = out_place {
                                self.add_place(*place);
                            }
                            self.add_operand(in_value);
                        }
                        InlineAsmOperand::Const { .. }
                        | InlineAsmOperand::SymFn { .. }
                        | InlineAsmOperand::SymStatic { .. } => (),
                    }
                }
            }
            TerminatorKind::Goto { .. }
            | TerminatorKind::Resume { .. }
            | TerminatorKind::Abort { .. }
            | TerminatorKind::Return
            | TerminatorKind::Unreachable { .. } => (),
            TerminatorKind::Drop { .. } => {
                // `Drop`s create a `&mut` and so are not considered
            }
            TerminatorKind::DropAndReplace { .. }
            | TerminatorKind::Yield { .. }
            | TerminatorKind::GeneratorDrop
            | TerminatorKind::FalseEdge { .. }
            | TerminatorKind::FalseUnwind { .. } => {
                bug!("{:?} not found in this MIR phase", terminator)
            }
        }
    }

    fn add_place(&mut self, place: Place<'_>) {
        self.writes.push(place.local);
    }

    fn add_operand<'tcx>(&mut self, op: &Operand<'tcx>) {
        match op {
            // FIXME(JakobDegen): In a previous version, the `Move` case was incorrectly treated as
            // being a read only. This was unsound, however we cannot add a regression test because
            // it is not possible to set this off with current MIR. Once we have that ability, a
            // regression test should be added.
            Operand::Move(p) => self.add_place(*p),
            Operand::Copy(_) | Operand::Constant(_) => (),
        }
    }

    fn reset(&mut self) {
        self.writes.clear();
        self.skip_pair = None;
    }
}

/////////////////////////////////////////////////////
// Candidate accumulation

/// If the pair of places is being considered for merging, returns the candidate which would be
/// merged in order to accomplish this.
///
/// The contract here is in one direction - there is a guarantee that merging the locals that are
/// outputted by this function would result in an assignment between the inputs becoming a
/// self-assignment. However, there is no guarantee that the returned pair is actually suitable for
/// merging - candidate collection must still check this independently.
///
/// This output is unique for each unordered pair of input places.
fn places_to_candidate_pair<'tcx>(
    a: Place<'tcx>,
    b: Place<'tcx>,
    body: &Body<'tcx>,
) -> Option<(Local, Local)> {
    let (mut a, mut b) = if a.projection.len() == 0 && b.projection.len() == 0 {
        (a.local, b.local)
    } else {
        return None;
    };

    // By sorting, we make sure we're input order independent
    if a > b {
        std::mem::swap(&mut a, &mut b);
    }

    // We could now return `(a, b)`, but then we miss some candidates in the case where `a` can't be
    // used as a `src`.
    if is_local_required(a, body) {
        std::mem::swap(&mut a, &mut b);
    }
    // We could check `is_local_required` again here, but there's no need - after all, we make no
    // promise that the candidate pair is actually valid
    Some((a, b))
}

/// Collects the candidates for merging
///
/// This is responsible for enforcing the first and third bullet point.
fn find_candidates<'alloc, 'tcx>(
    body: &Body<'tcx>,
    borrowed: &BitSet<Local>,
    candidates: &'alloc mut FxHashMap<Local, Vec<Local>>,
    candidates_reverse: &'alloc mut FxHashMap<Local, Vec<Local>>,
) -> Candidates<'alloc> {
    candidates.clear();
    candidates_reverse.clear();
    let mut visitor = FindAssignments { body, candidates, borrowed };
    visitor.visit_body(body);
    // Deduplicate candidates
    for (_, cands) in candidates.iter_mut() {
        cands.sort();
        cands.dedup();
    }
    // Generate the reverse map
    for (src, cands) in candidates.iter() {
        for dest in cands.iter().copied() {
            candidates_reverse.entry(dest).or_default().push(*src);
        }
    }
    Candidates { c: candidates, reverse: candidates_reverse }
}

struct FindAssignments<'a, 'alloc, 'tcx> {
    body: &'a Body<'tcx>,
    candidates: &'alloc mut FxHashMap<Local, Vec<Local>>,
    borrowed: &'a BitSet<Local>,
}

impl<'tcx> Visitor<'tcx> for FindAssignments<'_, '_, 'tcx> {
    fn visit_statement(&mut self, statement: &Statement<'tcx>, _: Location) {
        if let StatementKind::Assign(box (
            lhs,
            Rvalue::CopyForDeref(rhs) | Rvalue::Use(Operand::Copy(rhs) | Operand::Move(rhs)),
        )) = &statement.kind
        {
            let Some((src, dest)) = places_to_candidate_pair(*lhs, *rhs, self.body) else {
                return;
            };

            // As described at the top of the file, we do not go near things that have their address
            // taken.
            if self.borrowed.contains(src) || self.borrowed.contains(dest) {
                return;
            }

            // Also, we need to make sure that MIR actually allows the `src` to be removed
            if is_local_required(src, self.body) {
                return;
            }

            // We may insert duplicates here, but that's fine
            self.candidates.entry(src).or_default().push(dest);
        }
    }
}

/// Some locals are part of the function's interface and can not be removed.
///
/// Note that these locals *can* still be merged with non-required locals by removing that other
/// local.
fn is_local_required(local: Local, body: &Body<'_>) -> bool {
    match body.local_kind(local) {
        LocalKind::Arg | LocalKind::ReturnPointer => true,
        LocalKind::Var | LocalKind::Temp => false,
    }
}

/////////////////////////////////////////////////////////
// MIR Dump

fn dest_prop_mir_dump<'body, 'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &'body Body<'tcx>,
    live: &mut ResultsCursor<'body, 'tcx, MaybeLiveLocals>,
    round: usize,
) {
    let mut reachable = None;
    dump_mir(tcx, false, "DestinationPropagation-dataflow", &round, body, |pass_where, w| {
        let reachable = reachable.get_or_insert_with(|| traversal::reachable_as_bitset(body));

        match pass_where {
            PassWhere::BeforeLocation(loc) if reachable.contains(loc.block) => {
                live.seek_after_primary_effect(loc);
                writeln!(w, "        // live: {:?}", live.get())?;
            }
            PassWhere::AfterTerminator(bb) if reachable.contains(bb) => {
                let loc = body.terminator_loc(bb);
                live.seek_before_primary_effect(loc);
                writeln!(w, "        // live: {:?}", live.get())?;
            }

            PassWhere::BeforeBlock(bb) if reachable.contains(bb) => {
                live.seek_to_block_start(bb);
                writeln!(w, "    // live: {:?}", live.get())?;
            }

            PassWhere::BeforeCFG | PassWhere::AfterCFG | PassWhere::AfterLocation(_) => {}

            PassWhere::BeforeLocation(_) | PassWhere::AfterTerminator(_) => {
                writeln!(w, "        // live: <unreachable>")?;
            }

            PassWhere::BeforeBlock(_) => {
                writeln!(w, "    // live: <unreachable>")?;
            }
        }

        Ok(())
    });
}
