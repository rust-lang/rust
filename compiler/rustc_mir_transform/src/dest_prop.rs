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
//! * `p` and `q` must have the **same type**. If we replace a local with a subtype or supertype,
//!   we may end up with a different vtable for that local. See the `subtyping-impacts-selection`
//!   tests for an example where that causes issues.
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
//! ## Current implementation
//!
//! The current implementation relies on live range computation to check for conflicts. We only
//! allow to merge locals that have disjoint live ranges. The live range are defined with
//! half-statement granularity, so as to make all writes be live for at least a half statement.
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
//!   ```ignore (syntax-highlighting-only)
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
//! * Various perf improvements. There are a bunch of comments in here marked `PERF` with ideas for
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
//! - Layout optimizations for coroutines have been added to improve code generation for
//!   async/await, which are very similar in spirit to what this optimization does.
//!
//! [The next approach][attempt 4] computes a conflict matrix between locals by forbidding merging
//! locals with competing writes or with one write while the other is live.
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
//! [attempt 4]: https://github.com/rust-lang/rust/pull/96451

use rustc_data_structures::union_find::UnionFind;
use rustc_index::bit_set::DenseBitSet;
use rustc_index::interval::SparseIntervalMatrix;
use rustc_index::{IndexVec, newtype_index};
use rustc_middle::mir::visit::{MutVisitor, PlaceContext, VisitPlacesWith, Visitor};
use rustc_middle::mir::*;
use rustc_middle::ty::TyCtxt;
use rustc_mir_dataflow::impls::{DefUse, MaybeLiveLocals};
use rustc_mir_dataflow::points::DenseLocationMap;
use rustc_mir_dataflow::{Analysis, Results};
use tracing::{debug, trace};

pub(super) struct DestinationPropagation;

impl<'tcx> crate::MirPass<'tcx> for DestinationPropagation {
    fn is_enabled(&self, sess: &rustc_session::Session) -> bool {
        sess.mir_opt_level() >= 2
    }

    #[tracing::instrument(level = "trace", skip(self, tcx, body))]
    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        let def_id = body.source.def_id();
        trace!(?def_id);

        let borrowed = rustc_mir_dataflow::impls::borrowed_locals(body);

        let candidates = Candidates::find(body, &borrowed);
        trace!(?candidates);
        if candidates.c.is_empty() {
            return;
        }

        let live = MaybeLiveLocals.iterate_to_fixpoint(tcx, body, Some("MaybeLiveLocals-DestProp"));

        let points = DenseLocationMap::new(body);
        let mut relevant = RelevantLocals::compute(&candidates, body.local_decls.len());
        let mut live = save_as_intervals(&points, body, &relevant, live.results);

        dest_prop_mir_dump(tcx, body, &points, &live, &relevant);

        let mut merged_locals = DenseBitSet::new_empty(body.local_decls.len());

        for (src, dst) in candidates.c.into_iter() {
            trace!(?src, ?dst);

            let Some(mut src) = relevant.find(src) else { continue };
            let Some(mut dst) = relevant.find(dst) else { continue };
            if src == dst {
                continue;
            }

            let Some(src_live_ranges) = live.row(src) else { continue };
            let Some(dst_live_ranges) = live.row(dst) else { continue };
            trace!(?src, ?src_live_ranges);
            trace!(?dst, ?dst_live_ranges);

            if src_live_ranges.disjoint(dst_live_ranges) {
                // We want to replace `src` by `dst`.
                let mut orig_src = relevant.original[src];
                let mut orig_dst = relevant.original[dst];

                // The return place and function arguments are required and cannot be renamed.
                // This check cannot be made during candidate collection, as we may want to
                // unify the same non-required local with several required locals.
                match (is_local_required(orig_src, body), is_local_required(orig_dst, body)) {
                    // Renaming `src` is ok.
                    (false, _) => {}
                    // Renaming `src` is wrong, but renaming `dst` is ok.
                    (true, false) => {
                        std::mem::swap(&mut src, &mut dst);
                        std::mem::swap(&mut orig_src, &mut orig_dst);
                    }
                    // Neither local can be renamed, so skip this case.
                    (true, true) => continue,
                }

                trace!(?src, ?dst, "merge");
                merged_locals.insert(orig_src);
                merged_locals.insert(orig_dst);

                // Replace `src` by `dst`.
                let head = relevant.union(src, dst);
                live.union_rows(/* read */ src, /* write */ head);
                live.union_rows(/* read */ dst, /* write */ head);
            }
        }
        trace!(?merged_locals);
        trace!(?relevant.renames);

        if merged_locals.is_empty() {
            return;
        }

        apply_merges(body, tcx, relevant, merged_locals);
    }

    fn is_required(&self) -> bool {
        false
    }
}

//////////////////////////////////////////////////////////
// Merging
//
// Applies the actual optimization

fn apply_merges<'tcx>(
    body: &mut Body<'tcx>,
    tcx: TyCtxt<'tcx>,
    relevant: RelevantLocals,
    merged_locals: DenseBitSet<Local>,
) {
    let mut merger = Merger { tcx, relevant, merged_locals };
    merger.visit_body_preserves_cfg(body);
}

struct Merger<'tcx> {
    tcx: TyCtxt<'tcx>,
    relevant: RelevantLocals,
    merged_locals: DenseBitSet<Local>,
}

impl<'tcx> MutVisitor<'tcx> for Merger<'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn visit_local(&mut self, local: &mut Local, _: PlaceContext, _location: Location) {
        if let Some(relevant) = self.relevant.find(*local) {
            *local = self.relevant.original[relevant];
        }
    }

    fn visit_statement(&mut self, statement: &mut Statement<'tcx>, location: Location) {
        match &statement.kind {
            // FIXME: Don't delete storage statements, but "merge" the storage ranges instead.
            StatementKind::StorageDead(local) | StatementKind::StorageLive(local)
                if self.merged_locals.contains(*local) =>
            {
                statement.make_nop(true);
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
                            statement.make_nop(true);
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
// Relevant locals
//
// Small utility to reduce size of the conflict matrix by only considering locals that appear in
// the candidates

newtype_index! {
    /// Represent a subset of locals which appear in candidates.
    struct RelevantLocal {}
}

#[derive(Debug)]
struct RelevantLocals {
    original: IndexVec<RelevantLocal, Local>,
    shrink: IndexVec<Local, Option<RelevantLocal>>,
    renames: UnionFind<RelevantLocal>,
}

impl RelevantLocals {
    #[tracing::instrument(level = "trace", skip(candidates, num_locals), ret)]
    fn compute(candidates: &Candidates, num_locals: usize) -> RelevantLocals {
        let mut original = IndexVec::with_capacity(candidates.c.len());
        let mut shrink = IndexVec::from_elem_n(None, num_locals);

        // Mark a local as relevant and record it into the maps.
        let mut declare = |local| {
            shrink.get_or_insert_with(local, || original.push(local));
        };

        for &(src, dest) in candidates.c.iter() {
            declare(src);
            declare(dest)
        }

        let renames = UnionFind::new(original.len());
        RelevantLocals { original, shrink, renames }
    }

    fn find(&mut self, src: Local) -> Option<RelevantLocal> {
        let src = self.shrink[src]?;
        let src = self.renames.find(src);
        Some(src)
    }

    fn union(&mut self, lhs: RelevantLocal, rhs: RelevantLocal) -> RelevantLocal {
        let head = self.renames.unify(lhs, rhs);
        // We need to ensure we keep the original local of the RHS, as it may be a required local.
        self.original[head] = self.original[rhs];
        head
    }
}

/////////////////////////////////////////////////////
// Candidate accumulation

#[derive(Debug, Default)]
struct Candidates {
    /// The set of candidates we are considering in this optimization.
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
    c: Vec<(Local, Local)>,
}

// We first implement some utility functions which we will expose removing candidates according to
// different needs. Throughout the liveness filtering, the `candidates` are only ever accessed
// through these methods, and not directly.
impl Candidates {
    /// Collects the candidates for merging.
    ///
    /// This is responsible for enforcing the first and third bullet point.
    fn find(body: &Body<'_>, borrowed: &DenseBitSet<Local>) -> Candidates {
        let mut visitor = FindAssignments { body, candidates: Default::default(), borrowed };
        visitor.visit_body(body);

        Candidates { c: visitor.candidates }
    }
}

struct FindAssignments<'a, 'tcx> {
    body: &'a Body<'tcx>,
    candidates: Vec<(Local, Local)>,
    borrowed: &'a DenseBitSet<Local>,
}

impl<'tcx> Visitor<'tcx> for FindAssignments<'_, 'tcx> {
    fn visit_statement(&mut self, statement: &Statement<'tcx>, _: Location) {
        if let StatementKind::Assign(box (
            lhs,
            Rvalue::CopyForDeref(rhs) | Rvalue::Use(Operand::Copy(rhs) | Operand::Move(rhs)),
        )) = &statement.kind
            && let Some(src) = lhs.as_local()
            && let Some(dest) = rhs.as_local()
        {
            // As described at the top of the file, we do not go near things that have
            // their address taken.
            if self.borrowed.contains(src) || self.borrowed.contains(dest) {
                return;
            }

            // As described at the top of this file, we do not touch locals which have
            // different types.
            let src_ty = self.body.local_decls()[src].ty;
            let dest_ty = self.body.local_decls()[dest].ty;
            if src_ty != dest_ty {
                // FIXME(#112651): This can be removed afterwards. Also update the module description.
                trace!("skipped `{src:?} = {dest:?}` due to subtyping: {src_ty} != {dest_ty}");
                return;
            }

            // We may insert duplicates here, but that's fine
            self.candidates.push((src, dest));
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
        LocalKind::Temp => false,
    }
}

/////////////////////////////////////////////////////////
// MIR Dump

fn dest_prop_mir_dump<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
    points: &DenseLocationMap,
    live: &SparseIntervalMatrix<RelevantLocal, TwoStepIndex>,
    relevant: &RelevantLocals,
) {
    let locals_live_at = |location| {
        live.rows()
            .filter(|&r| live.contains(r, location))
            .map(|rl| relevant.original[rl])
            .collect::<Vec<_>>()
    };

    if let Some(dumper) = MirDumper::new(tcx, "DestinationPropagation-dataflow", body) {
        let extra_data = &|pass_where, w: &mut dyn std::io::Write| {
            if let PassWhere::BeforeLocation(loc) = pass_where {
                let location = TwoStepIndex::new(points, loc, Effect::Before);
                let live = locals_live_at(location);
                writeln!(w, "        // before: {:?} => {:?}", location, live)?;
            }
            if let PassWhere::AfterLocation(loc) = pass_where {
                let location = TwoStepIndex::new(points, loc, Effect::After);
                let live = locals_live_at(location);
                writeln!(w, "        // after: {:?} => {:?}", location, live)?;
            }
            Ok(())
        };

        dumper.set_extra_data(extra_data).dump_mir(body)
    }
}

#[derive(Copy, Clone, Debug)]
enum Effect {
    Before,
    After,
}

rustc_index::newtype_index! {
    /// A reversed `PointIndex` but with the lower bit encoding early/late inside the statement.
    /// The reversed order allows to use the more efficient `IntervalSet::append` method while we
    /// iterate on the statements in reverse order.
    #[orderable]
    #[debug_format = "TwoStepIndex({})"]
    struct TwoStepIndex {}
}

impl TwoStepIndex {
    fn new(elements: &DenseLocationMap, location: Location, effect: Effect) -> TwoStepIndex {
        let point = elements.point_from_location(location);
        let effect = match effect {
            Effect::Before => 0,
            Effect::After => 1,
        };
        let max_index = 2 * elements.num_points() as u32 - 1;
        let index = 2 * point.as_u32() + (effect as u32);
        // Reverse the indexing to use more efficient `IntervalSet::append`.
        TwoStepIndex::from_u32(max_index - index)
    }
}

/// Add points depending on the result of the given dataflow analysis.
fn save_as_intervals<'tcx>(
    elements: &DenseLocationMap,
    body: &Body<'tcx>,
    relevant: &RelevantLocals,
    results: Results<DenseBitSet<Local>>,
) -> SparseIntervalMatrix<RelevantLocal, TwoStepIndex> {
    let mut values = SparseIntervalMatrix::new(2 * elements.num_points());
    let mut state = MaybeLiveLocals.bottom_value(body);
    let reachable_blocks = traversal::reachable_as_bitset(body);

    let two_step_loc = |location, effect| TwoStepIndex::new(elements, location, effect);
    let append_at =
        |values: &mut SparseIntervalMatrix<_, _>, state: &DenseBitSet<Local>, twostep| {
            for (relevant, &original) in relevant.original.iter_enumerated() {
                if state.contains(original) {
                    values.append(relevant, twostep);
                }
            }
        };

    // Iterate blocks in decreasing order, to visit locations in decreasing order. This
    // allows to use the more efficient `append` method to interval sets.
    for block in body.basic_blocks.indices().rev() {
        if !reachable_blocks.contains(block) {
            continue;
        }

        state.clone_from(&results[block]);

        let block_data = &body.basic_blocks[block];
        let loc = Location { block, statement_index: block_data.statements.len() };

        let term = block_data.terminator();
        let mut twostep = two_step_loc(loc, Effect::After);
        append_at(&mut values, &state, twostep);
        // Ensure we have a non-zero live range even for dead stores. This is done by marking all
        // the written-to locals as live in the second half of the statement.
        // We also ensure that operands read by terminators conflict with writes by that terminator.
        // For instance a function call may read args after having written to the destination.
        VisitPlacesWith(|place: Place<'tcx>, ctxt| {
            if let Some(relevant) = relevant.shrink[place.local] {
                match DefUse::for_place(place, ctxt) {
                    DefUse::Def | DefUse::Use | DefUse::PartialWrite => {
                        values.insert(relevant, twostep);
                    }
                    DefUse::NonUse => {}
                }
            }
        })
        .visit_terminator(term, loc);

        twostep = TwoStepIndex::from_u32(twostep.as_u32() + 1);
        debug_assert_eq!(twostep, two_step_loc(loc, Effect::Before));
        MaybeLiveLocals.apply_early_terminator_effect(&mut state, term, loc);
        MaybeLiveLocals.apply_primary_terminator_effect(&mut state, term, loc);
        append_at(&mut values, &state, twostep);

        for (statement_index, stmt) in block_data.statements.iter().enumerate().rev() {
            let loc = Location { block, statement_index };
            twostep = TwoStepIndex::from_u32(twostep.as_u32() + 1);
            debug_assert_eq!(twostep, two_step_loc(loc, Effect::After));
            append_at(&mut values, &state, twostep);
            // Like terminators, ensure we have a non-zero live range even for dead stores.
            // Some rvalues interleave reads and writes, for instance `Rvalue::Aggregate`, see
            // https://github.com/rust-lang/rust/issues/146383. By precaution, treat statements
            // as behaving so by default.
            // We make an exception for simple assignments `_a.stuff = {copy|move} _b.stuff`,
            // as marking `_b` live here would prevent unification.
            let is_simple_assignment = match stmt.kind {
                StatementKind::Assign(box (
                    lhs,
                    Rvalue::CopyForDeref(rhs)
                    | Rvalue::Use(Operand::Copy(rhs) | Operand::Move(rhs)),
                )) => lhs.projection == rhs.projection,
                _ => false,
            };
            VisitPlacesWith(|place: Place<'tcx>, ctxt| {
                if let Some(relevant) = relevant.shrink[place.local] {
                    match DefUse::for_place(place, ctxt) {
                        DefUse::Def | DefUse::PartialWrite => {
                            values.insert(relevant, twostep);
                        }
                        DefUse::Use if !is_simple_assignment => {
                            values.insert(relevant, twostep);
                        }
                        DefUse::Use | DefUse::NonUse => {}
                    }
                }
            })
            .visit_statement(stmt, loc);

            twostep = TwoStepIndex::from_u32(twostep.as_u32() + 1);
            debug_assert_eq!(twostep, two_step_loc(loc, Effect::Before));
            MaybeLiveLocals.apply_early_statement_effect(&mut state, stmt, loc);
            MaybeLiveLocals.apply_primary_statement_effect(&mut state, stmt, loc);
            // ... but reads from operands are marked as live here so they do not conflict with
            // the all the writes we manually marked as live in the second half of the statement.
            append_at(&mut values, &state, twostep);
        }
    }

    values
}
