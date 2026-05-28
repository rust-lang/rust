use std::fmt;

use rustc_index::IndexVec;
use rustc_index::bit_set::DenseBitSet;
use rustc_index::interval::SparseIntervalMatrix;
use rustc_middle::mir::visit::{
    MutatingUseContext, NonMutatingUseContext, PlaceContext, VisitPlacesWith, Visitor,
};
use rustc_middle::mir::{self, BasicBlock, Local, Location, MirDumper, PassWhere, Place};
use rustc_middle::ty::TyCtxt;
use tracing::trace;

use crate::fmt::DebugWithContext;
use crate::impls::{DefUse, MaybeBorrowedLocals, MaybeLiveLocals};
use crate::points::{DenseLocationMap, PointIndex};
use crate::{Analysis, GenKill, JoinSemiLattice, ResultsVisitor, visit_reachable_results};

struct KillPointsVisitor<'a> {
    kill_points: &'a mut Vec<(Local, Location)>,
    live_on_entry: &'a mut IndexVec<BasicBlock, DenseBitSet<Local>>,
}

impl<'tcx> ResultsVisitor<'tcx, MaybeLiveLocals> for KillPointsVisitor<'_> {
    fn visit_block_start(&mut self, state: &DenseBitSet<Local>, block: BasicBlock) {
        self.live_on_entry[block].clone_from(state);
    }

    fn visit_after_early_statement_effect(
        &mut self,
        _analysis: &MaybeLiveLocals,
        state: &DenseBitSet<Local>,
        statement: &mir::Statement<'tcx>,
        location: Location,
    ) {
        VisitPlacesWith(|place: Place<'tcx>, ctxt| {
            // Ignore non-uses.
            match ctxt {
                PlaceContext::NonMutatingUse(_) | PlaceContext::MutatingUse(_) => {}
                PlaceContext::NonUse(_) => return,
            }

            // If a local is used in a statement but is dead after it then this
            // location is a kill point.
            if !state.contains(place.local) {
                self.kill_points.push((place.local, location));
            }
        })
        .visit_statement(statement, location);
    }

    fn visit_after_early_terminator_effect(
        &mut self,
        _analysis: &MaybeLiveLocals,
        state: &DenseBitSet<Local>,
        terminator: &mir::Terminator<'tcx>,
        location: Location,
    ) {
        VisitPlacesWith(|place: Place<'tcx>, ctxt| {
            // Ignore non-uses (they don't do anything) and edge uses (kill
            // points for those go at the start of the corresponding successor).
            match ctxt {
                PlaceContext::MutatingUse(
                    MutatingUseContext::AsmOutput
                    | MutatingUseContext::Call
                    | MutatingUseContext::Yield,
                )
                | PlaceContext::NonUse(_) => return,
                PlaceContext::NonMutatingUse(_) | PlaceContext::MutatingUse(_) => {}
            }

            // If a local is used in a terminator but is dead after it then this
            // location is a kill point.
            if !state.contains(place.local) {
                self.kill_points.push((place.local, location));
            }
        })
        .visit_terminator(terminator, location);
    }
}

#[derive(Debug, PartialEq, Eq)]
struct Domain {
    maybe_live: DenseBitSet<Local>,
    maybe_borrowed: DenseBitSet<Local>,
}

impl Clone for Domain {
    fn clone(&self) -> Self {
        Domain { maybe_live: self.maybe_live.clone(), maybe_borrowed: self.maybe_borrowed.clone() }
    }

    // Data flow engine when possible uses `clone_from` for domain values.
    // Providing an implementation will avoid some intermediate memory allocations.
    fn clone_from(&mut self, other: &Self) {
        self.maybe_live.clone_from(&other.maybe_live);
        self.maybe_borrowed.clone_from(&other.maybe_borrowed);
    }
}

impl JoinSemiLattice for Domain {
    fn join(&mut self, other: &Self) -> bool {
        self.maybe_live.join(&other.maybe_live) | self.maybe_borrowed.join(&other.maybe_borrowed)
    }
}

impl<C> DebugWithContext<C> for Domain {
    fn fmt_with(&self, ctxt: &C, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("maybe_live: ")?;
        self.maybe_live.fmt_with(ctxt, f)?;
        f.write_str("maybe_borrowed: ")?;
        self.maybe_borrowed.fmt_with(ctxt, f)?;
        Ok(())
    }

    fn fmt_diff_with(&self, old: &Self, ctxt: &C, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self == old {
            return Ok(());
        }

        if self.maybe_live != old.maybe_live {
            f.write_str("maybe_live: ")?;
            self.maybe_live.fmt_diff_with(&old.maybe_live, ctxt, f)?;
            f.write_str("\n")?;
        }

        if self.maybe_borrowed != old.maybe_borrowed {
            f.write_str("maybe_borrowed: ")?;
            self.maybe_borrowed.fmt_diff_with(&old.maybe_borrowed, ctxt, f)?;
            f.write_str("\n")?;
        }

        Ok(())
    }
}

struct PreciseLiveness<'a> {
    kill_point_map: &'a IndexVec<PointIndex, &'a [(Local, Location)]>,
    live_on_entry: &'a IndexVec<BasicBlock, DenseBitSet<Local>>,
    points: &'a DenseLocationMap,
}

impl PreciseLiveness<'_> {
    fn apply_block_start_effect(&self, state: &mut Domain, block: BasicBlock) {
        // Only keep locals that are either live or borrowed.
        //
        // Notably this kills any dead results produced by a predecessor's
        // terminator.
        state.maybe_live.intersect_with_union(&self.live_on_entry[block], &state.maybe_borrowed);
    }
}

impl<'tcx> Analysis<'tcx> for PreciseLiveness<'_> {
    type Domain = Domain;

    const NAME: &'static str = "precise_liveness";

    fn bottom_value(&self, body: &mir::Body<'tcx>) -> Domain {
        Domain {
            maybe_live: DenseBitSet::new_empty(body.local_decls.len()),
            maybe_borrowed: DenseBitSet::new_empty(body.local_decls.len()),
        }
    }

    fn initialize_start_block(&self, body: &mir::Body<'tcx>, state: &mut Domain) {
        // Function arguments start out as live.
        for arg in body.args_iter() {
            state.maybe_live.gen_(arg);
        }
    }

    fn apply_primary_statement_effect(
        &self,
        state: &mut Domain,
        statement: &mir::Statement<'tcx>,
        location: Location,
    ) {
        if location.statement_index == 0 {
            self.apply_block_start_effect(state, location.block);
        }

        // StorageDead always kills a local, even if it has been borrowed.
        if let mir::StatementKind::StorageDead(local) = statement.kind {
            state.maybe_live.kill(local);
            state.maybe_borrowed.kill(local);
            return;
        }

        MaybeBorrowedLocals::transfer_function(&mut state.maybe_borrowed)
            .visit_statement(statement, location);

        // Kill moved operands if the whole local was moved.
        VisitPlacesWith(|place: Place<'tcx>, ctxt| {
            if ctxt == PlaceContext::NonMutatingUse(NonMutatingUseContext::Move) {
                if let Some(local) = place.as_local() {
                    state.maybe_live.kill(local);
                    state.maybe_borrowed.kill(local);
                }
            }
        })
        .visit_statement(statement, location);

        // Gen destination places.
        VisitPlacesWith(|place: Place<'tcx>, ctxt| match DefUse::for_place(place, ctxt) {
            DefUse::Def | DefUse::PartialWrite => state.maybe_live.gen_(place.local),
            DefUse::Use | DefUse::NonUse => {}
        })
        .visit_statement(statement, location);

        // Apply kill points at this statement: if a variable is dead
        // then it doesn't need storage, *except* if its address has been taken.
        let point = self.points.point_from_location(location);
        for &(local, _) in self.kill_point_map[point] {
            if !state.maybe_borrowed.contains(local) {
                state.maybe_live.kill(local);
            }
        }
    }

    fn apply_primary_terminator_effect<'mir>(
        &self,
        state: &mut Domain,
        terminator: &'mir mir::Terminator<'tcx>,
        location: Location,
    ) -> mir::TerminatorEdges<'mir, 'tcx> {
        if location.statement_index == 0 {
            self.apply_block_start_effect(state, location.block);
        }

        MaybeBorrowedLocals::transfer_function(&mut state.maybe_borrowed)
            .visit_terminator(terminator, location);

        // Kill moved operands if the whole local was moved. Also kill dropped
        // places if the entire local was dropped.
        VisitPlacesWith(|place: Place<'tcx>, ctxt| {
            if let PlaceContext::NonMutatingUse(NonMutatingUseContext::Move)
            | PlaceContext::MutatingUse(MutatingUseContext::Drop) = ctxt
            {
                if let Some(local) = place.as_local() {
                    state.maybe_live.kill(local);
                    state.maybe_borrowed.kill(local);
                }
            }
        })
        .visit_terminator(terminator, location);

        // Gen destination places.
        VisitPlacesWith(|place: Place<'tcx>, ctxt| {
            // These are handled through `apply_call_return_effect`.
            if let PlaceContext::MutatingUse(
                MutatingUseContext::AsmOutput
                | MutatingUseContext::Call
                | MutatingUseContext::Yield,
            ) = ctxt
            {
                return;
            }

            match DefUse::for_place(place, ctxt) {
                DefUse::Def | DefUse::PartialWrite => state.maybe_live.gen_(place.local),
                DefUse::Use | DefUse::NonUse => {}
            }
        })
        .visit_terminator(terminator, location);

        terminator.edges()
    }

    fn apply_call_return_effect(
        &self,
        state: &mut Domain,
        _block: BasicBlock,
        return_places: mir::CallReturnPlaces<'_, 'tcx>,
    ) {
        return_places.for_each(|place| state.maybe_live.gen_(place.local));
    }
}

/// Different "phases" of a single MIR statement, used to describe how
/// overlapping operands are handled.
///
/// As a general rule, source operands are read in the `Early` phase and
/// destination places are written in the `Late` phase.
#[derive(Copy, Clone, Debug)]
pub enum SplitPointEffect {
    Early = 0,
    Late = 1,
}

rustc_index::newtype_index! {
    /// A `PointIndex` with the lower bit encoding early/late inside a statement.
    ///
    /// This is used to model overlap constraints within a MIR statement: if a
    /// source/destination are allowed to overlap then the source is read in
    /// `SplitPointEffect::Early` and the write is done in
    /// `SplitPointEffect::Late`.
    #[orderable]
    #[debug_format = "SplitPointIndex({})"]
    pub struct SplitPointIndex {}
}

impl SplitPointIndex {
    pub fn new(point: PointIndex, effect: SplitPointEffect) -> SplitPointIndex {
        let index = (point.as_u32() << 1) | (effect as u32);
        SplitPointIndex::from_u32(index)
    }

    pub fn point(self) -> PointIndex {
        PointIndex::from_u32(self.as_u32() >> 1)
    }

    pub fn effect(self) -> SplitPointEffect {
        match self.as_u32() & 1 {
            0 => SplitPointEffect::Early,
            1 => SplitPointEffect::Late,
            _ => unreachable!(),
        }
    }
}

fn compute_kill_points<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &mir::Body<'tcx>,
    pass_name: Option<&'static str>,
) -> (Vec<(Local, Location)>, IndexVec<BasicBlock, DenseBitSet<Local>>) {
    let maybe_live_locals = MaybeLiveLocals.iterate_to_fixpoint(tcx, body, pass_name);
    let mut kill_points = vec![];
    let mut live_on_entry = IndexVec::from_elem_n(
        DenseBitSet::new_empty(body.local_decls.len()),
        body.basic_blocks.len(),
    );
    let mut visitor =
        KillPointsVisitor { kill_points: &mut kill_points, live_on_entry: &mut live_on_entry };
    visit_reachable_results(body, &maybe_live_locals, &mut visitor);
    trace!(?kill_points);
    trace!(?live_on_entry);
    (kill_points, live_on_entry)
}

fn kill_point_map<'a>(
    kill_points: &'a [(Local, Location)],
    points: &DenseLocationMap,
) -> IndexVec<PointIndex, &'a [(Local, Location)]> {
    let mut out = IndexVec::from_elem_n(&[][..], points.num_points());
    for chunk in kill_points.chunk_by(|a, b| a.1 == b.1) {
        let point = points.point_from_location(chunk[0].1);
        trace!("Kill points at {:?}: {:?}", chunk[0].1, chunk);
        out[point] = chunk;
    }
    out
}

/// Helper type to construct a `SparseIntervalMatrix`.
struct MatrixBuilder {
    matrix: SparseIntervalMatrix<Local, SplitPointIndex>,
    range_start: IndexVec<Local, Option<SplitPointIndex>>,

    // Track locals that have been live at any point in a block so that at the
    // end of a block we don't need to iterate over all locals. This
    // significantly speeds up matrix building.
    maybe_live_locals: Vec<Local>,
}

impl MatrixBuilder {
    fn gen_(&mut self, local: Local, point: PointIndex, effect: SplitPointEffect) {
        let split_point = SplitPointIndex::new(point, effect);

        // No-op if the local is already live.
        if self.range_start[local].is_none() {
            self.range_start[local] = Some(split_point);
            self.maybe_live_locals.push(local);
        }
    }

    fn kill(&mut self, local: Local, point: PointIndex, effect: SplitPointEffect) {
        let end = SplitPointIndex::new(point, effect);

        // No-op if the local is already dead.
        if let Some(start) = self.range_start[local].take() {
            debug_assert!(end >= start);
            self.matrix.append_range(local, start..=end);
        }
    }

    fn kill_all(&mut self, point: PointIndex, effect: SplitPointEffect) {
        while let Some(local) = self.maybe_live_locals.pop() {
            self.kill(local, point, effect);
        }
    }
}

pub fn liveness_matrix<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &mir::Body<'tcx>,
    points: &DenseLocationMap,
    pass_name: Option<&'static str>,
) -> SparseIntervalMatrix<Local, SplitPointIndex> {
    let (kill_points, live_on_entry) = compute_kill_points(tcx, body, pass_name);
    let kill_point_map = &kill_point_map(&kill_points, points);
    let mut results = PreciseLiveness { kill_point_map, live_on_entry: &live_on_entry, points }
        .iterate_to_fixpoint(tcx, body, pass_name);

    let mut builder = MatrixBuilder {
        matrix: SparseIntervalMatrix::new(points.num_points() * 2),
        range_start: IndexVec::from_elem_n(None, body.local_decls.len()),
        maybe_live_locals: Vec::new(),
    };
    for (block, block_data) in body.basic_blocks.iter_enumerated() {
        // We can mutate the state in-place since we're not using it any more
        // after this point.
        let state = &mut results.entry_states[block];

        // Only keep locals that are either live or borrowed.
        //
        // Notably this kills any dead results produced by a predecessor's
        // terminator.
        state.maybe_live.intersect_with_union(&live_on_entry[block], &state.maybe_borrowed);

        for local in state.maybe_live.iter() {
            builder.gen_(local, points.entry_point(block), SplitPointEffect::Early);
        }

        for (statement_index, statement) in block_data.statements.iter().enumerate() {
            let location = Location { block, statement_index };
            let point = points.point_from_location(location);

            // StorageDead always kills a local, even if it has been borrowed.
            if let mir::StatementKind::StorageDead(local) = statement.kind {
                builder.kill(local, point, SplitPointEffect::Late);
                state.maybe_borrowed.kill(local);
                continue;
            }

            MaybeBorrowedLocals::transfer_function(&mut state.maybe_borrowed)
                .visit_statement(statement, location);

            // Kill moved operands if the whole local was moved.
            VisitPlacesWith(|place: Place<'tcx>, ctxt| {
                if ctxt == PlaceContext::NonMutatingUse(NonMutatingUseContext::Move) {
                    if let Some(local) = place.as_local() {
                        builder.kill(local, point, SplitPointEffect::Early);
                        state.maybe_borrowed.kill(local);
                    }
                }
            })
            .visit_statement(statement, location);

            // Kill any locals which are no longer used after this statement,
            // but only if they have not been borrowed.
            for &(local, _) in kill_point_map[point] {
                if !state.maybe_borrowed.contains(local) {
                    builder.kill(local, point, SplitPointEffect::Early);
                }
            }

            // Gen destination places.
            VisitPlacesWith(|place: Place<'tcx>, ctxt| match DefUse::for_place(place, ctxt) {
                DefUse::Def | DefUse::PartialWrite => {
                    builder.gen_(place.local, point, SplitPointEffect::Late)
                }
                DefUse::Use | DefUse::NonUse => {}
            })
            .visit_statement(statement, location);

            // Kill any dead destination places: they will only appear at
            // the late point of the statement they are generated in, which is
            // sufficient for determining overlap.
            for &(local, _) in kill_point_map[point] {
                if !state.maybe_borrowed.contains(local) {
                    builder.kill(local, point, SplitPointEffect::Late);
                }
            }
        }

        let location = Location { block, statement_index: block_data.statements.len() };
        let point = points.point_from_location(location);
        let terminator = block_data.terminator();

        MaybeBorrowedLocals::transfer_function(&mut state.maybe_borrowed)
            .visit_terminator(terminator, location);

        // Kill moved operands if the whole local was moved. Also kill dropped
        // places if the entire local was dropped.
        VisitPlacesWith(|place: Place<'tcx>, ctxt| {
            if let PlaceContext::NonMutatingUse(NonMutatingUseContext::Move)
            | PlaceContext::MutatingUse(MutatingUseContext::Drop) = ctxt
            {
                if let Some(local) = place.as_local() {
                    builder.kill(local, point, SplitPointEffect::Early);
                    state.maybe_borrowed.kill(local);
                }
            }
        })
        .visit_terminator(terminator, location);

        // Kill any locals which are no longer used after this terminator,
        // but only if they have not been borrowed.
        for &(local, _) in kill_point_map[point] {
            if !state.maybe_borrowed.contains(local) {
                builder.kill(local, point, SplitPointEffect::Early);
            }
        }

        // Gen destination places.
        VisitPlacesWith(|place: Place<'tcx>, ctxt| match DefUse::for_place(place, ctxt) {
            DefUse::Def | DefUse::PartialWrite => {
                builder.gen_(place.local, point, SplitPointEffect::Late)
            }
            DefUse::Use | DefUse::NonUse => {}
        })
        .visit_terminator(terminator, location);

        // Move arguments to a call are treated specially: the place that they
        // represent is passed directly to the callee, which means that they are
        // not allowed to alias any other move operand or the destination place.
        // This is represented here by extending their live range to the late
        // part, making it overlap with that of the destination place.
        //
        // Notably, this *doesn't* apply to TailCall.
        if let mir::TerminatorKind::Call {
            func: _,
            args,
            destination: _,
            target: _,
            unwind: _,
            call_source: _,
            fn_span: _,
        } = &terminator.kind
        {
            for arg in args {
                if let mir::Operand::Move(place) = arg.node {
                    builder.gen_(place.local, point, SplitPointEffect::Late);
                    builder.kill(place.local, point, SplitPointEffect::Late);
                }
            }
        }

        // End the lifetimes of all locals at the end of the block. Successor
        // blocks (which may not be continuous in the index space!) will
        // initialize the lifetimes again from their entry state.
        builder.kill_all(point, SplitPointEffect::Late);
    }

    builder.matrix
}

pub fn dump_liveness_matrix<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &mir::Body<'tcx>,
    pass_name: &'static str,
    points: &DenseLocationMap,
    matrix: &SparseIntervalMatrix<Local, SplitPointIndex>,
) {
    let locals_live_at = |split_point| {
        matrix.rows().filter(|&r| matrix.contains(r, split_point)).collect::<Vec<_>>()
    };

    if let Some(dumper) = MirDumper::new(tcx, pass_name, body) {
        let extra_data = &|pass_where, w: &mut dyn std::io::Write| {
            if let PassWhere::BeforeLocation(loc) = pass_where {
                let point = points.point_from_location(loc);
                let split_point = SplitPointIndex::new(point, SplitPointEffect::Early);
                let live = locals_live_at(split_point);
                writeln!(w, "        // {loc:?}-early => {live:?}")?;
                let split_point = SplitPointIndex::new(point, SplitPointEffect::Late);
                let live = locals_live_at(split_point);
                writeln!(w, "        // {loc:?}-late => {live:?}")?;
            }
            Ok(())
        };

        dumper.set_extra_data(extra_data).dump_mir(body)
    }
}
