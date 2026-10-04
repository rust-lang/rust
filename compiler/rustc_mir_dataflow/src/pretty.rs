use std::io;

use rustc_attr_ir::{RustcMirKind, find_attr};
use rustc_data_structures::fx::FxIndexMap;
use rustc_middle::mir::{self, Body, Local, PassWhere, traversal};
use rustc_middle::ty::TyCtxt;

use crate::debuginfo::debuginfo_locals;
use crate::framework::Analysis;
use crate::impls::{
    MaybeLiveLocals, MaybeTransitiveLiveLocals, SplitPointEffect, SplitPointIndex, borrowed_locals,
    dump_liveness_matrix, liveness_matrix,
};
use crate::points::DenseLocationMap;
use crate::{ResultsVisitor, visit_results};

type ExtraDataFn = dyn Fn(PassWhere, &mut dyn io::Write) -> io::Result<()>;

pub(crate) fn mir_pretty_extra_data<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
) -> Option<Box<ExtraDataFn>> {
    let def_id = body.source.def_id();
    let Some(kind) = find_attr!(tcx, def_id, RustcMir(kind) => kind) else {
        return None;
    };
    // FIXME(tmiasko): maybe collect results across multiple dataflows?
    if kind.contains(&RustcMirKind::PrettyLiveLocals) {
        let results = MaybeLiveLocals.iterate_to_fixpoint(tcx, body, None);
        let mut annotator = Annotator::new();
        let blocks = traversal::reachable(body).map(|(bb, _)| bb);
        visit_results(body, blocks, &results, &mut annotator);
        return Some(annotator.into_extra_data());
    }
    if kind.contains(&RustcMirKind::PrettyTransitiveLiveLocals) {
        let borrowed_locals = borrowed_locals(body);
        let debuginfo_locals = debuginfo_locals(body);
        let results = MaybeTransitiveLiveLocals::new(&borrowed_locals, &debuginfo_locals)
            .iterate_to_fixpoint(tcx, body, None);
        let mut annotator = Annotator::new();
        let blocks = traversal::reachable(body).map(|(bb, _)| bb);
        visit_results(body, blocks, &results, &mut annotator);
        return Some(annotator.into_extra_data());
    }
    if kind.contains(&RustcMirKind::PrettyPreciseLiveness) {
        let points = DenseLocationMap::new(body);
        let matrix = liveness_matrix(tcx, body, &points, None);
        // Dump to a file for EMIT_MIR
        dump_liveness_matrix(tcx, body, "PreciseLiveness", &points, &matrix);
        let locals_live_at = move |split_point| {
            matrix.rows().filter(|&r| matrix.contains(r, split_point)).collect::<Vec<_>>()
        };
        return Some(Box::new(move |pass_where, out| {
            if let PassWhere::BeforeLocation(location) = pass_where {
                let point = points.point_from_location(location);
                let split_point = SplitPointIndex::new(point, SplitPointEffect::Early);
                let live = locals_live_at(split_point);
                writeln!(out, "        // early: {live:?}")?;
                let split_point = SplitPointIndex::new(point, SplitPointEffect::Late);
                let live = locals_live_at(split_point);
                writeln!(out, "        // late: {live:?}")?;
            }
            Ok(())
        }));
    }
    None
}

/// Examines dataflow results and collects extra annotations for MIR pretty printing.
struct Annotator {
    // FIXME(tmiasko) maybe use multimap?
    annotations: FxIndexMap<PassWhere, String>,
}

impl Annotator {
    fn new() -> Self {
        Annotator { annotations: Default::default() }
    }

    fn into_extra_data(self) -> Box<ExtraDataFn> {
        Box::new(move |pass_where: PassWhere, out: &mut dyn io::Write| {
            let Some(s) = self.annotations.get(&pass_where) else {
                return Ok(());
            };
            writeln!(out, "        // {s}")
        })
    }
}

impl<'tcx> ResultsVisitor<'tcx, MaybeLiveLocals> for Annotator {
    fn visit_after_primary_statement_effect(
        &mut self,
        state: &<MaybeLiveLocals as Analysis<'tcx>>::Domain,
        _statement: &mir::Statement<'tcx>,
        location: mir::Location,
    ) {
        let live: Vec<Local> = state.iter().collect();
        self.annotations.insert(PassWhere::BeforeLocation(location), format!("live: {live:?}"));
    }

    fn visit_after_primary_terminator_effect(
        &mut self,
        state: &<MaybeLiveLocals as Analysis<'tcx>>::Domain,
        _terminator: &mir::Terminator<'tcx>,
        location: mir::Location,
    ) {
        let live: Vec<Local> = state.iter().collect();
        self.annotations.insert(PassWhere::BeforeLocation(location), format!("live: {live:?}"));
    }
}

impl<'tcx> ResultsVisitor<'tcx, MaybeTransitiveLiveLocals<'_>> for Annotator {
    fn visit_after_primary_statement_effect(
        &mut self,
        state: &<MaybeLiveLocals as Analysis<'tcx>>::Domain,
        _statement: &mir::Statement<'tcx>,
        location: mir::Location,
    ) {
        let live: Vec<Local> = state.iter().collect();
        self.annotations.insert(PassWhere::BeforeLocation(location), format!("live: {live:?}"));
    }

    fn visit_after_primary_terminator_effect(
        &mut self,
        state: &<MaybeLiveLocals as Analysis<'tcx>>::Domain,
        _terminator: &mir::Terminator<'tcx>,
        location: mir::Location,
    ) {
        let live: Vec<Local> = state.iter().collect();
        self.annotations.insert(PassWhere::BeforeLocation(location), format!("live: {live:?}"));
    }
}
