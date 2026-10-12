use std::io;

use rustc_attr_ir::{RustcMirKind, find_attr};
use rustc_data_structures::fx::FxIndexMap;
use rustc_middle::mir::{self, Body, Local, MirDumper, PassWhere, traversal};
use rustc_middle::ty::TyCtxt;

use crate::debuginfo::debuginfo_locals;
use crate::framework::Analysis;
use crate::impls::{
    MaybeLiveLocals, MaybeTransitiveLiveLocals, SplitPointEffect, SplitPointIndex, borrowed_locals,
    liveness_matrix,
};
use crate::points::DenseLocationMap;
use crate::{ResultsVisitor, visit_results};

type ExtraDataFn = dyn Fn(PassWhere, &mut dyn io::Write) -> io::Result<()>;

pub(crate) fn mir_pretty_extra_data<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
) -> Option<Box<ExtraDataFn>> {
    let def_id = body.source.def_id();
    let Some(kinds) = find_attr!(tcx, def_id, RustcMir(kind) => kind) else {
        return None;
    };
    let mut annotations = Annotations::default();
    for kind in kinds {
        match kind {
            RustcMirKind::PrettyLiveLocals => {
                let results = MaybeLiveLocals.iterate_to_fixpoint(tcx, body, None);
                let blocks = traversal::reachable(body).map(|(bb, _)| bb);
                visit_results(
                    body,
                    blocks,
                    &results,
                    &mut Annotator { annotations: &mut annotations },
                );
            }
            RustcMirKind::PrettyPreciseLiveness => {
                let points = DenseLocationMap::new(body);
                let matrix = liveness_matrix(tcx, body, &points, None);
                for (block, data) in body.basic_blocks.iter_enumerated() {
                    for statement_index in 0..=data.statements.len() {
                        let location = mir::Location { block, statement_index };
                        let point = points.point_from_location(location);
                        let locals_live_at = |effect| {
                            let split_point = SplitPointIndex::new(point, effect);
                            matrix
                                .rows()
                                .filter(|&r| matrix.contains(r, split_point))
                                .collect::<Vec<_>>()
                        };
                        let early = locals_live_at(SplitPointEffect::Early);
                        annotations
                            .add(PassWhere::BeforeLocation(location), format!("early: {early:?}"));
                        let late = locals_live_at(SplitPointEffect::Late);
                        annotations
                            .add(PassWhere::BeforeLocation(location), format!("late: {late:?}"));
                    }
                }
            }
            RustcMirKind::PrettyTransitiveLiveLocals => {
                let borrowed_locals = borrowed_locals(body);
                let debuginfo_locals = debuginfo_locals(body);
                let results = MaybeTransitiveLiveLocals::new(&borrowed_locals, &debuginfo_locals)
                    .iterate_to_fixpoint(tcx, body, None);
                let blocks = traversal::reachable(body).map(|(bb, _)| bb);
                visit_results(
                    body,
                    blocks,
                    &results,
                    &mut Annotator { annotations: &mut annotations },
                );
            }
            _ => {}
        }
    }
    let extra_data = annotations.into_extra_data()?;
    if let Some(dumper) = MirDumper::new(tcx, "dataflow", body) {
        dumper.set_extra_data(&*extra_data).dump_mir(body);
    }
    Some(extra_data)
}

#[derive(Default)]
struct Annotations {
    entries: FxIndexMap<PassWhere, Vec<String>>,
}

impl Annotations {
    fn add(&mut self, pass_where: PassWhere, s: String) {
        self.entries.entry(pass_where).or_default().push(s);
    }

    fn into_extra_data(self) -> Option<Box<ExtraDataFn>> {
        if self.entries.is_empty() {
            return None;
        }
        Some(Box::new(move |pass_where: PassWhere, out: &mut dyn io::Write| {
            let Some(notes) = self.entries.get(&pass_where) else {
                return Ok(());
            };
            for s in notes {
                writeln!(out, "        // {s}")?;
            }
            Ok(())
        }))
    }
}

/// Examines dataflow results and collects extra annotations for inclusion in MIR pretty printing.
struct Annotator<'a> {
    annotations: &'a mut Annotations,
}

impl<'tcx> ResultsVisitor<'tcx, MaybeLiveLocals> for Annotator<'_> {
    fn visit_after_primary_statement_effect(
        &mut self,
        state: &<MaybeLiveLocals as Analysis<'tcx>>::Domain,
        _statement: &mir::Statement<'tcx>,
        location: mir::Location,
    ) {
        let live: Vec<Local> = state.iter().collect();
        self.annotations.add(PassWhere::BeforeLocation(location), format!("live: {live:?}"));
    }

    fn visit_after_primary_terminator_effect(
        &mut self,
        state: &<MaybeLiveLocals as Analysis<'tcx>>::Domain,
        _terminator: &mir::Terminator<'tcx>,
        location: mir::Location,
    ) {
        let live: Vec<Local> = state.iter().collect();
        self.annotations.add(PassWhere::BeforeLocation(location), format!("live: {live:?}"));
    }
}

impl<'tcx> ResultsVisitor<'tcx, MaybeTransitiveLiveLocals<'_>> for Annotator<'_> {
    fn visit_after_primary_statement_effect(
        &mut self,
        state: &<MaybeLiveLocals as Analysis<'tcx>>::Domain,
        _statement: &mir::Statement<'tcx>,
        location: mir::Location,
    ) {
        let live: Vec<Local> = state.iter().collect();
        self.annotations.add(PassWhere::BeforeLocation(location), format!("live: {live:?}"));
    }

    fn visit_after_primary_terminator_effect(
        &mut self,
        state: &<MaybeLiveLocals as Analysis<'tcx>>::Domain,
        _terminator: &mir::Terminator<'tcx>,
        location: mir::Location,
    ) {
        let live: Vec<Local> = state.iter().collect();
        self.annotations.add(PassWhere::BeforeLocation(location), format!("live: {live:?}"));
    }
}
