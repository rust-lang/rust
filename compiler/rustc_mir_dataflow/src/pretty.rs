use std::io;

use rustc_attr_ir::{RustcMirKind, find_attr};
use rustc_data_structures::fx::FxIndexMap;
use rustc_middle::mir::{self, Body, Local, PassWhere, traversal};
use rustc_middle::ty::TyCtxt;

use crate::debuginfo::debuginfo_locals;
use crate::framework::Analysis;
use crate::impls::{MaybeLiveLocals, MaybeTransitiveLiveLocals, borrowed_locals};
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
    annotations.into_extra_data()
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
