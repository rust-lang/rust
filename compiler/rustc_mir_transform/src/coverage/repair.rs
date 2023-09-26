use rustc_middle::mir::coverage::{CounterId, CoverageKind};
use rustc_middle::mir::{
    self, Coverage, MirPass, SourceInfo, Statement, StatementKind, START_BLOCK,
};
use rustc_middle::ty::TyCtxt;
use rustc_span::DUMMY_SP;

/// If a function has been [instrumented for coverage](super::InstrumentCoverage),
/// but MIR optimizations subsequently remove all of its [`CoverageKind::CounterIncrement`]
/// statements (e.g. because bb0 is unreachable), then we won't generate any
/// `llvm.instrprof.increment` intrinsics. LLVM will think the function is not
/// instrumented, and it will disappear from coverage mappings and coverage reports.
///
/// This pass detects when that has happened, and re-inserts a dummy counter-increment
/// statement so that LLVM knows to treat the function as instrumented.
pub struct RepairCoverage;

impl<'tcx> MirPass<'tcx> for RepairCoverage {
    fn is_enabled(&self, sess: &rustc_session::Session) -> bool {
        sess.instrument_coverage()
    }

    fn run_pass(&self, _tcx: TyCtxt<'tcx>, body: &mut mir::Body<'tcx>) {
        // If a function wasn't instrumented for coverage in the first place,
        // then there's no need to repair anything.
        if body.function_coverage_info.is_none() {
            return;
        }

        // If the body still contains one or more counter-increment statements,
        // there's no need to repair anything.
        let has_counter = body
            .basic_blocks
            .iter()
            .flat_map(|bb_data| &bb_data.statements)
            .filter_map(|statement| match statement.kind {
                StatementKind::Coverage(box ref coverage) => Some(coverage),
                _ => None,
            })
            .any(|coverage| matches!(coverage.kind, CoverageKind::CounterIncrement { .. }));
        if has_counter {
            return;
        }

        debug!(
            "all counters were removed after instrumentation; restoring one counter in {:?}",
            body.source.def_id()
        );

        let statement = Statement {
            source_info: SourceInfo::outermost(DUMMY_SP),
            kind: StatementKind::Coverage(Box::new(Coverage {
                kind: CoverageKind::CounterIncrement { id: CounterId::START },
            })),
        };
        body[START_BLOCK].statements.insert(0, statement);
    }
}
