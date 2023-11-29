use std::assert_matches::assert_matches;
use std::collections::hash_map::Entry;

use rustc_data_structures::fx::FxHashMap;
use rustc_middle::mir::coverage::{BlockMarkerId, BranchSpan, CoverageKind};
use rustc_middle::mir::{self, BasicBlock, UnOp};
use rustc_middle::thir::{ExprId, ExprKind, Thir};
use rustc_middle::ty::TyCtxt;
use rustc_span::def_id::LocalDefId;

use crate::build::Builder;

pub(crate) struct HirBranchInfoBuilder {
    /// Maps condition expressions to their enclosing `!`, for better instrumentation.
    inversions: FxHashMap<ExprId, Inversion>,

    num_block_markers: usize,
    branch_spans: Vec<BranchSpan>,
}

#[derive(Clone, Copy)]
struct Inversion {
    /// When visiting the associated expression as a branch condition, treat this
    /// enclosing `!` as the branch condition instead.
    enclosing_not: ExprId,
    /// True if the associated expression is nested within an odd number of `!`
    /// expressions relative to `enclosing_not` (inclusive of `enclosing_not`).
    is_inverted: bool,
}

impl HirBranchInfoBuilder {
    /// Creates a new branch info builder, but only if branch coverage instrumentation
    /// is enabled and `def_id` represents a function that is eligible for coverage.
    pub(crate) fn new_if_enabled(tcx: TyCtxt<'_>, def_id: LocalDefId) -> Option<Self> {
        if tcx.sess.instrument_coverage_branch() && tcx.is_eligible_for_coverage(def_id) {
            Some(Self {
                inversions: FxHashMap::default(),
                num_block_markers: 0,
                branch_spans: vec![],
            })
        } else {
            None
        }
    }

    /// Unary `!` expressions inside an `if` condition are lowered by lowering
    /// their argument instead, and then reversing the then/else arms of that `if`.
    ///
    /// That's awkward for branch coverage instrumentation, so to work around that
    /// we pre-emptively visit any affected `!` expressions, and record extra
    /// information that [`Builder::visit_coverage_branch_condition`] can use to
    /// synthesize branch instrumentation for the enclosing `!`.
    pub(crate) fn visit_unary_not(&mut self, thir: &Thir<'_>, unary_not: ExprId) {
        assert_matches!(thir[unary_not].kind, ExprKind::Unary { op: UnOp::Not, .. });

        self.visit_inverted(
            thir,
            unary_not,
            // Set `is_inverted: false` for the `!` itself, so that its enclosed
            // expression will have `is_inverted: true`.
            Inversion { enclosing_not: unary_not, is_inverted: false },
        );
    }

    fn visit_inverted(&mut self, thir: &Thir<'_>, expr_id: ExprId, inversion: Inversion) {
        match self.inversions.entry(expr_id) {
            // This expression has already been marked by an enclosing `!`.
            Entry::Occupied(_) => return,
            Entry::Vacant(entry) => entry.insert(inversion),
        };

        match thir[expr_id].kind {
            ExprKind::Unary { op: UnOp::Not, arg } => {
                // Flip the `is_inverted` flag for the contents of this `!`.
                let inversion = Inversion { is_inverted: !inversion.is_inverted, ..inversion };
                self.visit_inverted(thir, arg, inversion);
            }
            ExprKind::Scope { value, .. } => self.visit_inverted(thir, value, inversion),
            ExprKind::Use { source } => self.visit_inverted(thir, source, inversion),
            // All other expressions (including `&&` and `||`) don't need any
            // special handling of their contents, so stop visiting.
            _ => {}
        }
    }

    fn next_block_marker_id(&mut self) -> BlockMarkerId {
        let id = BlockMarkerId::from_usize(self.num_block_markers);
        self.num_block_markers += 1;
        id
    }

    pub(crate) fn into_done(self) -> Option<Box<mir::coverage::HirBranchInfo>> {
        let Self { inversions: _, num_block_markers, branch_spans } = self;

        if num_block_markers == 0 {
            assert!(branch_spans.is_empty());
            return None;
        }

        Some(Box::new(mir::coverage::HirBranchInfo { num_block_markers, branch_spans }))
    }
}

impl Builder<'_, '_> {
    /// If branch coverage is enabled, inject marker statements into `then_block`
    /// and `else_block`, and record their IDs in the table of branch spans.
    pub(crate) fn visit_coverage_branch_condition(
        &mut self,
        expr_id: ExprId,
        then_block: BasicBlock,
        else_block: BasicBlock,
    ) {
        // Bail out if branch coverage is not enabled for this function.
        let Some(branch_info) = self.coverage_branch_info.as_ref() else { return };

        // If this condition expression is nested within one or more `!` expressions,
        // replace it with the enclosing `!` collected by `visit_unary_not`.
        let (expr_id, is_inverted) = match branch_info.inversions.get(&expr_id) {
            Some(&Inversion { enclosing_not, is_inverted }) => (enclosing_not, is_inverted),
            None => (expr_id, false),
        };
        let source_info = self.source_info(self.thir[expr_id].span);

        // Now that we have `source_info`, we can upgrade to a &mut reference.
        let branch_info = self.coverage_branch_info.as_mut().expect("upgrading & to &mut");

        let mut inject_branch_marker = |block: BasicBlock| {
            let id = branch_info.next_block_marker_id();

            let marker_statement = mir::Statement {
                source_info,
                kind: mir::StatementKind::Coverage(Box::new(mir::Coverage {
                    kind: CoverageKind::BlockMarker { id },
                })),
            };
            self.cfg.push(block, marker_statement);

            id
        };

        let mut true_marker = inject_branch_marker(then_block);
        let mut false_marker = inject_branch_marker(else_block);
        if is_inverted {
            std::mem::swap(&mut true_marker, &mut false_marker);
        }

        branch_info.branch_spans.push(BranchSpan {
            span: source_info.span,
            true_marker,
            false_marker,
        });
    }
}
