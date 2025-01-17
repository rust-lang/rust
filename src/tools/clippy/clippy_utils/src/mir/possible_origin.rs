use super::transitive_relation::TransitiveRelation;
use crate::ty::is_copy;
use rustc_data_structures::fx::FxHashMap;
use rustc_index::bit_set::DenseBitSet;
use rustc_lint::LateContext;
use rustc_middle::mir;

/// Collect possible borrowed for every `&mut` local.
/// For example, `_1 = &mut _2` generate _1: {_2,...}
/// Known Problems: not sure all borrowed are tracked
#[allow(clippy::module_name_repetitions)]
pub(super) struct PossibleOriginVisitor<'a, 'tcx> {
    possible_origin: TransitiveRelation,
    body: &'a mir::Body<'tcx>,
}

impl<'a, 'tcx> PossibleOriginVisitor<'a, 'tcx> {
    pub fn new(body: &'a mir::Body<'tcx>) -> Self {
        Self {
            possible_origin: TransitiveRelation::default(),
            body,
        }
    }

    pub fn into_map(self, cx: &LateContext<'tcx>) -> FxHashMap<mir::Local, DenseBitSet<mir::Local>> {
        let mut map = FxHashMap::default();
        for row in (1..self.body.local_decls.len()).map(mir::Local::from_usize) {
            if is_copy(cx, self.body.local_decls[row].ty) {
                continue;
            }

            let mut borrowers = self.possible_origin.reachable_from(row, self.body.local_decls.len());
            borrowers.remove(mir::Local::from_usize(0));
            if !borrowers.is_empty() {
                map.insert(row, borrowers);
            }
        }
        map
    }
}

impl<'tcx> mir::visit::Visitor<'tcx> for PossibleOriginVisitor<'_, 'tcx> {
    fn visit_assign(&mut self, place: &mir::Place<'tcx>, rvalue: &mir::Rvalue<'_>, _location: mir::Location) {
        let lhs = place.local;
        match rvalue {
            // Only consider `&mut`, which can modify origin place
            mir::Rvalue::Ref(_, mir::BorrowKind::Mut { .. }, borrowed) |
            // _2: &mut _;
            // _3 = move _2
            mir::Rvalue::Use(mir::Operand::Move(borrowed))  |
            // _3 = move _2 as &mut _;
            mir::Rvalue::Cast(_, mir::Operand::Move(borrowed), _)
                => {
                self.possible_origin.add(lhs, borrowed.local);
            },
            _ => {},
        }
    }
}
