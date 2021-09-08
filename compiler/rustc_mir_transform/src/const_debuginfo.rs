//! Finds locals which are assigned once to a const and unused except for debuginfo and converts
//! their debuginfo to use the const directly, allowing the local to be removed.

use rustc_middle::{
    mir::{
        visit::{PlaceContext, Visitor},
        Body, Constant, Local, Location, Operand, Rvalue, StatementKind, VarDebugInfoContents,
    },
    ty::TyCtxt,
};

use crate::MirPass;
use rustc_index::{bit_set::BitSet, vec::IndexVec};

pub struct ConstDebugInfo;

impl<'tcx> MirPass<'tcx> for ConstDebugInfo {
    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        if !tcx.sess.opts.debugging_opts.unsound_mir_opts {
            return;
        }

        trace!("running ConstDebugInfo on {:?}", body.source);

        for (local, constant) in find_optimization_oportunities(body) {
            for debuginfo in &mut body.var_debug_info {
                if let VarDebugInfoContents::Place(p) = debuginfo.value {
                    if p.local == local && p.projection.is_empty() {
                        trace!(
                            "changing debug info for {:?} from place {:?} to constant {:?}",
                            debuginfo.name,
                            p,
                            constant
                        );
                        debuginfo.value = VarDebugInfoContents::Const(constant);
                    }
                }
            }
        }
    }
}

struct LocalUseVisitor {
    local_mutating_uses: IndexVec<Local, u8>,
    local_assignment_locations: IndexVec<Local, Option<Location>>,
}

fn find_optimization_oportunities<'tcx>(body: &Body<'tcx>) -> Vec<(Local, Constant<'tcx>)> {
    let mut visitor = LocalUseVisitor {
        local_mutating_uses: IndexVec::from_elem(0, &body.local_decls),
        local_assignment_locations: IndexVec::from_elem(None, &body.local_decls),
    };

    visitor.visit_body(body);

    let mut locals_to_debuginfo = BitSet::new_empty(body.local_decls.len());
    for debuginfo in &body.var_debug_info {
        if let VarDebugInfoContents::Place(p) = debuginfo.value {
            if let Some(l) = p.as_local() {
                locals_to_debuginfo.insert(l);
            }
        }
    }

    let mut eligable_locals = Vec::new();
    for (local, mutating_uses) in visitor.local_mutating_uses.drain_enumerated(..) {
        if mutating_uses != 1 || !locals_to_debuginfo.contains(local) {
            continue;
        }

        if let Some(location) = visitor.local_assignment_locations[local] {
            let bb = &body[location.block];

            // The value is assigned as the result of a call, not a constant
            if bb.statements.len() == location.statement_index {
                continue;
            }

            if let StatementKind::Assign(box (p, Rvalue::Use(Operand::Constant(box c)))) =
                &bb.statements[location.statement_index].kind
            {
                if let Some(local) = p.as_local() {
                    eligable_locals.push((local, *c));
                }
            }
        }
    }

    eligable_locals
}

impl<'tcx> Visitor<'tcx> for LocalUseVisitor {
    fn visit_local(&mut self, local: &Local, context: PlaceContext, location: Location) {
        if context.is_mutating_use() {
            self.local_mutating_uses[*local] = self.local_mutating_uses[*local].saturating_add(1);

            if context.is_place_assignment() {
                self.local_assignment_locations[*local] = Some(location);
            }
        }
    }
}
