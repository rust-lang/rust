use crate::MirPass;
use rustc_middle::mir::visit::MutVisitor;
use rustc_middle::mir::*;
use rustc_middle::ty::{FnDef, TyCtxt};

/// Within-block replacement empowers ConstProp, DataflowConstProp, and DestProp. (based on looking
/// at the tests)
///
/// SwitchInt replacement causes some basic blocks which only contain a goto to be deleted. (based
/// on looking at core and alloc)
/// It also makes EarlyOtherwiseBranch much more effective, but that pass is currently marked as
/// unsound so this effect is not useful yet.
///
/// At time of writing, Assert replacement has no effect. Most likely we don't often need the
/// asserted predicate for anything else, and aren't smart enough to optimize into a form that
/// could use it anyway.
///
/// Enabling this pass for Call arguments breaks the moved-from local reuse optimization that
/// Inline does, so without DestinationPropagation, modifying call arguments is just a regression.
/// Except for calls to intrinsics, because those cannot be inlined.
pub struct MoveToCopy;

impl<'tcx> MirPass<'tcx> for MoveToCopy {
    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        let mut visitor = MoveToCopyVisitor { tcx };
        let mut call_visitor = MoveToCopyCallVisitor { tcx, local_decls: &body.local_decls };
        for (block, block_data) in body.basic_blocks.as_mut().iter_enumerated_mut() {
            for (statement_index, statement) in block_data.statements.iter_mut().enumerate() {
                visitor.visit_statement(statement, Location { block, statement_index });
            }
            let Some(terminator) = &mut block_data.terminator else {
                continue;
            };
            match &terminator.kind {
                TerminatorKind::SwitchInt { .. } | TerminatorKind::Assert { .. } => {
                    visitor.visit_terminator(
                        terminator,
                        Location { block, statement_index: block_data.statements.len() },
                    );
                }
                TerminatorKind::Call { func, .. } => {
                    let func_ty = func.ty(call_visitor.local_decls, tcx);
                    let is_intrinsic = if let FnDef(def_id, _) = *func_ty.kind() {
                        tcx.is_intrinsic(def_id)
                    } else {
                        false
                    };
                    if is_intrinsic || tcx.sess.mir_opt_level() >= 3 {
                        call_visitor.visit_terminator(
                            terminator,
                            Location { block, statement_index: block_data.statements.len() },
                        );
                    }
                }
                _ => {}
            }
        }
    }
}

struct MoveToCopyCallVisitor<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    local_decls: &'a LocalDecls<'tcx>,
}

impl<'a, 'tcx> MutVisitor<'tcx> for MoveToCopyCallVisitor<'a, 'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn visit_operand(&mut self, operand: &mut Operand<'tcx>, location: Location) {
        if let Operand::Move(place) = operand {
            let ty = place.ty(self.local_decls, self.tcx).ty;
            if ty.is_numeric() || ty.is_bool() || ty.is_char() {
                *operand = Operand::Copy(*place);
            }
        }
        self.super_operand(operand, location);
    }
}

struct MoveToCopyVisitor<'tcx> {
    tcx: TyCtxt<'tcx>,
}

impl<'tcx> MutVisitor<'tcx> for MoveToCopyVisitor<'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn visit_operand(&mut self, operand: &mut Operand<'tcx>, location: Location) {
        if let Operand::Move(place) = operand {
            *operand = Operand::Copy(*place);
        }
        self.super_operand(operand, location);
    }
}
