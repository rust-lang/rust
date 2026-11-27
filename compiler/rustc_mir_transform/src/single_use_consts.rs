use rustc_index::IndexVec;
use rustc_middle::mir::visit::{MutVisitor, PlaceContext, Visitor};
use rustc_middle::mir::*;
use rustc_middle::ty::TyCtxt;
use rustc_span::bug;

use crate::PassPolicy;
use crate::strip_debuginfo::drop_invalid_debuginfos;

/// Various parts of MIR building introduce temporaries that are commonly not needed.
///
/// Notably, `if CONST` and `match CONST` end up being used-once temporaries, which
/// obfuscates the structure for other passes and codegen, which would like to always
/// be able to just see the constant directly.
///
/// At higher optimization levels fancier passes like GVN will take care of this
/// in a more general fashion, but this handles the easy cases so can run in debug.
///
/// This only removes constants with a single-use because re-evaluating constants
/// isn't always an improvement, especially for large ones.
///
/// It also removes *never*-used constants, since it had all the information
/// needed to do that too, including updating the debug info.
pub(super) struct SingleUseConsts;

impl<'tcx> crate::MirPass<'tcx> for SingleUseConsts {
    fn policy(&self, ctx: &crate::PassCtx<'_>) -> PassPolicy {
        PassPolicy::optional(ctx.mir_opt_level() >= 1)
    }

    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        let mut finder = SingleUseConstsFinder {
            candidates: IndexVec::from_fn_n(
                |_| Status::Prospect { init_loc: None, use_loc: None },
                body.local_decls.len(),
            ),
        };
        finder.visit_body(body);
        let mut candidates = finder.candidates;

        let mut changed = false;
        for (local, status) in candidates.iter_enumerated_mut() {
            if let LocalKind::ReturnPointer | LocalKind::Arg = body.local_kind(local) {
                status.reject();
                continue;
            }
            let Status::Prospect { init_loc: Some(init_loc), use_loc } = *status else {
                status.reject();
                continue;
            };
            changed = true;

            // We're only changing an operand, not the terminator kinds or successors
            let basic_blocks = body.basic_blocks.as_mut_preserves_cfg();
            let init_statement_kind = std::mem::replace(
                &mut basic_blocks[init_loc.block].statements[init_loc.statement_index].kind,
                StatementKind::Nop,
            );
            let StatementKind::Assign(place_and_rvalue) = init_statement_kind else {
                bug!("No longer an assign?");
            };
            let (place, rvalue) = *place_and_rvalue;
            assert_eq!(place.as_local(), Some(local));
            let Rvalue::Use(Operand::Constant(const_op), _) = rvalue else {
                bug!("No longer a use?")
            };
            *status = Status::Replaced { const_op: Some(const_op), use_loc };
        }

        if !changed {
            return;
        }

        let mut replacer = LocalReplacer { tcx, candidates };
        for var_debug_info in &mut body.var_debug_info {
            replacer.visit_var_debug_info(var_debug_info);
        }
        let basic_blocks = body.basic_blocks.as_mut_preserves_cfg();
        for local in body.local_decls.indices() {
            let Status::Replaced { use_loc: Some(use_loc), .. } = replacer.candidates[local] else {
                continue;
            };
            let use_block = &mut basic_blocks[use_loc.block];
            if let Some(use_statement) = use_block.statements.get_mut(use_loc.statement_index) {
                replacer.visit_statement(use_statement, use_loc);
            } else {
                replacer.visit_terminator(&mut use_block.terminator, use_loc);
            }
            if replacer.candidates[local].take_const().is_some() {
                bug!(
                    "operand wasn't used replacing local {local:?} with location {use_loc:?} in body {body:#?}"
                );
            }
        }

        drop_invalid_debuginfos(body);
    }
}

#[derive(Debug)]
enum Status<'tcx> {
    /// Local is illegible for the optimization.
    Rejected,
    /// Local is a potential candidate for the optimization.
    Prospect { init_loc: Option<Location>, use_loc: Option<Location> },
    /// Local has been replaced with a constant.
    Replaced { const_op: Option<Box<ConstOperand<'tcx>>>, use_loc: Option<Location> },
}

impl<'tcx> Status<'tcx> {
    #[inline]
    fn insert_init_location(&mut self, location: Location) {
        match self {
            Status::Prospect { init_loc: init_loc @ None, .. } => {
                *init_loc = Some(location);
            }
            _ => *self = Status::Rejected,
        }
    }

    #[inline]
    fn insert_use_location(&mut self, location: Location) {
        match self {
            Status::Prospect { use_loc: use_loc @ None, .. } => {
                *use_loc = Some(location);
            }
            _ => *self = Status::Rejected,
        }
    }

    #[inline]
    fn reject(&mut self) {
        *self = Status::Rejected;
    }

    #[inline]
    fn as_const(&self) -> Option<&ConstOperand<'tcx>> {
        match self {
            Status::Replaced { const_op, .. } => const_op.as_deref(),
            _ => None,
        }
    }

    #[inline]
    fn take_const(&mut self) -> Option<Box<ConstOperand<'tcx>>> {
        match self {
            Status::Replaced { const_op, .. } => const_op.take(),
            _ => None,
        }
    }
}

struct SingleUseConstsFinder<'tcx> {
    candidates: IndexVec<Local, Status<'tcx>>,
}

impl<'tcx> Visitor<'tcx> for SingleUseConstsFinder<'tcx> {
    fn visit_assign(&mut self, place: &Place<'tcx>, rvalue: &Rvalue<'tcx>, location: Location) {
        if let Some(local) = place.as_local()
            && let Rvalue::Use(operand, _) = rvalue
            && let Operand::Constant(_) = operand
        {
            self.candidates[local].insert_init_location(location);
        } else {
            self.super_assign(place, rvalue, location);
        }
    }

    fn visit_operand(&mut self, operand: &Operand<'tcx>, location: Location) {
        if let Some(place) = operand.place()
            && let Some(local) = place.as_local()
        {
            self.candidates[local].insert_use_location(location);
        } else {
            self.super_operand(operand, location);
        }
    }

    fn visit_statement(&mut self, statement: &Statement<'tcx>, location: Location) {
        match &statement.kind {
            // Storage markers are irrelevant to this.
            StatementKind::StorageLive(_) | StatementKind::StorageDead(_) => {
                // Accepted.
            }
            _ => self.super_statement(statement, location),
        }
    }

    fn visit_var_debug_info(&mut self, var_debug_info: &VarDebugInfo<'tcx>) {
        if let VarDebugInfoContents::Place(place) = &var_debug_info.value
            && let Some(_) = place.as_local()
        {
            // Accepted.
        } else {
            self.super_var_debug_info(var_debug_info);
        }
    }

    fn visit_local(&mut self, local: Local, _context: PlaceContext, _location: Location) {
        // If there's any path that gets here, rather than being understood elsewhere,
        // then we'd better not do anything with this local.
        self.candidates[local].reject();
    }
}

struct LocalReplacer<'tcx> {
    tcx: TyCtxt<'tcx>,
    candidates: IndexVec<Local, Status<'tcx>>,
}

impl<'tcx> MutVisitor<'tcx> for LocalReplacer<'tcx> {
    #[inline]
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn visit_operand(&mut self, operand: &mut Operand<'tcx>, _location: Location) {
        if let Operand::Copy(place) | Operand::Move(place) = operand
            && let Some(local) = place.as_local()
            && let Some(const_op) = self.candidates[local].take_const()
        {
            *operand = Operand::Constant(const_op);
        }
    }

    fn visit_var_debug_info(&mut self, var_debug_info: &mut VarDebugInfo<'tcx>) {
        if let VarDebugInfoContents::Place(place) = &var_debug_info.value
            && let Some(local) = place.as_local()
            && let Some(const_op) = self.candidates[local].as_const()
        {
            var_debug_info.value = VarDebugInfoContents::Const(*const_op);
        }
    }
}
