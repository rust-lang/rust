use rustc_index::IndexVec;
use rustc_index::bit_set::DenseBitSet;
use rustc_middle::bug;
use rustc_middle::mir::visit::{MutVisitor, PlaceContext, Visitor};
use rustc_middle::mir::*;
use rustc_middle::ty::TyCtxt;

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
    fn is_enabled(&self, sess: &rustc_session::Session) -> bool {
        sess.mir_opt_level() > 0
    }

    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        let mut finder = SingleUseConstsFinder {
            ineligible_locals: DenseBitSet::new_empty(body.local_decls.len()),
            locations: IndexVec::from_elem(LocationPair::new(), &body.local_decls),
            locals_in_debug_info: DenseBitSet::new_empty(body.local_decls.len()),
        };

        finder.ineligible_locals.insert_range(..=Local::from_usize(body.arg_count));

        finder.visit_body(body);

        for (local, locations) in finder.locations.iter_enumerated() {
            if finder.ineligible_locals.contains(local) {
                continue;
            }

            let Some(init_loc) = locations.init_loc else {
                continue;
            };

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
            let Rvalue::Use(operand) = rvalue else { bug!("No longer a use?") };

            let mut replacer = LocalReplacer { tcx, local, operand: Some(operand) };

            if finder.locals_in_debug_info.contains(local) {
                for var_debug_info in &mut body.var_debug_info {
                    replacer.visit_var_debug_info(var_debug_info);
                }
            }

            let Some(use_loc) = locations.use_loc else { continue };

            let use_block = &mut basic_blocks[use_loc.block];
            if let Some(use_statement) = use_block.statements.get_mut(use_loc.statement_index) {
                replacer.visit_statement(use_statement, use_loc);
            } else {
                replacer.visit_terminator(use_block.terminator_mut(), use_loc);
            }

            if replacer.operand.is_some() {
                bug!(
                    "operand wasn't used replacing local {local:?} with locations {locations:?} in body {body:#?}"
                );
            }
        }
    }

    fn is_required(&self) -> bool {
        true
    }
}

#[derive(Copy, Clone, Debug)]
struct LocationPair {
    init_loc: Option<Location>,
    use_loc: Option<Location>,
}

impl LocationPair {
    fn new() -> Self {
        Self { init_loc: None, use_loc: None }
    }
}

struct SingleUseConstsFinder {
    ineligible_locals: DenseBitSet<Local>,
    locations: IndexVec<Local, LocationPair>,
    locals_in_debug_info: DenseBitSet<Local>,
}

impl<'tcx> Visitor<'tcx> for SingleUseConstsFinder {
    fn visit_assign(&mut self, place: &Place<'tcx>, rvalue: &Rvalue<'tcx>, location: Location) {
        if let Some(local) = place.as_local()
            && let Rvalue::Use(operand) = rvalue
            && let Operand::Constant(_) = operand
        {
            let locations = &mut self.locations[local];
            if locations.init_loc.is_some() {
                self.ineligible_locals.insert(local);
            } else {
                locations.init_loc = Some(location);
            }
        } else {
            self.super_assign(place, rvalue, location);
        }
    }

    fn visit_operand(&mut self, operand: &Operand<'tcx>, location: Location) {
        if let Some(place) = operand.place()
            && let Some(local) = place.as_local()
        {
            let locations = &mut self.locations[local];
            if locations.use_loc.is_some() {
                self.ineligible_locals.insert(local);
            } else {
                locations.use_loc = Some(location);
            }
        } else {
            self.super_operand(operand, location);
        }
    }

    fn visit_statement(&mut self, statement: &Statement<'tcx>, location: Location) {
        match &statement.kind {
            // Storage markers are irrelevant to this.
            StatementKind::StorageLive(_) | StatementKind::StorageDead(_) => {}
            _ => self.super_statement(statement, location),
        }
    }

    fn visit_var_debug_info(&mut self, var_debug_info: &VarDebugInfo<'tcx>) {
        if let VarDebugInfoContents::Place(place) = &var_debug_info.value
            && let Some(local) = place.as_local()
        {
            self.locals_in_debug_info.insert(local);
        } else {
            self.super_var_debug_info(var_debug_info);
        }
    }

    fn visit_local(&mut self, local: Local, _context: PlaceContext, _location: Location) {
        // If there's any path that gets here, rather than being understood elsewhere,
        // then we'd better not do anything with this local.
        self.ineligible_locals.insert(local);
    }
}

struct LocalReplacer<'tcx> {
    tcx: TyCtxt<'tcx>,
    local: Local,
    operand: Option<Operand<'tcx>>,
}

impl<'tcx> MutVisitor<'tcx> for LocalReplacer<'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn visit_operand(&mut self, operand: &mut Operand<'tcx>, _location: Location) {
        if let Operand::Copy(place) | Operand::Move(place) = operand
            && let Some(local) = place.as_local()
            && local == self.local
        {
            *operand = self.operand.take().unwrap_or_else(|| {
                bug!("there was a second use of the operand");
            });
        }
    }

    fn visit_var_debug_info(&mut self, var_debug_info: &mut VarDebugInfo<'tcx>) {
        if let VarDebugInfoContents::Place(place) = &var_debug_info.value
            && let Some(local) = place.as_local()
            && local == self.local
        {
            let const_op = *self
                .operand
                .as_ref()
                .unwrap_or_else(|| {
                    bug!("the operand was already stolen");
                })
                .constant()
                .unwrap();
            var_debug_info.value = VarDebugInfoContents::Const(const_op);
        }
    }
}
