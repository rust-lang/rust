//! Removes operations on ZST places, and convert ZST operands to constants.

use crate::MirPass;
use rustc_middle::mir::interpret::ConstValue;
use rustc_middle::mir::visit::*;
use rustc_middle::mir::*;
use rustc_middle::ty::{self, Ty, TyCtxt};

pub struct RemoveZsts;

impl<'tcx> MirPass<'tcx> for RemoveZsts {
    fn is_enabled(&self, sess: &rustc_session::Session) -> bool {
        sess.mir_opt_level() > 0
    }

    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        // Avoid query cycles (generators require optimized MIR for layout).
        if tcx.type_of(body.source.def_id()).instantiate_identity().is_generator() {
            return;
        }
        let param_env = tcx.param_env_reveal_all_normalized(body.source.def_id());
        let local_decls = &body.local_decls;
        let mut replacer = Replacer { tcx, param_env, local_decls };
        for var_debug_info in &mut body.var_debug_info {
            replacer.visit_var_debug_info(var_debug_info);
        }
        for (bb, data) in body.basic_blocks.as_mut_preserves_cfg().iter_enumerated_mut() {
            replacer.visit_basic_block_data(bb, data);
        }
    }
}

struct Replacer<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    local_decls: &'a LocalDecls<'tcx>,
}

/// A cheap, approximate check to avoid unnecessary `layout_of` calls.
fn maybe_zst(ty: Ty<'_>) -> bool {
    match ty.kind() {
        // maybe ZST (could be more precise)
        ty::Adt(..)
        | ty::Array(..)
        | ty::Closure(..)
        | ty::Tuple(..)
        | ty::Alias(ty::Opaque, ..) => true,
        // definitely ZST
        ty::FnDef(..) | ty::Never => true,
        // unreachable or can't be ZST
        _ => false,
    }
}

impl<'tcx> Replacer<'_, 'tcx> {
    fn known_to_be_zst(&self, ty: Ty<'tcx>) -> bool {
        if !maybe_zst(ty) {
            return false;
        }
        let Ok(layout) = self.tcx.layout_of(self.param_env.and(ty)) else {
            return false;
        };
        layout.is_zst()
    }

    fn make_zst(&self, ty: Ty<'tcx>) -> Constant<'tcx> {
        debug_assert!(self.known_to_be_zst(ty));
        Constant {
            span: rustc_span::DUMMY_SP,
            user_ty: None,
            literal: ConstantKind::Val(ConstValue::ZeroSized, ty),
        }
    }
}

impl<'tcx> MutVisitor<'tcx> for Replacer<'_, 'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn visit_var_debug_info(&mut self, var_debug_info: &mut VarDebugInfo<'tcx>) {
        match var_debug_info.value {
            VarDebugInfoContents::Const(_) => {}
            VarDebugInfoContents::Place(place) => {
                let place_ty = place.ty(self.local_decls, self.tcx).ty;
                if self.known_to_be_zst(place_ty) {
                    var_debug_info.value = VarDebugInfoContents::Const(self.make_zst(place_ty))
                }
            }
            VarDebugInfoContents::Composite { ty, fragments: _ } => {
                if self.known_to_be_zst(ty) {
                    var_debug_info.value = VarDebugInfoContents::Const(self.make_zst(ty))
                }
            }
        }
    }

    fn visit_operand(&mut self, operand: &mut Operand<'tcx>, loc: Location) {
        if let Operand::Constant(_) = operand {
            return;
        }
        let op_ty = operand.ty(self.local_decls, self.tcx);
        if self.known_to_be_zst(op_ty)
            && self.tcx.consider_optimizing(|| {
                format!("RemoveZsts - Operand: {:?} Location: {:?}", operand, loc)
            })
        {
            *operand = Operand::Constant(Box::new(self.make_zst(op_ty)))
        }
    }

    fn visit_statement(&mut self, statement: &mut Statement<'tcx>, loc: Location) {
        let place_for_ty = match statement.kind {
            StatementKind::Assign(box (place, ref rvalue)) => {
                rvalue.is_safe_to_remove().then_some(place)
            }
            StatementKind::Deinit(box place)
            | StatementKind::SetDiscriminant { box place, variant_index: _ }
            | StatementKind::AscribeUserType(box (place, _), _)
            | StatementKind::Retag(_, box place)
            | StatementKind::PlaceMention(box place)
            | StatementKind::FakeRead(box (_, place)) => Some(place),
            StatementKind::StorageLive(local) | StatementKind::StorageDead(local) => {
                Some(local.into())
            }
            StatementKind::Coverage(_)
            | StatementKind::Intrinsic(_)
            | StatementKind::Nop
            | StatementKind::ConstEvalCounter => None,
        };
        if let Some(place_for_ty) = place_for_ty
            && let ty = place_for_ty.ty(self.local_decls, self.tcx).ty
            && self.known_to_be_zst(ty)
            && self.tcx.consider_optimizing(|| {
                format!("RemoveZsts - Place: {:?} SourceInfo: {:?}", place_for_ty, statement.source_info)
            })
        {
            statement.make_nop();
        } else {
            self.super_statement(statement, loc);
        }
    }
}
