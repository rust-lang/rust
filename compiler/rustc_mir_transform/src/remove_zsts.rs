//! Removes operations on ZST places, and convert ZST operands to constants.

use rustc_middle::mir::visit::*;
use rustc_middle::mir::*;
use rustc_middle::ty::{self, Ty, TyCtxt};

pub(super) struct RemoveZsts;

impl<'tcx> crate::MirPass<'tcx> for RemoveZsts {
    fn is_enabled(&self, sess: &rustc_session::Session) -> bool {
        sess.mir_opt_level() > 0
    }

    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        // Avoid query cycles (coroutines require optimized MIR for layout).
        if tcx.type_of(body.source.def_id()).instantiate_identity().is_coroutine() {
            return;
        }

        let typing_env = body.typing_env(tcx);
        let local_decls = &body.local_decls;
        let mut replacer = Replacer { tcx, typing_env, local_decls };
        for var_debug_info in &mut body.var_debug_info {
            replacer.visit_var_debug_info(var_debug_info);
        }
        for (bb, data) in body.basic_blocks.as_mut_preserves_cfg().iter_enumerated_mut() {
            replacer.visit_basic_block_data(bb, data);
        }
    }

    fn is_required(&self) -> bool {
        true
    }
}

struct Replacer<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    typing_env: ty::TypingEnv<'tcx>,
    local_decls: &'a LocalDecls<'tcx>,
}

/// A cheap, approximate check to avoid unnecessary `layout_of` calls.
///
/// `Some(true)` is definitely ZST; `Some(false)` is definitely *not* ZST.
///
/// `None` may or may not be, and must check `layout_of` to be sure.
fn trivially_zst<'tcx>(ty: Ty<'tcx>, tcx: TyCtxt<'tcx>) -> Option<bool> {
    match ty.kind() {
        // definitely ZST
        ty::FnDef(..) | ty::Never => Some(true),
        ty::Tuple(fields) if fields.is_empty() => Some(true),
        ty::Array(_ty, len) if let Some(0) = len.try_to_target_usize(tcx) => Some(true),
        // clearly not ZST
        ty::Bool
        | ty::Char
        | ty::Int(..)
        | ty::Uint(..)
        | ty::Float(..)
        | ty::RawPtr(..)
        | ty::Ref(..)
        | ty::FnPtr(..) => Some(false),
        ty::Coroutine(def_id, _) => {
            // For async_drop_in_place::{closure} this is load bearing, not just a perf fix,
            // because we don't want to compute the layout before mir analysis is done
            if tcx.is_async_drop_in_place_coroutine(*def_id) { Some(false) } else { None }
        }
        // check `layout_of` to see (including unreachable things we won't actually see)
        _ => None,
    }
}

impl<'tcx> Replacer<'_, 'tcx> {
    fn known_to_be_zst(&self, ty: Ty<'tcx>) -> bool {
        if let Some(is_zst) = trivially_zst(ty, self.tcx) {
            is_zst
        } else {
            self.tcx
                .layout_of(self.typing_env.as_query_input(ty))
                .is_ok_and(|layout| layout.is_zst())
        }
    }

    fn make_zst(&self, ty: Ty<'tcx>) -> ConstOperand<'tcx> {
        debug_assert!(self.known_to_be_zst(ty));
        ConstOperand {
            span: rustc_span::DUMMY_SP,
            user_ty: None,
            const_: Const::Val(ConstValue::ZeroSized, ty),
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
        }
    }

    fn visit_operand(&mut self, operand: &mut Operand<'tcx>, _: Location) {
        if let Operand::Constant(_) = operand {
            return;
        }
        let op_ty = operand.ty(self.local_decls, self.tcx);
        if self.known_to_be_zst(op_ty) {
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
            | StatementKind::BackwardIncompatibleDropHint { .. }
            | StatementKind::ConstEvalCounter => None,
        };
        if let Some(place_for_ty) = place_for_ty
            && let ty = place_for_ty.ty(self.local_decls, self.tcx).ty
            && self.known_to_be_zst(ty)
        {
            statement.make_nop(true);
        } else {
            self.super_statement(statement, loc);
        }
    }
}
