//! Performs various peephole optimizations.

use crate::MirPass;
use rustc_hir::Mutability;
use rustc_middle::mir::{
    BinOp, Body, Constant, ConstantKind, LocalDecls, Operand, Place, ProjectionElem, Rvalue,
    SourceInfo, Statement, StatementKind, Terminator, TerminatorKind, UnOp,
};
use rustc_middle::ty::layout::LayoutError;
use rustc_middle::ty::{self, ParamEnv, ParamEnvAnd, SubstsRef, Ty, TyCtxt};
use rustc_span::symbol::{sym, Symbol};

pub struct InstCombine;

impl<'tcx> MirPass<'tcx> for InstCombine {
    fn is_enabled(&self, sess: &rustc_session::Session) -> bool {
        sess.mir_opt_level() > 0
    }

    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        let ctx = InstCombineContext {
            tcx,
            local_decls: &body.local_decls,
            param_env: tcx.param_env_reveal_all_normalized(body.source.def_id()),
        };
        for block in body.basic_blocks.as_mut() {
            for statement in block.statements.iter_mut() {
                match statement.kind {
                    StatementKind::Assign(box (_place, ref mut rvalue)) => {
                        ctx.combine_bool_cmp(&statement.source_info, rvalue);
                        ctx.combine_ref_deref(&statement.source_info, rvalue);
                        ctx.combine_len(&statement.source_info, rvalue);
                        ctx.combine_cast(&statement.source_info, rvalue);
                    }
                    _ => {}
                }
            }

            ctx.combine_primitive_clone(
                &mut block.terminator.as_mut().unwrap(),
                &mut block.statements,
            );
            ctx.combine_intrinsic_assert(
                &mut block.terminator.as_mut().unwrap(),
                &mut block.statements,
            );
        }
    }
}

struct InstCombineContext<'tcx, 'a> {
    tcx: TyCtxt<'tcx>,
    local_decls: &'a LocalDecls<'tcx>,
    param_env: ParamEnv<'tcx>,
}

impl<'tcx> InstCombineContext<'tcx, '_> {
    fn should_combine(&self, source_info: &SourceInfo, rvalue: &Rvalue<'tcx>) -> bool {
        self.tcx.consider_optimizing(|| {
            format!("InstCombine - Rvalue: {:?} SourceInfo: {:?}", rvalue, source_info)
        })
    }

    /// Transform boolean comparisons into logical operations.
    fn combine_bool_cmp(&self, source_info: &SourceInfo, rvalue: &mut Rvalue<'tcx>) {
        match rvalue {
            Rvalue::BinaryOp(op @ (BinOp::Eq | BinOp::Ne), box (a, b)) => {
                let new = match (op, self.try_eval_bool(a), self.try_eval_bool(b)) {
                    // Transform "Eq(a, true)" ==> "a"
                    (BinOp::Eq, _, Some(true)) => Some(Rvalue::Use(a.clone())),

                    // Transform "Ne(a, false)" ==> "a"
                    (BinOp::Ne, _, Some(false)) => Some(Rvalue::Use(a.clone())),

                    // Transform "Eq(true, b)" ==> "b"
                    (BinOp::Eq, Some(true), _) => Some(Rvalue::Use(b.clone())),

                    // Transform "Ne(false, b)" ==> "b"
                    (BinOp::Ne, Some(false), _) => Some(Rvalue::Use(b.clone())),

                    // Transform "Eq(false, b)" ==> "Not(b)"
                    (BinOp::Eq, Some(false), _) => Some(Rvalue::UnaryOp(UnOp::Not, b.clone())),

                    // Transform "Ne(true, b)" ==> "Not(b)"
                    (BinOp::Ne, Some(true), _) => Some(Rvalue::UnaryOp(UnOp::Not, b.clone())),

                    // Transform "Eq(a, false)" ==> "Not(a)"
                    (BinOp::Eq, _, Some(false)) => Some(Rvalue::UnaryOp(UnOp::Not, a.clone())),

                    // Transform "Ne(a, true)" ==> "Not(a)"
                    (BinOp::Ne, _, Some(true)) => Some(Rvalue::UnaryOp(UnOp::Not, a.clone())),

                    _ => None,
                };

                if let Some(new) = new && self.should_combine(source_info, rvalue) {
                    *rvalue = new;
                }
            }

            _ => {}
        }
    }

    fn try_eval_bool(&self, a: &Operand<'_>) -> Option<bool> {
        let a = a.constant()?;
        if a.literal.ty().is_bool() { a.literal.try_to_bool() } else { None }
    }

    /// Transform "&(*a)" ==> "a".
    fn combine_ref_deref(&self, source_info: &SourceInfo, rvalue: &mut Rvalue<'tcx>) {
        if let Rvalue::Ref(_, _, place) = rvalue {
            if let Some((base, ProjectionElem::Deref)) = place.as_ref().last_projection() {
                if rvalue.ty(self.local_decls, self.tcx) != base.ty(self.local_decls, self.tcx).ty {
                    return;
                }

                if !self.should_combine(source_info, rvalue) {
                    return;
                }

                *rvalue = Rvalue::Use(Operand::Copy(Place {
                    local: base.local,
                    projection: self.tcx.mk_place_elems(base.projection),
                }));
            }
        }
    }

    /// Transform "Len([_; N])" ==> "N".
    fn combine_len(&self, source_info: &SourceInfo, rvalue: &mut Rvalue<'tcx>) {
        if let Rvalue::Len(ref place) = *rvalue {
            let place_ty = place.ty(self.local_decls, self.tcx).ty;
            if let ty::Array(_, len) = *place_ty.kind() {
                if !self.should_combine(source_info, rvalue) {
                    return;
                }

                let literal = ConstantKind::from_const(len, self.tcx);
                let constant = Constant { span: source_info.span, literal, user_ty: None };
                *rvalue = Rvalue::Use(Operand::Constant(Box::new(constant)));
            }
        }
    }

    fn combine_cast(&self, _source_info: &SourceInfo, rvalue: &mut Rvalue<'tcx>) {
        if let Rvalue::Cast(_kind, operand, ty) = rvalue {
            if operand.ty(self.local_decls, self.tcx) == *ty {
                *rvalue = Rvalue::Use(operand.clone());
            }
        }
    }

    fn combine_primitive_clone(
        &self,
        terminator: &mut Terminator<'tcx>,
        statements: &mut Vec<Statement<'tcx>>,
    ) {
        let TerminatorKind::Call { func, args, destination, target, .. } = &mut terminator.kind
        else { return };

        // It's definitely not a clone if there are multiple arguments
        if args.len() != 1 {
            return;
        }

        let Some(destination_block) = *target
        else { return };

        // Only bother looking more if it's easy to know what we're calling
        let Some((fn_def_id, fn_substs)) = func.const_fn_def()
        else { return };

        // Clone needs one subst, so we can cheaply rule out other stuff
        if fn_substs.len() != 1 {
            return;
        }

        // These types are easily available from locals, so check that before
        // doing DefId lookups to figure out what we're actually calling.
        let arg_ty = args[0].ty(self.local_decls, self.tcx);

        let ty::Ref(_region, inner_ty, Mutability::Not) = *arg_ty.kind()
        else { return };

        if !inner_ty.is_trivially_pure_clone_copy() {
            return;
        }

        let trait_def_id = self.tcx.trait_of_item(fn_def_id);
        if trait_def_id.is_none() || trait_def_id != self.tcx.lang_items().clone_trait() {
            return;
        }

        if !self.tcx.consider_optimizing(|| {
            format!(
                "InstCombine - Call: {:?} SourceInfo: {:?}",
                (fn_def_id, fn_substs),
                terminator.source_info
            )
        }) {
            return;
        }

        let Some(arg_place) = args.pop().unwrap().place()
        else { return };

        statements.push(Statement {
            source_info: terminator.source_info,
            kind: StatementKind::Assign(Box::new((
                *destination,
                Rvalue::Use(Operand::Copy(
                    arg_place.project_deeper(&[ProjectionElem::Deref], self.tcx),
                )),
            ))),
        });
        terminator.kind = TerminatorKind::Goto { target: destination_block };
    }

    fn combine_intrinsic_assert(
        &self,
        terminator: &mut Terminator<'tcx>,
        _statements: &mut Vec<Statement<'tcx>>,
    ) {
        let TerminatorKind::Call { func, target, .. } = &mut terminator.kind  else { return; };
        let Some(target_block) = target else { return; };
        let func_ty = func.ty(self.local_decls, self.tcx);
        let Some((intrinsic_name, substs)) = resolve_rust_intrinsic(self.tcx, func_ty) else {
            return;
        };
        // The intrinsics we are interested in have one generic parameter
        if substs.is_empty() {
            return;
        }
        let ty = substs.type_at(0);

        // Check this is a foldable intrinsic before we query the layout of our generic parameter
        let Some(assert_panics) = intrinsic_assert_panics(intrinsic_name) else { return; };
        match assert_panics(self.tcx, self.param_env.and(ty)) {
            // We don't know the layout, don't touch the assertion
            Err(_) => {}
            Ok(true) => {
                // If we know the assert panics, indicate to later opts that the call diverges
                *target = None;
            }
            Ok(false) => {
                // If we know the assert does not panic, turn the call into a Goto
                terminator.kind = TerminatorKind::Goto { target: *target_block };
            }
        }
    }
}

fn intrinsic_assert_panics<'tcx>(
    intrinsic_name: Symbol,
) -> Option<fn(TyCtxt<'tcx>, ParamEnvAnd<'tcx, Ty<'tcx>>) -> Result<bool, LayoutError<'tcx>>> {
    fn inhabited_predicate<'tcx>(
        tcx: TyCtxt<'tcx>,
        param_env_and_ty: ParamEnvAnd<'tcx, Ty<'tcx>>,
    ) -> Result<bool, LayoutError<'tcx>> {
        Ok(tcx.layout_of(param_env_and_ty)?.abi.is_uninhabited())
    }
    fn zero_valid_predicate<'tcx>(
        tcx: TyCtxt<'tcx>,
        param_env_and_ty: ParamEnvAnd<'tcx, Ty<'tcx>>,
    ) -> Result<bool, LayoutError<'tcx>> {
        Ok(!tcx.permits_zero_init(param_env_and_ty)?)
    }
    fn mem_uninitialized_valid_predicate<'tcx>(
        tcx: TyCtxt<'tcx>,
        param_env_and_ty: ParamEnvAnd<'tcx, Ty<'tcx>>,
    ) -> Result<bool, LayoutError<'tcx>> {
        Ok(!tcx.permits_uninit_init(param_env_and_ty)?)
    }

    match intrinsic_name {
        sym::assert_inhabited => Some(inhabited_predicate),
        sym::assert_zero_valid => Some(zero_valid_predicate),
        sym::assert_mem_uninitialized_valid => Some(mem_uninitialized_valid_predicate),
        _ => None,
    }
}

fn resolve_rust_intrinsic<'tcx>(
    tcx: TyCtxt<'tcx>,
    func_ty: Ty<'tcx>,
) -> Option<(Symbol, SubstsRef<'tcx>)> {
    if let ty::FnDef(def_id, substs) = *func_ty.kind() {
        if tcx.is_intrinsic(def_id) {
            return Some((tcx.item_name(def_id), substs));
        }
    }
    None
}
