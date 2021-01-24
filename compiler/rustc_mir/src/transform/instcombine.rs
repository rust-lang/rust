//! Performs various peephole optimizations.

use crate::transform::MirPass;
use rustc_hir::Mutability;
use rustc_middle::mir::{
    BasicBlock, LocalDecls, PlaceElem, SourceInfo, Statement, StatementKind, Terminator,
    TerminatorKind,
};
use rustc_middle::mir::{BinOp, Body, Constant, Local, Operand, Place, ProjectionElem, Rvalue};
use rustc_middle::ty::{self, TyCtxt};
use rustc_span::{sym, Symbol};
use rustc_target::spec::abi::Abi;

pub struct InstCombine;

impl<'tcx> MirPass<'tcx> for InstCombine {
    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        let (basic_blocks, local_decls) = body.basic_blocks_and_local_decls_mut();
        let ctx = InstCombineContext { tcx, local_decls };
        for block in basic_blocks.iter_mut() {
            for statement in block.statements.iter_mut() {
                match statement.kind {
                    StatementKind::Assign(box (_place, ref mut rvalue)) => {
                        ctx.combine_bool_cmp(&statement.source_info, rvalue);
                        ctx.combine_ref_deref(&statement.source_info, rvalue);
                        ctx.combine_len(&statement.source_info, rvalue);
                    }
                    _ => {}
                }
            }

            if let Some(terminator) = &mut block.terminator {
                ctx.combine_copy_nonoverlapping(terminator, local_decls, &mut block.statements);
            }
        }
    }
}

struct InstCombineContext<'tcx, 'a> {
    tcx: TyCtxt<'tcx>,
    local_decls: &'a LocalDecls<'tcx>,
}

impl<'tcx, 'a> InstCombineContext<'tcx, 'a> {
    fn should_combine(&self, source_info: &SourceInfo, rvalue: &Rvalue<'tcx>) -> bool {
        self.tcx.consider_optimizing(|| {
            format!("InstCombine - Rvalue: {:?} SourceInfo: {:?}", rvalue, source_info)
        })
    }

    /// Transform boolean comparisons into logical operations.
    fn combine_bool_cmp(&self, source_info: &SourceInfo, rvalue: &mut Rvalue<'tcx>) {
        match rvalue {
            Rvalue::BinaryOp(op @ (BinOp::Eq | BinOp::Ne), a, b) => {
                let new = match (op, self.try_eval_bool(a), self.try_eval_bool(b)) {
                    // Transform "Eq(a, true)" ==> "a"
                    (BinOp::Eq, _, Some(true)) => Some(a.clone()),

                    // Transform "Ne(a, false)" ==> "a"
                    (BinOp::Ne, _, Some(false)) => Some(a.clone()),

                    // Transform "Eq(true, b)" ==> "b"
                    (BinOp::Eq, Some(true), _) => Some(b.clone()),

                    // Transform "Ne(false, b)" ==> "b"
                    (BinOp::Ne, Some(false), _) => Some(b.clone()),

                    // FIXME: Consider combining remaining comparisons into logical operations:
                    // Transform "Eq(false, b)" ==> "Not(b)"
                    // Transform "Ne(true, b)" ==> "Not(b)"
                    // Transform "Eq(a, false)" ==> "Not(a)"
                    // Transform "Ne(a, true)" ==> "Not(a)"
                    _ => None,
                };

                if let Some(new) = new {
                    if self.should_combine(source_info, rvalue) {
                        *rvalue = Rvalue::Use(new);
                    }
                }
            }

            _ => {}
        }
    }

    fn try_eval_bool(&self, a: &Operand<'_>) -> Option<bool> {
        let a = a.constant()?;
        if a.literal.ty.is_bool() { a.literal.val.try_to_bool() } else { None }
    }

    /// Transform "&(*a)" ==> "a".
    fn combine_ref_deref(&self, source_info: &SourceInfo, rvalue: &mut Rvalue<'tcx>) {
        if let Rvalue::Ref(_, _, place) = rvalue {
            if let Some((base, ProjectionElem::Deref)) = place.as_ref().last_projection() {
                if let ty::Ref(_, _, Mutability::Not) =
                    base.ty(self.local_decls, self.tcx).ty.kind()
                {
                    // The dereferenced place must have type `&_`, so that we don't copy `&mut _`.
                } else {
                    return;
                }

                if !self.should_combine(source_info, rvalue) {
                    return;
                }

                *rvalue = Rvalue::Use(Operand::Copy(Place {
                    local: base.local,
                    projection: self.tcx.intern_place_elems(base.projection),
                }));
            }
        }
    }

    /// Transform "Len([_; N])" ==> "N".
    fn combine_len(&self, source_info: &SourceInfo, rvalue: &mut Rvalue<'tcx>) {
        if let Rvalue::Len(ref place) = *rvalue {
            let place_ty = place.ty(self.local_decls, self.tcx).ty;
            if let ty::Array(_, len) = place_ty.kind() {
                if !self.should_combine(source_info, rvalue) {
                    return;
                }

                let constant = Constant { span: source_info.span, literal: len, user_ty: None };
                *rvalue = Rvalue::Use(Operand::Constant(box constant));
            }
        }
    }

    fn func_as_intrinsic(
        &self,
        operand: &Operand<'tcx>,
        locals: &LocalDecls<'tcx>,
    ) -> Option<Symbol> {
        let func_ty = operand.ty(locals, self.tcx);

        if let ty::FnDef(def_id, _) = *func_ty.kind() {
            let fn_sig = func_ty.fn_sig(self.tcx);

            if fn_sig.abi() == Abi::RustIntrinsic {
                return Some(self.tcx.item_name(def_id));
            }
        }

        None
    }

    fn find_copy_nonoverlapping(
        &self,
        terminator: &Terminator<'tcx>,
        locals: &LocalDecls<'tcx>,
    ) -> Option<(Local, Local, BasicBlock)> {
        if let TerminatorKind::Call { func, args, destination: Some((_, next_bb)), .. } =
            &terminator.kind
        {
            let intrinsic = self.func_as_intrinsic(func, locals)?;

            if intrinsic == sym::copy_nonoverlapping && args.len() == 3 {
                let src = args[0].place()?.as_local()?;
                let dest = args[1].place()?.as_local()?;
                let constant = args[2].constant()?;

                if constant.literal.ty == self.tcx.types.usize {
                    let val = constant
                        .literal
                        .val
                        .try_to_value()?
                        .try_to_scalar()?
                        .to_machine_usize(&self.tcx)
                        .ok()?;

                    if val == 1 {
                        return Some((src, dest, *next_bb));
                    }
                }
            }
        }

        None
    }

    fn combine_copy_nonoverlapping(
        &self,
        terminator: &mut Terminator<'tcx>,
        locals: &LocalDecls<'tcx>,
        statements: &mut Vec<Statement<'tcx>>,
    ) {
        if let Some((src, dest, next_bb)) = self.find_copy_nonoverlapping(terminator, locals) {
            trace!("replacing call to copy_nonoverlapping({:?}, {:?}, 1) intrinsic", src, dest);
            let deref_projection = self.tcx._intern_place_elems(&[PlaceElem::Deref]);

            statements.push(Statement {
                source_info: terminator.source_info,
                kind: StatementKind::Assign(Box::new((
                    Place { local: dest, projection: deref_projection },
                    Rvalue::Use(Operand::Copy(Place { local: src, projection: deref_projection })),
                ))),
            });

            terminator.kind = TerminatorKind::Goto { target: next_bb };
        }
    }
}
