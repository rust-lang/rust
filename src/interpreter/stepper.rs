//! This module contains the `EvalContext` methods for executing a single step of the interpreter.
//!
//! The main entry point is the `step` method.

use super::{
    CachedMir,
    ConstantId,
    EvalContext,
    ConstantKind,
};
use error::EvalResult;
use rustc::mir::repr as mir;
use rustc::ty::{subst, self};
use rustc::hir::def_id::DefId;
use rustc::mir::visit::{Visitor, LvalueContext};
use syntax::codemap::Span;
use std::rc::Rc;

impl<'a, 'tcx> EvalContext<'a, 'tcx> {
    /// Returns true as long as there are more things to do.
    pub fn step(&mut self) -> EvalResult<'tcx, bool> {
        if self.stack.is_empty() {
            return Ok(false);
        }

        let block = self.frame().block;
        let stmt = self.frame().stmt;
        let mir = self.mir();
        let basic_block = &mir.basic_blocks()[block];

        if let Some(ref stmt) = basic_block.statements.get(stmt) {
            let current_stack = self.stack.len();
            ConstantExtractor {
                span: stmt.source_info.span,
                substs: self.substs(),
                def_id: self.frame().def_id,
                ecx: self,
                mir: &mir,
            }.visit_statement(block, stmt);
            if current_stack == self.stack.len() {
                self.statement(stmt)?;
            } else {
                // ConstantExtractor added some new frames for statics/constants/promoteds
                // self.step() can't be "done", so it can't return false
                assert!(self.step()?);
            }
            return Ok(true);
        }

        let terminator = basic_block.terminator();
        let current_stack = self.stack.len();
        ConstantExtractor {
            span: terminator.source_info.span,
            substs: self.substs(),
            def_id: self.frame().def_id,
            ecx: self,
            mir: &mir,
        }.visit_terminator(block, terminator);
        if current_stack == self.stack.len() {
            self.terminator(terminator)?;
        } else {
            // ConstantExtractor added some new frames for statics/constants/promoteds
            // self.step() can't be "done", so it can't return false
            assert!(self.step()?);
        }
        Ok(true)
    }

    fn statement(&mut self, stmt: &mir::Statement<'tcx>) -> EvalResult<'tcx, ()> {
        trace!("{:?}", stmt);
        let mir::StatementKind::Assign(ref lvalue, ref rvalue) = stmt.kind;
        self.eval_assignment(lvalue, rvalue)?;
        self.frame_mut().stmt += 1;
        Ok(())
    }

    fn terminator(&mut self, terminator: &mir::Terminator<'tcx>) -> EvalResult<'tcx, ()> {
        // after a terminator we go to a new block
        self.frame_mut().stmt = 0;
        trace!("{:?}", terminator.kind);
        self.eval_terminator(terminator)?;
        if !self.stack.is_empty() {
            trace!("// {:?}", self.frame().block);
        }
        Ok(())
    }
}

// WARNING: make sure that any methods implemented on this type don't ever access ecx.stack
// this includes any method that might access the stack
// basically don't call anything other than `load_mir`, `alloc_ret_ptr`, `push_stack_frame`
// The reason for this is, that `push_stack_frame` modifies the stack out of obvious reasons
struct ConstantExtractor<'a, 'b: 'a, 'tcx: 'b> {
    span: Span,
    ecx: &'a mut EvalContext<'b, 'tcx>,
    mir: &'a mir::Mir<'tcx>,
    def_id: DefId,
    substs: &'tcx subst::Substs<'tcx>,
}

impl<'a, 'b, 'tcx> ConstantExtractor<'a, 'b, 'tcx> {
    fn global_item(&mut self, def_id: DefId, substs: &'tcx subst::Substs<'tcx>, span: Span) {
        let cid = ConstantId {
            def_id: def_id,
            substs: substs,
            kind: ConstantKind::Global,
        };
        if self.ecx.statics.contains_key(&cid) {
            return;
        }
        let mir = self.ecx.load_mir(def_id);
        let ptr = self.ecx.alloc_ret_ptr(mir.return_ty, substs).expect("there's no such thing as an unreachable static");
        self.ecx.statics.insert(cid.clone(), ptr);
        self.ecx.push_stack_frame(def_id, span, mir, substs, Some(ptr));
    }
}

impl<'a, 'b, 'tcx> Visitor<'tcx> for ConstantExtractor<'a, 'b, 'tcx> {
    fn visit_constant(&mut self, constant: &mir::Constant<'tcx>) {
        self.super_constant(constant);
        match constant.literal {
            // already computed by rustc
            mir::Literal::Value { .. } => {}
            mir::Literal::Item { def_id, substs } => {
                if let ty::TyFnDef(..) = constant.ty.sty {
                    // No need to do anything here,
                    // because the type is the actual function, not the signature of the function.
                    // Thus we can simply create a zero sized allocation in `evaluate_operand`
                } else {
                    self.global_item(def_id, substs, constant.span);
                }
            },
            mir::Literal::Promoted { index } => {
                let cid = ConstantId {
                    def_id: self.def_id,
                    substs: self.substs,
                    kind: ConstantKind::Promoted(index),
                };
                if self.ecx.statics.contains_key(&cid) {
                    return;
                }
                let mir = self.mir.promoted[index].clone();
                let return_ty = mir.return_ty;
                let return_ptr = self.ecx.alloc_ret_ptr(return_ty, cid.substs).expect("there's no such thing as an unreachable static");
                let mir = CachedMir::Owned(Rc::new(mir));
                self.ecx.statics.insert(cid.clone(), return_ptr);
                self.ecx.push_stack_frame(self.def_id, constant.span, mir, self.substs, Some(return_ptr));
            }
        }
    }

    fn visit_lvalue(&mut self, lvalue: &mir::Lvalue<'tcx>, context: LvalueContext) {
        self.super_lvalue(lvalue, context);
        if let mir::Lvalue::Static(def_id) = *lvalue {
            let substs = self.ecx.tcx.mk_substs(subst::Substs::empty());
            let span = self.span;
            self.global_item(def_id, substs, span);
        }
    }
}
