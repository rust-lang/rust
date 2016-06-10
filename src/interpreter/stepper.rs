use super::{
    CachedMir,
    ConstantId,
    GlobalEvalContext,
    ConstantKind,
};
use error::EvalResult;
use rustc::mir::repr as mir;
use rustc::ty::{subst, self};
use rustc::hir::def_id::DefId;
use rustc::mir::visit::{Visitor, LvalueContext};
use syntax::codemap::Span;
use std::rc::Rc;

struct Stepper<'fncx, 'a: 'fncx, 'tcx: 'a>{
    gecx: &'fncx mut GlobalEvalContext<'a, 'tcx>,
}

pub fn step<'fncx, 'a: 'fncx, 'tcx: 'a>(gecx: &'fncx mut GlobalEvalContext<'a, 'tcx>) -> EvalResult<bool> {
    Stepper::new(gecx).step()
}

impl<'fncx, 'a, 'tcx> Stepper<'fncx, 'a, 'tcx> {
    fn new(gecx: &'fncx mut GlobalEvalContext<'a, 'tcx>) -> Self {
        Stepper {
            gecx: gecx,
        }
    }

    fn statement(&mut self, stmt: &mir::Statement<'tcx>) -> EvalResult<()> {
        trace!("{:?}", stmt);
        let mir::StatementKind::Assign(ref lvalue, ref rvalue) = stmt.kind;
        self.gecx.eval_assignment(lvalue, rvalue)?;
        self.gecx.frame_mut().stmt += 1;
        Ok(())
    }

    fn terminator(&mut self, terminator: &mir::Terminator<'tcx>) -> EvalResult<()> {
        // after a terminator we go to a new block
        self.gecx.frame_mut().stmt = 0;
        trace!("{:?}", terminator.kind);
        self.gecx.eval_terminator(terminator)?;
        if !self.gecx.stack.is_empty() {
            trace!("// {:?}", self.gecx.frame().next_block);
        }
        Ok(())
    }

    // returns true as long as there are more things to do
    fn step(&mut self) -> EvalResult<bool> {
        if self.gecx.stack.is_empty() {
            return Ok(false);
        }

        let block = self.gecx.frame().next_block;
        let stmt = self.gecx.frame().stmt;
        let mir = self.gecx.mir();
        let basic_block = mir.basic_block_data(block);

        if let Some(ref stmt) = basic_block.statements.get(stmt) {
            let current_stack = self.gecx.stack.len();
            ConstantExtractor {
                span: stmt.span,
                substs: self.gecx.substs(),
                def_id: self.gecx.frame().def_id,
                gecx: self.gecx,
                mir: &mir,
            }.visit_statement(block, stmt);
            if current_stack == self.gecx.stack.len() {
                self.statement(stmt)?;
            } else {
                // ConstantExtractor added some new frames for statics/constants/promoteds
                // self.step() can't be "done", so it can't return false
                assert!(self.step()?);
            }
            return Ok(true);
        }

        let terminator = basic_block.terminator();
        let current_stack = self.gecx.stack.len();
        ConstantExtractor {
            span: terminator.span,
            substs: self.gecx.substs(),
            def_id: self.gecx.frame().def_id,
            gecx: self.gecx,
            mir: &mir,
        }.visit_terminator(block, terminator);
        if current_stack == self.gecx.stack.len() {
            self.terminator(terminator)?;
        } else {
            // ConstantExtractor added some new frames for statics/constants/promoteds
            // self.step() can't be "done", so it can't return false
            assert!(self.step()?);
        }
        Ok(true)
    }
}

// WARNING: make sure that any methods implemented on this type don't ever access gecx.stack
// this includes any method that might access the stack
// basically don't call anything other than `load_mir`, `alloc_ret_ptr`, `push_stack_frame`
// The reason for this is, that `push_stack_frame` modifies the stack out of obvious reasons
struct ConstantExtractor<'a, 'b: 'a, 'tcx: 'b> {
    span: Span,
    gecx: &'a mut GlobalEvalContext<'b, 'tcx>,
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
        if self.gecx.statics.contains_key(&cid) {
            return;
        }
        let mir = self.gecx.load_mir(def_id);
        let ptr = self.gecx.alloc_ret_ptr(mir.return_ty, substs).expect("there's no such thing as an unreachable static");
        self.gecx.statics.insert(cid.clone(), ptr);
        self.gecx.push_stack_frame(def_id, span, mir, substs, Some(ptr));
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
                    // No need to do anything here, even if function pointers are implemented,
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
                if self.gecx.statics.contains_key(&cid) {
                    return;
                }
                let mir = self.mir.promoted[index].clone();
                let return_ty = mir.return_ty;
                let return_ptr = self.gecx.alloc_ret_ptr(return_ty, cid.substs).expect("there's no such thing as an unreachable static");
                let mir = CachedMir::Owned(Rc::new(mir));
                self.gecx.statics.insert(cid.clone(), return_ptr);
                self.gecx.push_stack_frame(self.def_id, constant.span, mir, self.substs, Some(return_ptr));
            }
        }
    }

    fn visit_lvalue(&mut self, lvalue: &mir::Lvalue<'tcx>, context: LvalueContext) {
        self.super_lvalue(lvalue, context);
        if let mir::Lvalue::Static(def_id) = *lvalue {
            let substs = self.gecx.tcx.mk_substs(subst::Substs::empty());
            let span = self.span;
            self.global_item(def_id, substs, span);
        }
    }
}
