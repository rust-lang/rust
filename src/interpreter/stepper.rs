use super::{
    FnEvalContext,
    CachedMir,
    TerminatorTarget,
    ConstantId,
};
use error::EvalResult;
use rustc::mir::repr as mir;
use rustc::ty::subst::{self, Subst};
use rustc::hir::def_id::DefId;
use rustc::mir::visit::{Visitor, LvalueContext};
use syntax::codemap::Span;
use memory::Pointer;
use std::rc::Rc;

pub enum Event {
    Constant,
    Assignment,
    Terminator,
    Done,
}

pub struct Stepper<'fncx, 'a: 'fncx, 'b: 'a + 'mir, 'mir: 'fncx, 'tcx: 'b>{
    fncx: &'fncx mut FnEvalContext<'a, 'b, 'mir, 'tcx>,
    mir: CachedMir<'mir, 'tcx>,
    process: fn (&mut Stepper<'fncx, 'a, 'b, 'mir, 'tcx>) -> EvalResult<()>,
    // a stack of constants
    constants: Vec<Vec<(ConstantId<'tcx>, Span, Pointer, CachedMir<'mir, 'tcx>)>>,
}

impl<'fncx, 'a, 'b: 'a + 'mir, 'mir, 'tcx: 'b> Stepper<'fncx, 'a, 'b, 'mir, 'tcx> {
    pub(super) fn new(fncx: &'fncx mut FnEvalContext<'a, 'b, 'mir, 'tcx>) -> Self {
        Stepper {
            mir: fncx.mir(),
            fncx: fncx,
            process: Self::dummy,
            constants: vec![Vec::new()],
        }
    }

    fn dummy(&mut self) -> EvalResult<()> { Ok(()) }

    fn statement(&mut self) -> EvalResult<()> {
        let block_data = self.mir.basic_block_data(self.fncx.frame().next_block);
        let stmt = &block_data.statements[self.fncx.frame().stmt];
        let mir::StatementKind::Assign(ref lvalue, ref rvalue) = stmt.kind;
        let result = self.fncx.eval_assignment(lvalue, rvalue);
        self.fncx.maybe_report(stmt.span, result)?;
        self.fncx.frame_mut().stmt += 1;
        Ok(())
    }

    fn terminator(&mut self) -> EvalResult<()> {
        // after a terminator we go to a new block
        self.fncx.frame_mut().stmt = 0;
        let term = {
            let block_data = self.mir.basic_block_data(self.fncx.frame().next_block);
            let terminator = block_data.terminator();
            let result = self.fncx.eval_terminator(terminator);
            self.fncx.maybe_report(terminator.span, result)?
        };
        match term {
            TerminatorTarget::Block => {},
            TerminatorTarget::Return => {
                self.fncx.pop_stack_frame();
                assert!(self.constants.last().unwrap().is_empty());
                self.constants.pop();
                if !self.fncx.stack.is_empty() {
                    self.mir = self.fncx.mir();
                }
            },
            TerminatorTarget::Call => {
                self.mir = self.fncx.mir();
                self.constants.push(Vec::new());
            },
        }
        Ok(())
    }

    fn constant(&mut self) -> EvalResult<()> {
        match self.constants.last_mut().unwrap().pop() {
            Some((ConstantId::Promoted { index }, span, return_ptr, mir)) => {
                trace!("adding promoted constant {}, {:?}", index, span);
                let substs = self.fncx.substs();
                // FIXME: somehow encode that this is a promoted constant's frame
                let def_id = self.fncx.frame().def_id;
                self.fncx.push_stack_frame(def_id, span, mir, substs, Some(return_ptr));
                self.constants.push(Vec::new());
                self.mir = self.fncx.mir();
            },
            Some((ConstantId::Static { def_id, substs }, span, return_ptr, mir)) => {
                trace!("adding static {:?}, {:?}", def_id, span);
                self.fncx.gecx.statics.insert(def_id, return_ptr);
                self.fncx.push_stack_frame(def_id, span, mir, substs, Some(return_ptr));
                self.constants.push(Vec::new());
                self.mir = self.fncx.mir();
            },
            None => unreachable!(),
        }
        Ok(())
    }

    pub fn step(&mut self) -> EvalResult<Event> {
        (self.process)(self)?;

        if self.fncx.stack.is_empty() {
            // fuse the iterator
            self.process = Self::dummy;
            return Ok(Event::Done);
        }

        if !self.constants.last().unwrap().is_empty() {
            self.process = Self::constant;
            return Ok(Event::Constant);
        }

        let block = self.fncx.frame().next_block;
        let stmt = self.fncx.frame().stmt;
        let basic_block = self.mir.basic_block_data(block);

        if let Some(ref stmt) = basic_block.statements.get(stmt) {
            assert!(self.constants.last().unwrap().is_empty());
            ConstantExtractor {
                constants: &mut self.constants.last_mut().unwrap(),
                span: stmt.span,
                fncx: self.fncx,
                mir: &self.mir,
            }.visit_statement(block, stmt);
            if self.constants.last().unwrap().is_empty() {
                self.process = Self::statement;
                return Ok(Event::Assignment);
            } else {
                self.process = Self::constant;
                return Ok(Event::Constant);
            }
        }

        let terminator = basic_block.terminator();
        ConstantExtractor {
            constants: &mut self.constants.last_mut().unwrap(),
            span: terminator.span,
            fncx: self.fncx,
            mir: &self.mir,
        }.visit_terminator(block, terminator);
        if self.constants.last().unwrap().is_empty() {
            self.process = Self::terminator;
            Ok(Event::Terminator)
        } else {
            self.process = Self::constant;
            return Ok(Event::Constant);
        }
    }

    /// returns the statement that will be processed next
    pub fn stmt(&self) -> &mir::Statement {
        &self.fncx.basic_block().statements[self.fncx.frame().stmt]
    }

    /// returns the terminator of the current block
    pub fn term(&self) -> &mir::Terminator {
        self.fncx.basic_block().terminator()
    }

    pub fn block(&self) -> mir::BasicBlock {
        self.fncx.frame().next_block
    }
}

struct ConstantExtractor<'a: 'c, 'b: 'a + 'mir + 'c, 'c, 'mir: 'c, 'tcx: 'a + 'b + 'c> {
    constants: &'c mut Vec<(ConstantId<'tcx>, Span, Pointer, CachedMir<'mir, 'tcx>)>,
    span: Span,
    mir: &'c mir::Mir<'tcx>,
    fncx: &'c mut FnEvalContext<'a, 'b, 'mir, 'tcx>,
}

impl<'a, 'b, 'c, 'mir, 'tcx> ConstantExtractor<'a, 'b, 'c, 'mir, 'tcx> {
    fn constant(&mut self, def_id: DefId, substs: &'tcx subst::Substs<'tcx>, span: Span) {
        if self.fncx.gecx.statics.contains_key(&def_id) {
            return;
        }
        let cid = ConstantId::Static {
            def_id: def_id,
            substs: substs,
        };
        let mir = self.fncx.load_mir(def_id);
        let ptr = self.fncx.alloc_ret_ptr(mir.return_ty).expect("there's no such thing as an unreachable static");
        self.constants.push((cid, span, ptr, mir));
    }
}

impl<'a, 'b, 'c, 'mir, 'tcx> Visitor<'tcx> for ConstantExtractor<'a, 'b, 'c, 'mir, 'tcx> {
    fn visit_constant(&mut self, constant: &mir::Constant<'tcx>) {
        self.super_constant(constant);
        match constant.literal {
            // already computed by rustc
            mir::Literal::Value { .. } => {}
            mir::Literal::Item { def_id, substs } => {
                let item_ty = self.fncx.tcx.lookup_item_type(def_id).subst(self.fncx.tcx, substs);
                if item_ty.ty.is_fn() {
                    // unimplemented
                } else {
                    self.constant(def_id, substs, constant.span);
                }
            },
            mir::Literal::Promoted { index } => {
                if self.fncx.frame().promoted.contains_key(&index) {
                    return;
                }
                let mir = self.mir.promoted[index].clone();
                let return_ty = mir.return_ty;
                let return_ptr = self.fncx.alloc_ret_ptr(return_ty).expect("there's no such thing as an unreachable static");
                self.fncx.frame_mut().promoted.insert(index, return_ptr);
                let mir = CachedMir::Owned(Rc::new(mir));
                self.constants.push((ConstantId::Promoted { index: index }, constant.span, return_ptr, mir));
            }
        }
    }

    fn visit_lvalue(&mut self, lvalue: &mir::Lvalue<'tcx>, context: LvalueContext) {
        self.super_lvalue(lvalue, context);
        if let mir::Lvalue::Static(def_id) = *lvalue {
            let substs = self.fncx.tcx.mk_substs(subst::Substs::empty());
            let span = self.span;
            self.constant(def_id, substs, span);
        }
    }
}
