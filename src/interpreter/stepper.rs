use super::{
    FnEvalContext,
    CachedMir,
    TerminatorTarget,
    ConstantId,
};
use error::EvalResult;
use rustc::mir::repr as mir;
use rustc::ty::{self, subst};
use rustc::mir::visit::Visitor;
use syntax::codemap::Span;
use memory::Pointer;
use std::rc::Rc;

pub enum Event {
    Assignment,
    Terminator,
    Done,
}

pub struct Stepper<'fncx, 'a: 'fncx, 'b: 'a + 'mir, 'mir: 'fncx, 'tcx: 'b>{
    fncx: &'fncx mut FnEvalContext<'a, 'b, 'mir, 'tcx>,
    block: mir::BasicBlock,
    // a stack of statement positions
    stmt: Vec<usize>,
    mir: CachedMir<'mir, 'tcx>,
    process: fn (&mut Stepper<'fncx, 'a, 'b, 'mir, 'tcx>) -> EvalResult<()>,
    // a stack of constants
    constants: Vec<Vec<(ConstantId, Span)>>,
}

impl<'fncx, 'a, 'b: 'a + 'mir, 'mir, 'tcx: 'b> Stepper<'fncx, 'a, 'b, 'mir, 'tcx> {
    pub(super) fn new(fncx: &'fncx mut FnEvalContext<'a, 'b, 'mir, 'tcx>) -> Self {
        let mut stepper = Stepper {
            block: fncx.frame().next_block,
            mir: fncx.mir(),
            fncx: fncx,
            stmt: vec![0],
            process: Self::dummy,
            constants: Vec::new(),
        };
        stepper.extract_constants();
        stepper
    }

    fn dummy(&mut self) -> EvalResult<()> { Ok(()) }

    fn statement(&mut self) -> EvalResult<()> {
        let block_data = self.mir.basic_block_data(self.block);
        let stmt = &block_data.statements[*self.stmt.last().unwrap()];
        let mir::StatementKind::Assign(ref lvalue, ref rvalue) = stmt.kind;
        let result = self.fncx.eval_assignment(lvalue, rvalue);
        self.fncx.maybe_report(stmt.span, result)?;
        *self.stmt.last_mut().unwrap() += 1;
        Ok(())
    }

    fn terminator(&mut self) -> EvalResult<()> {
        *self.stmt.last_mut().unwrap() = 0;
        let term = {
            let block_data = self.mir.basic_block_data(self.block);
            let terminator = block_data.terminator();
            let result = self.fncx.eval_terminator(terminator);
            self.fncx.maybe_report(terminator.span, result)?
        };
        match term {
            TerminatorTarget::Block(block) => {
                self.block = block;
            },
            TerminatorTarget::Return => {
                self.fncx.pop_stack_frame();
                self.fncx.name_stack.pop();
                self.stmt.pop();
                assert!(self.constants.last().unwrap().is_empty());
                self.constants.pop();
                if !self.fncx.stack.is_empty() {
                    self.block = self.fncx.frame().next_block;
                    self.mir = self.fncx.mir();
                }
            },
            TerminatorTarget::Call => {
                self.block = self.fncx.frame().next_block;
                self.mir = self.fncx.mir();
                self.stmt.push(0);
                self.extract_constants();
            },
        }
        Ok(())
    }

    fn alloc(&mut self, ty: ty::FnOutput<'tcx>) -> Pointer {
        match ty {
            ty::FnConverging(ty) => {
                let size = self.fncx.type_size(ty);
                self.fncx.memory.allocate(size)
            }
            ty::FnDiverging => panic!("there's no such thing as an unreachable static"),
        }
    }

    pub fn step(&mut self) -> EvalResult<Event> {
        (self.process)(self)?;

        if self.fncx.stack.is_empty() {
            // fuse the iterator
            self.process = Self::dummy;
            return Ok(Event::Done);
        }

        match self.constants.last_mut().unwrap().pop() {
            Some((ConstantId::Promoted { index }, span)) => {
                trace!("adding promoted constant {}", index);
                let mir = self.mir.promoted[index].clone();
                let return_ptr = self.alloc(mir.return_ty);
                self.fncx.frame_mut().promoted.insert(index, return_ptr);
                let substs = self.fncx.substs();
                // FIXME: somehow encode that this is a promoted constant's frame
                let def_id = self.fncx.name_stack.last().unwrap().0;
                self.fncx.name_stack.push((def_id, substs, span));
                self.fncx.push_stack_frame(CachedMir::Owned(Rc::new(mir)), substs, Some(return_ptr));
                self.stmt.push(0);
                self.constants.push(Vec::new());
                self.block = self.fncx.frame().next_block;
                self.mir = self.fncx.mir();
            },
            Some((ConstantId::Static { def_id }, span)) => {
                trace!("adding static {:?}", def_id);
                let mir = self.fncx.load_mir(def_id);
                let return_ptr = self.alloc(mir.return_ty);
                self.fncx.gecx.statics.insert(def_id, return_ptr);
                let substs = self.fncx.tcx.mk_substs(subst::Substs::empty());
                self.fncx.name_stack.push((def_id, substs, span));
                self.fncx.push_stack_frame(mir, substs, Some(return_ptr));
                self.stmt.push(0);
                self.constants.push(Vec::new());
                self.block = self.fncx.frame().next_block;
                self.mir = self.fncx.mir();
            },
            None => {},
        }

        let basic_block = self.mir.basic_block_data(self.block);

        if basic_block.statements.len() > *self.stmt.last().unwrap() {
            self.process = Self::statement;
            return Ok(Event::Assignment);
        }

        self.process = Self::terminator;
        Ok(Event::Terminator)
    }

    /// returns the basic block index of the currently processed block
    pub fn block(&self) -> mir::BasicBlock {
        self.block
    }

    /// returns the statement that will be processed next
    pub fn stmt(&self) -> &mir::Statement {
        let block_data = self.mir.basic_block_data(self.block);
        &block_data.statements[*self.stmt.last().unwrap()]
    }

    /// returns the terminator of the current block
    pub fn term(&self) -> &mir::Terminator {
        let block_data = self.mir.basic_block_data(self.block);
        block_data.terminator()
    }

    fn extract_constants(&mut self) {
        let mut extractor = ConstantExtractor {
            constants: Vec::new(),
        };
        extractor.visit_mir(&self.mir);
        self.constants.push(extractor.constants);
    }
}

struct ConstantExtractor {
    constants: Vec<(ConstantId, Span)>,
}

impl<'tcx> Visitor<'tcx> for ConstantExtractor {
    fn visit_constant(&mut self, constant: &mir::Constant<'tcx>) {
        self.super_constant(constant);
        match constant.literal {
            // already computed by rustc
            mir::Literal::Value { .. } => {}
            mir::Literal::Item { .. } => {}, // FIXME: unimplemented
            mir::Literal::Promoted { index } => {
                self.constants.push((ConstantId::Promoted { index: index }, constant.span));
            }
        }
    }

    fn visit_statement(&mut self, block: mir::BasicBlock, stmt: &mir::Statement<'tcx>) {
        self.super_statement(block, stmt);
        if let mir::StatementKind::Assign(mir::Lvalue::Static(def_id), _) = stmt.kind {
            self.constants.push((ConstantId::Static { def_id: def_id }, stmt.span));
        }
    }
}
