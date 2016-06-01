use super::{
    FnEvalContext,
    CachedMir,
    TerminatorTarget,
};
use error::EvalResult;
use rustc::mir::repr as mir;

pub enum Event<'a, 'tcx: 'a> {
    Assignment(&'a mir::Statement<'tcx>),
    Terminator(&'a mir::Terminator<'tcx>),
    Done,
}

pub struct Stepper<'fncx, 'a: 'fncx, 'b: 'a + 'mir, 'mir: 'fncx, 'tcx: 'b>{
    fncx: &'fncx mut FnEvalContext<'a, 'b, 'mir, 'tcx>,
    block: mir::BasicBlock,
    stmt: usize,
    mir: CachedMir<'mir, 'tcx>,
    process: fn (&mut Stepper<'fncx, 'a, 'b, 'mir, 'tcx>) -> EvalResult<()>,
}

impl<'fncx, 'a, 'b: 'a + 'mir, 'mir, 'tcx: 'b> Stepper<'fncx, 'a, 'b, 'mir, 'tcx> {
    pub(super) fn new(fncx: &'fncx mut FnEvalContext<'a, 'b, 'mir, 'tcx>) -> Self {
        Stepper {
            block: fncx.frame().next_block,
            mir: fncx.mir(),
            fncx: fncx,
            stmt: 0,
            process: Self::dummy,
        }
    }

    fn dummy(&mut self) -> EvalResult<()> { Ok(()) }

    fn statement(&mut self) -> EvalResult<()> {
        let block_data = self.mir.basic_block_data(self.block);
        let stmt = &block_data.statements[self.stmt];
        let mir::StatementKind::Assign(ref lvalue, ref rvalue) = stmt.kind;
        let result = self.fncx.eval_assignment(lvalue, rvalue);
        self.fncx.maybe_report(stmt.span, result)?;
        self.stmt += 1;
        Ok(())
    }

    fn terminator(&mut self) -> EvalResult<()> {
        self.stmt = 0;
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
                if !self.fncx.stack.is_empty() {
                    self.block = self.fncx.frame().next_block;
                    self.mir = self.fncx.mir();
                }
            },
            TerminatorTarget::Call => {
                self.block = self.fncx.frame().next_block;
                self.mir = self.fncx.mir();
            },
        }
        Ok(())
    }

    pub fn step<'step>(&'step mut self) -> EvalResult<Event<'step, 'tcx>> {
        (self.process)(self)?;

        if self.fncx.stack.is_empty() {
            // fuse the iterator
            self.process = Self::dummy;
            return Ok(Event::Done);
        }

        let basic_block = self.mir.basic_block_data(self.block);

        if let Some(stmt) = basic_block.statements.get(self.stmt) {
            self.process = Self::statement;
            return Ok(Event::Assignment(&stmt));
        }

        self.process = Self::terminator;
        Ok(Event::Terminator(basic_block.terminator()))
    }
    
    pub fn block(&self) -> mir::BasicBlock {
        self.block
    }
}
