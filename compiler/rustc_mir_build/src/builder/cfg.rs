//! Routines for manipulating the control-flow graph.

use itertools::Either;
use rustc_attr_ir::UnrollAttr;
use rustc_data_structures::thin_vec::ThinVec;
use rustc_index::IndexVec;
use rustc_middle::mir::*;
use rustc_middle::ty::TyCtxt;
use tracing::debug;

use crate::builder::{CFG, CFGBlock, UnterminatedBlockData};

impl<'tcx> CFG<'tcx> {
    fn block_data(&self, blk: BasicBlock) -> &CFGBlock<'tcx> {
        &self.basic_blocks[blk]
    }

    fn block_data_mut(&mut self, blk: BasicBlock) -> &mut CFGBlock<'tcx> {
        &mut self.basic_blocks[blk]
    }

    pub(crate) fn assert_terminator(&self, bb: BasicBlock) -> &Terminator<'tcx> {
        self.block_data(bb).as_ref().left().unwrap().terminator()
    }

    pub(crate) fn assert_terminator_mut(&mut self, bb: BasicBlock) -> &mut Terminator<'tcx> {
        self.block_data_mut(bb).as_mut().left().unwrap().terminator_mut()
    }

    pub(crate) fn statements(&self, blk: BasicBlock) -> &[Statement<'tcx>] {
        match &self.basic_blocks[blk] {
            CFGBlock::Left(bb) => &bb.statements,
            CFGBlock::Right(bb) => &bb.statements,
        }
    }

    pub(crate) fn statements_mut(&mut self, blk: BasicBlock) -> &mut Vec<Statement<'tcx>> {
        match &mut self.basic_blocks[blk] {
            CFGBlock::Left(bb) => &mut bb.statements,
            CFGBlock::Right(bb) => &mut bb.statements,
        }
    }

    pub(crate) fn into_basic_blocks(self) -> IndexVec<BasicBlock, BasicBlockData<'tcx>> {
        // CFGBlock has the same layout as BasicBlockData, so this call should be
        // almost no-op (we have a specialization for reusing the allocation in this case).
        self.basic_blocks.into_iter().map(|bb| bb.left().unwrap()).collect()
    }

    // llvm.org/PR32488 makes this function use an excess of stack space. Mark
    // it as #[inline(never)] to keep rustc's stack use in check.
    #[inline(never)]
    pub(crate) fn start_new_block(&mut self) -> BasicBlock {
        self.basic_blocks.push(Either::Right(UnterminatedBlockData::new(Vec::new(), false)))
    }

    pub(crate) fn start_new_cleanup_block(&mut self) -> BasicBlock {
        self.basic_blocks.push(Either::Right(UnterminatedBlockData::new(Vec::new(), true)))
    }

    pub(crate) fn push(&mut self, block: BasicBlock, statement: Statement<'tcx>) {
        debug!("push({:?}, {:?})", block, statement);
        self.statements_mut(block).push(statement);
    }

    pub(crate) fn push_assign(
        &mut self,
        block: BasicBlock,
        source_info: SourceInfo,
        place: Place<'tcx>,
        rvalue: Rvalue<'tcx>,
    ) {
        self.push(
            block,
            Statement::new(source_info, StatementKind::Assign(Box::new((place, rvalue)))),
        );
    }

    pub(crate) fn push_assign_constant(
        &mut self,
        block: BasicBlock,
        source_info: SourceInfo,
        temp: Place<'tcx>,
        constant: ConstOperand<'tcx>,
    ) {
        self.push_assign(
            block,
            source_info,
            temp,
            Rvalue::Use(Operand::Constant(Box::new(constant)), WithRetag::Yes),
        );
    }

    pub(crate) fn push_assign_unit(
        &mut self,
        block: BasicBlock,
        source_info: SourceInfo,
        place: Place<'tcx>,
        tcx: TyCtxt<'tcx>,
    ) {
        self.push_assign(
            block,
            source_info,
            place,
            Rvalue::Use(
                Operand::zero_sized_constant(tcx.types.unit, source_info.span),
                WithRetag::Yes,
            ),
        );
    }

    pub(crate) fn push_fake_read(
        &mut self,
        block: BasicBlock,
        source_info: SourceInfo,
        cause: FakeReadCause,
        place: Place<'tcx>,
    ) {
        let kind = StatementKind::FakeRead(Box::new((cause, place)));
        let stmt = Statement::new(source_info, kind);
        self.push(block, stmt);
    }

    pub(crate) fn push_place_mention(
        &mut self,
        block: BasicBlock,
        source_info: SourceInfo,
        place: Place<'tcx>,
    ) {
        let kind = StatementKind::PlaceMention(Box::new(place));
        let stmt = Statement::new(source_info, kind);
        self.push(block, stmt);
    }

    pub(crate) fn terminate(
        &mut self,
        block: BasicBlock,
        source_info: SourceInfo,
        kind: TerminatorKind<'tcx>,
    ) -> &mut Terminator<'tcx> {
        debug!("terminating block {:?} <- {:?}", block, kind);
        debug_assert!(
            self.block_data(block).is_right(),
            "terminate: block {:?}={:?} already has a terminator set",
            block,
            self.block_data(block)
        );
        let UnterminatedBlockData { statements, is_cleanup } =
            self.block_data_mut(block).as_mut().right().unwrap();

        *self.block_data_mut(block) = Either::Left(BasicBlockData::new_stmts(
            std::mem::take(statements),
            Terminator { source_info, kind },
            *is_cleanup,
        ));
        self.block_data_mut(block).as_mut().left().unwrap().terminator_mut()
    }

    /// In the `origin` block, push a `goto -> target` terminator.
    pub(crate) fn goto(
        &mut self,
        origin: BasicBlock,
        source_info: SourceInfo,
        target: BasicBlock,
    ) -> &mut Terminator<'tcx> {
        self.goto_with_hints(origin, source_info, target, ThinVec::new())
    }

    #[inline(always)]
    pub(crate) fn goto_with_hints(
        &mut self,
        origin: BasicBlock,
        source_info: SourceInfo,
        target: BasicBlock,
        loop_hint_attrs: ThinVec<UnrollAttr>,
    ) -> &mut Terminator<'tcx> {
        self.terminate(origin, source_info, TerminatorKind::Goto { target, loop_hint_attrs })
    }
}
