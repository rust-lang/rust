//! Routines for manipulating the control-flow graph.

use crate::build::CFG;
use rustc_middle::mir::*;
use rustc_middle::ty::{self, TyCtxt};

impl<'tcx> CFG<'tcx> {
    crate fn block_data(&self, blk: BasicBlock) -> &BasicBlockData<'tcx> {
        &self.basic_blocks[blk]
    }

    crate fn block_data_mut(&mut self, blk: BasicBlock) -> &mut BasicBlockData<'tcx> {
        &mut self.basic_blocks[blk]
    }

    // llvm.org/PR32488 makes this function use an excess of stack space. Mark
    // it as #[inline(never)] to keep rustc's stack use in check.
    #[inline(never)]
    crate fn start_new_block(&mut self) -> BasicBlock {
        self.basic_blocks.push(BasicBlockData::new(None))
    }

    crate fn start_new_cleanup_block(&mut self) -> BasicBlock {
        let bb = self.start_new_block();
        self.block_data_mut(bb).is_cleanup = true;
        bb
    }

    crate fn push(&mut self, block: BasicBlock, statement: Statement<'tcx>) {
        debug!("push({:?}, {:?})", block, statement);
        self.block_data_mut(block).statements.push(statement);
    }

    crate fn push_assign(
        &mut self,
        block: BasicBlock,
        source_info: SourceInfo,
        place: Place<'tcx>,
        rvalue: Rvalue<'tcx>,
    ) {
        self.push(
            block,
            Statement { source_info, kind: StatementKind::Assign(box (place, rvalue)) },
        );
    }

    crate fn push_assign_constant(
        &mut self,
        block: BasicBlock,
        source_info: SourceInfo,
        temp: Place<'tcx>,
        constant: Constant<'tcx>,
    ) {
        self.push_assign(block, source_info, temp, Rvalue::Use(Operand::Constant(box constant)));
    }

    crate fn push_assign_unit(
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
            Rvalue::Use(Operand::Constant(box Constant {
                span: source_info.span,
                user_ty: None,
                literal: ty::Const::zero_sized(tcx, tcx.types.unit),
            })),
        );
    }

    crate fn push_fake_read(
        &mut self,
        block: BasicBlock,
        source_info: SourceInfo,
        cause: FakeReadCause,
        place: Place<'tcx>,
    ) {
        let kind = StatementKind::FakeRead(cause, box place);
        let stmt = Statement { source_info, kind };
        self.push(block, stmt);
    }

    crate fn terminate(
        &mut self,
        block: BasicBlock,
        source_info: SourceInfo,
        kind: TerminatorKind<'tcx>,
    ) {
        debug!("terminating block {:?} <- {:?}", block, kind);
        debug_assert!(
            self.block_data(block).terminator.is_none(),
            "terminate: block {:?}={:?} already has a terminator set",
            block,
            self.block_data(block)
        );
        self.block_data_mut(block).terminator = Some(Terminator { source_info, kind });
    }

    /// In the `origin` block, push a `goto -> target` terminator.
    crate fn goto(&mut self, origin: BasicBlock, source_info: SourceInfo, target: BasicBlock) {
        self.terminate(origin, source_info, TerminatorKind::Goto { target })
    }
}
