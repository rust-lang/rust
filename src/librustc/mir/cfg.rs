use mir::repr::*;

use std::ops::{Index, IndexMut};
use syntax::codemap::Span;

#[derive(Clone, RustcEncodable, RustcDecodable)]
pub struct CFG<'tcx> {
    pub basic_blocks: Vec<BasicBlockData<'tcx>>,
}

pub struct PredecessorIter(::std::vec::IntoIter<BasicBlock>);
impl Iterator for PredecessorIter {
    type Item = BasicBlock;
    fn next(&mut self) -> Option<BasicBlock> {
        self.0.next()
    }
}

pub struct SuccessorIter(::std::vec::IntoIter<BasicBlock>);
impl<'a> Iterator for SuccessorIter {
    type Item = BasicBlock;
    fn next(&mut self) -> Option<BasicBlock> {
        self.0.next()
    }
}

pub struct SuccessorIterMut<'a>(::std::vec::IntoIter<&'a mut BasicBlock>);
impl<'a> Iterator for SuccessorIterMut<'a> {
    type Item = &'a mut BasicBlock;
    fn next(&mut self) -> Option<&'a mut BasicBlock> {
        self.0.next()
    }
}

impl<'tcx> CFG<'tcx> {
    pub fn len(&self) -> usize {
        self.basic_blocks.len()
    }

    pub fn predecessors(&self, b: BasicBlock) -> PredecessorIter {
        let mut preds = vec![];
        for idx in 0..self.len() {
            let bb = BasicBlock::new(idx);
            if let Some(_) = self.successors(bb).find(|&x| x == b) {
                preds.push(bb)
            }
        }
        PredecessorIter(preds.into_iter())
    }

    pub fn successors(&self, b: BasicBlock) -> SuccessorIter {
        let v: Vec<BasicBlock> = self[b].terminator().kind.successors().into_owned();
        SuccessorIter(v.into_iter())
    }

    pub fn successors_mut(&mut self, b: BasicBlock) -> SuccessorIterMut {
        SuccessorIterMut(self[b].terminator_mut().kind.successors_mut().into_iter())
    }


    pub fn swap(&mut self, b1: BasicBlock, b2: BasicBlock) {
        // TODO: find all the branches to b2 from subgraph starting at b2 and replace them with b1.
        self.basic_blocks.swap(b1.index(), b2.index());
    }

    pub fn start_new_block(&mut self) -> BasicBlock {
        let node_index = self.basic_blocks.len();
        self.basic_blocks.push(BasicBlockData::new(None));
        BasicBlock::new(node_index)
    }

    pub fn start_new_cleanup_block(&mut self) -> BasicBlock {
        let bb = self.start_new_block();
        self[bb].is_cleanup = true;
        bb
    }

    pub fn push(&mut self, block: BasicBlock, statement: Statement<'tcx>) {
        debug!("push({:?}, {:?})", block, statement);
        self[block].statements.push(statement);
    }

    pub fn terminate(&mut self,
                     block: BasicBlock,
                     scope: ScopeId,
                     span: Span,
                     kind: TerminatorKind<'tcx>) {
        debug_assert!(self[block].terminator.is_none(),
                      "terminate: block {:?} already has a terminator set", block);
        self[block].terminator = Some(Terminator {
            span: span,
            scope: scope,
            kind: kind,
        });
    }

    pub fn push_assign(&mut self,
                       block: BasicBlock,
                       scope: ScopeId,
                       span: Span,
                       lvalue: &Lvalue<'tcx>,
                       rvalue: Rvalue<'tcx>) {
        self.push(block, Statement {
            scope: scope,
            span: span,
            kind: StatementKind::Assign(lvalue.clone(), rvalue)
        });
    }

    pub fn push_assign_constant(&mut self,
                                block: BasicBlock,
                                scope: ScopeId,
                                span: Span,
                                temp: &Lvalue<'tcx>,
                                constant: Constant<'tcx>) {
        self.push_assign(block, scope, span, temp,
                         Rvalue::Use(Operand::Constant(constant)));
    }

    pub fn push_assign_unit(&mut self,
                            block: BasicBlock,
                            scope: ScopeId,
                            span: Span,
                            lvalue: &Lvalue<'tcx>) {
        self.push_assign(block, scope, span, lvalue, Rvalue::Aggregate(
            AggregateKind::Tuple, vec![]
        ));
    }
}

impl<'tcx> Index<BasicBlock> for CFG<'tcx> {
    type Output = BasicBlockData<'tcx>;

    #[inline]
    fn index(&self, index: BasicBlock) -> &BasicBlockData<'tcx> {
        &self.basic_blocks[index.index()]
    }
}

impl<'tcx> IndexMut<BasicBlock> for CFG<'tcx> {
    #[inline]
    fn index_mut(&mut self, index: BasicBlock) -> &mut BasicBlockData<'tcx> {
        &mut self.basic_blocks[index.index()]
    }
}

