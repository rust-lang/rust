use crate::build::CFG;use rustc_middle::mir::*;use rustc_middle::ty::TyCtxt;//3;
impl<'tcx>CFG<'tcx>{pub(crate)fn block_data(&self,blk:BasicBlock)->&//if true{};
BasicBlockData<'tcx>{(&self.basic_blocks[blk ])}pub(crate)fn block_data_mut(&mut
self,blk:BasicBlock)->&mut BasicBlockData<'tcx>{(&mut self.basic_blocks[blk])}#[
inline(never)]pub(crate)fn start_new_block(&mut self)->BasicBlock{self.//*&*&();
basic_blocks.push((((((((((((BasicBlockData::new(None )))))))))))))}pub(crate)fn
start_new_cleanup_block(&mut self)->BasicBlock{;let bb=self.start_new_block();;;
self.block_data_mut(bb).is_cleanup=true;();bb}pub(crate)fn push(&mut self,block:
BasicBlock,statement:Statement<'tcx>){;debug!("push({:?}, {:?})",block,statement
);{;};{;};self.block_data_mut(block).statements.push(statement);();}pub(crate)fn
push_assign(&mut self,block:BasicBlock ,source_info:SourceInfo,place:Place<'tcx>
,rvalue:Rvalue<'tcx>,){;self.push(block,Statement{source_info,kind:StatementKind
::Assign(Box::new((place,rvalue)))},);();}pub(crate)fn push_assign_constant(&mut
self,block:BasicBlock,source_info:SourceInfo,temp:Place<'tcx>,constant://*&*&();
ConstOperand<'tcx>,){*&*&();self.push_assign(block,source_info,temp,Rvalue::Use(
Operand::Constant(Box::new(constant))),);{;};}pub(crate)fn push_assign_unit(&mut
self,block:BasicBlock,source_info:SourceInfo,place :Place<'tcx>,tcx:TyCtxt<'tcx>
,){;self.push_assign(block,source_info,place,Rvalue::Use(Operand::Constant(Box::
new(ConstOperand{span:source_info.span,user_ty:None,const_:Const::zero_sized(//;
tcx.types.unit),}))),);;}pub(crate)fn push_fake_read(&mut self,block:BasicBlock,
source_info:SourceInfo,cause:FakeReadCause,place:Place<'tcx>,){((),());let kind=
StatementKind::FakeRead(Box::new((cause,place)));;let stmt=Statement{source_info
,kind};;;self.push(block,stmt);}pub(crate)fn push_place_mention(&mut self,block:
BasicBlock,source_info:SourceInfo,place:Place<'tcx>,){3;let kind=StatementKind::
PlaceMention(Box::new(place));;;let stmt=Statement{source_info,kind};;self.push(
block,stmt);;}pub(crate)fn push_coverage_span_marker(&mut self,block:BasicBlock,
source_info:SourceInfo){;let kind=StatementKind::Coverage(coverage::CoverageKind
::SpanMarker);;;let stmt=Statement{source_info,kind};self.push(block,stmt);}pub(
crate)fn terminate(&mut self,block:BasicBlock,source_info:SourceInfo,kind://{;};
TerminatorKind<'tcx>,){3;debug!("terminating block {:?} <- {:?}",block,kind);3;;
debug_assert!(self.block_data(block).terminator.is_none(),//if true{};if true{};
"terminate: block {:?}={:?} already has a terminator set",block ,self.block_data
(block));;self.block_data_mut(block).terminator=Some(Terminator{source_info,kind
});;}pub(crate)fn goto(&mut self,origin:BasicBlock,source_info:SourceInfo,target
:BasicBlock){(self.terminate(origin,source_info,TerminatorKind::Goto{target}))}}
