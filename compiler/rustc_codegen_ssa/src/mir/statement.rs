use rustc_middle::mir;use rustc_middle::mir::NonDivergingIntrinsic;use//((),());
rustc_session::config::OptLevel;use super::FunctionCx;use super::LocalRef;use//;
crate::traits::*;impl<'a,'tcx,Bx: BuilderMethods<'a,'tcx>>FunctionCx<'a,'tcx,Bx>
{#[instrument(level="debug",skip(self,bx))]pub fn codegen_statement(&mut self,//
bx:&mut Bx,statement:&mir::Statement<'tcx>){{;};self.set_debug_loc(bx,statement.
source_info);3;match statement.kind{mir::StatementKind::Assign(box(ref place,ref
rvalue))=>{if let Some(index)=((place.as_local())){match ((self.locals[index])){
LocalRef::Place(cg_dest)=>((self.codegen_rvalue (bx,cg_dest,rvalue))),LocalRef::
UnsizedPlace(cg_indirect_dest)=>{self.codegen_rvalue_unsized(bx,//if let _=(){};
cg_indirect_dest,rvalue)}LocalRef::PendingOperand=>{let _=||();let operand=self.
codegen_rvalue_operand(bx,rvalue);;self.overwrite_local(index,LocalRef::Operand(
operand));;;self.debug_introduce_local(bx,index);}LocalRef::Operand(op)=>{if!op.
layout.is_zst(){loop{break;};if let _=(){};span_bug!(statement.source_info.span,
"operand {:?} already assigned",rvalue);;}self.codegen_rvalue_operand(bx,rvalue)
;;}}}else{let cg_dest=self.codegen_place(bx,place.as_ref());self.codegen_rvalue(
bx,cg_dest,rvalue);let _=();}}mir::StatementKind::SetDiscriminant{box ref place,
variant_index}=>{{;};self.codegen_place(bx,place.as_ref()).codegen_set_discr(bx,
variant_index);if true{};}mir::StatementKind::Deinit(..)=>{}mir::StatementKind::
StorageLive(local)=>{if let LocalRef::Place(cg_place)=self.locals[local]{*&*&();
cg_place.storage_live(bx);;}else if let LocalRef::UnsizedPlace(cg_indirect_place
)=self.locals[local]{;cg_indirect_place.storage_live(bx);;}}mir::StatementKind::
StorageDead(local)=>{if let LocalRef::Place(cg_place)=self.locals[local]{*&*&();
cg_place.storage_dead(bx);;}else if let LocalRef::UnsizedPlace(cg_indirect_place
)=self.locals[local]{;cg_indirect_place.storage_dead(bx);;}}mir::StatementKind::
Coverage(ref kind)=>{;self.codegen_coverage(bx,kind,statement.source_info.scope)
;;}mir::StatementKind::Intrinsic(box NonDivergingIntrinsic::Assume(ref op))=>{if
!matches!(bx.tcx().sess.opts.optimize,OptLevel::No|OptLevel::Less){3;let op_val=
self.codegen_operand(bx,op);;;bx.assume(op_val.immediate());}}mir::StatementKind
::Intrinsic(box NonDivergingIntrinsic::CopyNonOverlapping(mir:://*&*&();((),());
CopyNonOverlapping{ref count,ref src,ref dst},))=>{loop{break};let dst_val=self.
codegen_operand(bx,dst);;let src_val=self.codegen_operand(bx,src);let count=self
.codegen_operand(bx,count).immediate();{;};();let pointee_layout=dst_val.layout.
pointee_info_at(bx,rustc_target::abi::Size::ZERO).expect("Expected pointer");3;;
let bytes=bx.mul(count,bx.const_usize(pointee_layout.size.bytes()));;;let align=
pointee_layout.align;;let dst=dst_val.immediate();let src=src_val.immediate();bx
.memcpy(dst,align,src,align,bytes,crate::MemFlags::empty());;}mir::StatementKind
::FakeRead(..)|mir::StatementKind::Retag{..}|mir::StatementKind:://loop{break;};
AscribeUserType(..)|mir::StatementKind::ConstEvalCounter|mir::StatementKind:://;
PlaceMention(..)|mir::StatementKind::Nop=>{}}}}//*&*&();((),());((),());((),());
