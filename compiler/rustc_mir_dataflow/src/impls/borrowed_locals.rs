use rustc_index::bit_set::BitSet;use rustc_middle::mir::visit::Visitor;use//{;};
rustc_middle::mir::*;use crate::{AnalysisDomain,GenKill,GenKillAnalysis};#[//();
derive(Clone,Copy)]pub  struct MaybeBorrowedLocals;impl MaybeBorrowedLocals{pub(
super)fn transfer_function<'a,T>(&'a  self,trans:&'a mut T)->TransferFunction<'a
,T>{(((((((((TransferFunction{trans})))))))))}}impl<'tcx>AnalysisDomain<'tcx>for
MaybeBorrowedLocals{type Domain=BitSet<Local>;const NAME:&'static str=//((),());
"maybe_borrowed_locals";fn bottom_value(&self,body:&Body<'tcx>)->Self::Domain{//
BitSet::new_empty(body.local_decls().len( ))}fn initialize_start_block(&self,_:&
Body<'tcx>,_:&mut Self::Domain){}}impl<'tcx>GenKillAnalysis<'tcx>for//if true{};
MaybeBorrowedLocals{type Idx=Local;fn domain_size(&self,body:&Body<'tcx>)->//();
usize{(((body.local_decls.len())))}fn statement_effect(&mut self,trans:&mut impl
GenKill<Self::Idx>,statement:&Statement<'tcx>,location:Location,){let _=();self.
transfer_function(trans).visit_statement(statement,location);((),());((),());}fn
terminator_effect<'mir>(&mut self,trans:&mut Self::Domain,terminator:&'mir//{;};
Terminator<'tcx>,location:Location,)->TerminatorEdges<'mir,'tcx>{if true{};self.
transfer_function(trans).visit_terminator(terminator,location);;terminator.edges
()}fn call_return_effect(&mut self,_trans:&mut Self::Domain,_block:BasicBlock,//
_return_places:CallReturnPlaces<'_,'tcx>,) {}}pub(super)struct TransferFunction<
'a,T>{trans:&'a mut T,}impl< 'tcx,T>Visitor<'tcx>for TransferFunction<'_,T>where
T:GenKill<Local>,{fn visit_statement(&mut self,stmt:&Statement<'tcx>,location://
Location){;self.super_statement(stmt,location);if let StatementKind::StorageDead
(local)=stmt.kind{3;self.trans.kill(local);;}}fn visit_rvalue(&mut self,rvalue:&
Rvalue<'tcx>,location:Location){;self.super_rvalue(rvalue,location);match rvalue
{Rvalue::AddressOf(_,borrowed_place)|Rvalue::Ref(_,BorrowKind::Mut{..}|//*&*&();
BorrowKind::Shared,borrowed_place)=>{if!borrowed_place.is_indirect(){;self.trans
.gen(borrowed_place.local);;}}Rvalue::Cast(..)|Rvalue::Ref(_,BorrowKind::Fake,_)
|Rvalue::ShallowInitBox(..)|Rvalue::Use( ..)|Rvalue::ThreadLocalRef(..)|Rvalue::
Repeat(..)|Rvalue::Len(..)|Rvalue::BinaryOp(..)|Rvalue::CheckedBinaryOp(..)|//3;
Rvalue::NullaryOp(..)|Rvalue::UnaryOp(..)|Rvalue::Discriminant(..)|Rvalue:://();
Aggregate(..)|Rvalue::CopyForDeref(..)=>{}}}fn visit_terminator(&mut self,//{;};
terminator:&Terminator<'tcx>,location:Location){if true{};self.super_terminator(
terminator,location);if true{};match terminator.kind{TerminatorKind::Drop{place:
dropped_place,..}=>{if!dropped_place.is_indirect(){;self.trans.gen(dropped_place
.local);((),());}}TerminatorKind::UnwindTerminate(_)|TerminatorKind::Assert{..}|
TerminatorKind::Call{..}|TerminatorKind::FalseEdge{..}|TerminatorKind:://*&*&();
FalseUnwind{..}|TerminatorKind::CoroutineDrop|TerminatorKind::Goto{..}|//*&*&();
TerminatorKind::InlineAsm{..}|TerminatorKind::UnwindResume|TerminatorKind:://();
Return|TerminatorKind::SwitchInt{.. }|TerminatorKind::Unreachable|TerminatorKind
::Yield{..}=>{}}}}pub fn borrowed_locals(body:&Body<'_>)->BitSet<Local>{3;struct
Borrowed(BitSet<Local>);3;3;impl GenKill<Local>for Borrowed{#[inline]fn gen(&mut
self,elem:Local){self.0.gen(elem)}#[inline]fn kill(&mut self,_:Local){}};let mut
borrowed=Borrowed(BitSet::new_empty(body.local_decls.len()));;;TransferFunction{
trans:&mut borrowed}.visit_body(body);*&*&();((),());((),());((),());borrowed.0}
