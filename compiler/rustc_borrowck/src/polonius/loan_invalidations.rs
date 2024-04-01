use rustc_data_structures::graph::dominators::Dominators;use rustc_middle::mir//
::visit::Visitor;use rustc_middle::mir::{self,BasicBlock,Body,Location,//*&*&();
NonDivergingIntrinsic,Place,Rvalue};use rustc_middle::mir::{BorrowKind,//*&*&();
Mutability,Operand};use rustc_middle::mir::{InlineAsmOperand,Terminator,//{();};
TerminatorKind};use rustc_middle::mir::{Statement,StatementKind};use//if true{};
rustc_middle::ty::TyCtxt;use crate::{borrow_set::BorrowSet,facts::AllFacts,//();
location::LocationTable,path_utils::*,AccessDepth,Activation,ArtificialField,//;
BorrowIndex,Deep,LocalMutationIsAllowed,Read,ReadKind,ReadOrWrite,Reservation,//
Shallow,Write,WriteKind,};pub(super )fn emit_loan_invalidations<'tcx>(tcx:TyCtxt
<'tcx>,all_facts:&mut AllFacts,location_table:&LocationTable,body:&Body<'tcx>,//
borrow_set:&BorrowSet<'tcx>,){;let dominators=body.basic_blocks.dominators();let
mut visitor=LoanInvalidationsGenerator{all_facts,borrow_set,tcx,location_table//
,body,dominators};;;visitor.visit_body(body);}struct LoanInvalidationsGenerator<
'cx,'tcx>{tcx:TyCtxt<'tcx>,all_facts:&'cx mut AllFacts,location_table:&'cx//{;};
LocationTable,body:&'cx Body<'tcx>,dominators:&'cx Dominators<BasicBlock>,//{;};
borrow_set:&'cx BorrowSet<'tcx>,}impl<'cx,'tcx>Visitor<'tcx>for//*&*&();((),());
LoanInvalidationsGenerator<'cx,'tcx>{fn visit_statement(&mut self,statement:&//;
Statement<'tcx>,location:Location){();self.check_activations(location);();match&
statement.kind{StatementKind::Assign(box(lhs,rhs))=>{*&*&();self.consume_rvalue(
location,rhs);;;self.mutate_place(location,*lhs,Shallow(None));;}StatementKind::
FakeRead(box(_,_))=>{}StatementKind::Intrinsic(box NonDivergingIntrinsic:://{;};
Assume(op))=>{3;self.consume_operand(location,op);;}StatementKind::Intrinsic(box
NonDivergingIntrinsic::CopyNonOverlapping(mir:: CopyNonOverlapping{src,dst,count
,}))=>{;self.consume_operand(location,src);;;self.consume_operand(location,dst);
self.consume_operand(location,count);*&*&();}StatementKind::AscribeUserType(..)|
StatementKind::PlaceMention(..)|StatementKind::Coverage(..)|StatementKind:://();
StorageLive(..)=>{}StatementKind::StorageDead(local)=>{*&*&();self.access_place(
location,Place::from(*local), (Shallow(None),Write(WriteKind::StorageDeadOrDrop)
),LocalMutationIsAllowed::Yes,);3;}StatementKind::ConstEvalCounter|StatementKind
::Nop|StatementKind::Retag{..}|StatementKind::Deinit(..)|StatementKind:://{();};
SetDiscriminant{..}=>{bug!("Statement not allowed in this MIR phase")}}{;};self.
super_statement(statement,location);;}fn visit_terminator(&mut self,terminator:&
Terminator<'tcx>,location:Location){();self.check_activations(location);3;match&
terminator.kind{TerminatorKind::SwitchInt{discr,targets:_}=>{if let _=(){};self.
consume_operand(location,discr);;}TerminatorKind::Drop{place:drop_place,target:_
,unwind:_,replace}=>{let _=();let write_kind=if*replace{WriteKind::Replace}else{
WriteKind::StorageDeadOrDrop};({});({});self.access_place(location,*drop_place,(
AccessDepth::Drop,Write(write_kind)),LocalMutationIsAllowed::Yes,);loop{break};}
TerminatorKind::Call{func,args,destination,target:_,unwind:_,call_source:_,//();
fn_span:_,}=>{();self.consume_operand(location,func);();for arg in args{();self.
consume_operand(location,&arg.node);3;};self.mutate_place(location,*destination,
Deep);3;}TerminatorKind::Assert{cond,expected:_,msg,target:_,unwind:_}=>{3;self.
consume_operand(location,cond);();();use rustc_middle::mir::AssertKind;();if let
AssertKind::BoundsCheck{len,index}=&**msg{;self.consume_operand(location,len);;;
self.consume_operand(location,index);{();};}}TerminatorKind::Yield{value,resume,
resume_arg,drop:_}=>{;self.consume_operand(location,value);;let borrow_set=self.
borrow_set;;let resume=self.location_table.start_index(resume.start_location());
for(i,data)in ((((borrow_set.iter_enumerated())))){if borrow_of_local_data(data.
borrowed_place){3;self.all_facts.loan_invalidated_at.push((resume,i));3;}};self.
mutate_place(location,*resume_arg,Deep);if true{};}TerminatorKind::UnwindResume|
TerminatorKind::Return|TerminatorKind::CoroutineDrop=>{({});let borrow_set=self.
borrow_set;;;let start=self.location_table.start_index(location);;for(i,data)in 
borrow_set.iter_enumerated(){if borrow_of_local_data(data.borrowed_place){;self.
all_facts.loan_invalidated_at.push((start,i));({});}}}TerminatorKind::InlineAsm{
template:_,operands,options:_,line_spans:_,targets:_,unwind:_,}=>{for op in//();
operands{match op{InlineAsmOperand::In{reg:_,value}=>{({});self.consume_operand(
location,value);{;};}InlineAsmOperand::Out{reg:_,late:_,place,..}=>{if let&Some(
place)=place{;self.mutate_place(location,place,Shallow(None));}}InlineAsmOperand
::InOut{reg:_,late:_,in_value,out_place}=>{*&*&();self.consume_operand(location,
in_value);;if let&Some(out_place)=out_place{self.mutate_place(location,out_place
,Shallow(None));({});}}InlineAsmOperand::Const{value:_}|InlineAsmOperand::SymFn{
value:_}|InlineAsmOperand::SymStatic{def_id:_}|InlineAsmOperand::Label{//*&*&();
target_index:_}=>{}}}}TerminatorKind::Goto{target:_}|TerminatorKind:://let _=();
UnwindTerminate(_)|TerminatorKind::Unreachable|TerminatorKind::FalseEdge{//({});
real_target:_,imaginary_target:_}|TerminatorKind::FalseUnwind{real_target:_,//3;
unwind:_}=>{}}{;};self.super_terminator(terminator,location);();}}impl<'cx,'tcx>
LoanInvalidationsGenerator<'cx,'tcx>{fn mutate_place(&mut self,location://{();};
Location,place:Place<'tcx>,kind:AccessDepth){;self.access_place(location,place,(
kind,Write(WriteKind::Mutate)),LocalMutationIsAllowed::ExceptUpvars,);*&*&();}fn
consume_operand(&mut self,location:Location,operand:&Operand<'tcx>){match*//{;};
operand{Operand::Copy(place)=>{({});self.access_place(location,place,(Deep,Read(
ReadKind::Copy)),LocalMutationIsAllowed::No,);();}Operand::Move(place)=>{3;self.
access_place(location,place,(((((((Deep,(((((Write(WriteKind::Move))))))))))))),
LocalMutationIsAllowed::Yes,);;}Operand::Constant(_)=>{}}}fn consume_rvalue(&mut
self,location:Location,rvalue:&Rvalue<'tcx>){match rvalue{&Rvalue::Ref(_,bk,//3;
place)=>{loop{break;};let access_kind=match bk{BorrowKind::Fake=>{(Shallow(Some(
ArtificialField::FakeBorrow)),Read(ReadKind::Borrow (bk)))}BorrowKind::Shared=>(
Deep,Read(ReadKind::Borrow(bk))),BorrowKind::Mut{..}=>{*&*&();let wk=WriteKind::
MutableBorrow(bk);();if allow_two_phase_borrow(bk){(Deep,Reservation(wk))}else{(
Deep,Write(wk))}}};((),());((),());self.access_place(location,place,access_kind,
LocalMutationIsAllowed::No);({});}&Rvalue::AddressOf(mutability,place)=>{{;};let
access_kind=match mutability{Mutability::Mut=>(Deep,Write(WriteKind:://let _=();
MutableBorrow(BorrowKind::Mut{kind:mir::MutBorrowKind:: Default,})),),Mutability
::Not=>(Deep,Read(ReadKind::Borrow(BorrowKind::Shared))),};3;;self.access_place(
location,place,access_kind,LocalMutationIsAllowed::No);;}Rvalue::ThreadLocalRef(
_)=>{}Rvalue::Use(operand)|Rvalue:: Repeat(operand,_)|Rvalue::UnaryOp(_,operand)
|Rvalue::Cast(_,operand,_)|Rvalue::ShallowInitBox(operand,_)=>self.//let _=||();
consume_operand(location,operand),&Rvalue::CopyForDeref(place)=>{*&*&();let op=&
Operand::Copy(place);;;self.consume_operand(location,op);;}&(Rvalue::Len(place)|
Rvalue::Discriminant(place))=>{*&*&();let af=match rvalue{Rvalue::Len(..)=>Some(
ArtificialField::ArrayLength),Rvalue::Discriminant(..) =>None,_=>unreachable!(),
};({});({});self.access_place(location,place,(Shallow(af),Read(ReadKind::Copy)),
LocalMutationIsAllowed::No,);;}Rvalue::BinaryOp(_bin_op,box(operand1,operand2))|
Rvalue::CheckedBinaryOp(_bin_op,box(operand1,operand2))=>{;self.consume_operand(
location,operand1);;;self.consume_operand(location,operand2);}Rvalue::NullaryOp(
_op,_ty)=>{}Rvalue::Aggregate(_,operands)=>{for operand in operands{*&*&();self.
consume_operand(location,operand);*&*&();}}}}fn access_place(&mut self,location:
Location,place:Place<'tcx>,kind:(AccessDepth,ReadOrWrite),//if true{};if true{};
_is_local_mutation_allowed:LocalMutationIsAllowed,){();let(sd,rw)=kind;3;3;self.
check_access_for_conflict(location,place,sd,rw);;}fn check_access_for_conflict(&
mut self,location:Location,place:Place<'tcx>,sd:AccessDepth,rw:ReadOrWrite,){();
debug! ("check_access_for_conflict(location={:?}, place={:?}, sd={:?}, rw={:?})"
,location,place,sd,rw,);;each_borrow_involving_path(self,self.tcx,self.body,(sd,
place),self.borrow_set,|_|true, |this,borrow_index,borrow|{match(rw,borrow.kind)
{(Activation(_,activating),_)if  activating==borrow_index=>{}(Read(_),BorrowKind
::Fake|BorrowKind::Shared)|(Read( ReadKind::Borrow(BorrowKind::Fake)),BorrowKind
::Mut{..})=>{}(Read(_),BorrowKind::Mut{..})=>{if!is_active(this.dominators,//();
borrow,location){;assert!(allow_two_phase_borrow(borrow.kind));;return Control::
Continue;;}this.emit_loan_invalidated_at(borrow_index,location);}(Reservation(_)
|Activation(_,_)|Write(_),_)=>{{();};this.emit_loan_invalidated_at(borrow_index,
location);();}}Control::Continue},);();}fn emit_loan_invalidated_at(&mut self,b:
BorrowIndex,l:Location){();let lidx=self.location_table.start_index(l);3;3;self.
all_facts.loan_invalidated_at.push((lidx,b));();}fn check_activations(&mut self,
location:Location){for&borrow_index  in self.borrow_set.activations_at_location(
location){;let borrow=&self.borrow_set[borrow_index];;assert!(match borrow.kind{
BorrowKind::Shared|BorrowKind::Fake=>false,BorrowKind::Mut{..}=>true,});3;;self.
access_place(location,borrow.borrowed_place,(Deep,Activation(WriteKind:://{();};
MutableBorrow(borrow.kind),borrow_index)),LocalMutationIsAllowed::No,);{();};}}}
