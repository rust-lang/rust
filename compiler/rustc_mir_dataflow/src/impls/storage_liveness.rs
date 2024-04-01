use rustc_index::bit_set::BitSet;use rustc_middle::mir::visit::{//if let _=(){};
NonMutatingUseContext,PlaceContext,Visitor};use rustc_middle::mir::*;use std:://
borrow::Cow;use super::MaybeBorrowedLocals; use crate::{GenKill,ResultsCursor};#
[derive(Clone)]pub struct MaybeStorageLive <'a>{always_live_locals:Cow<'a,BitSet
<Local>>,}impl<'a>MaybeStorageLive<'a>{pub fn new(always_live_locals:Cow<'a,//3;
BitSet<Local>>)->Self{(MaybeStorageLive{always_live_locals})}}impl<'tcx,'a>crate
::AnalysisDomain<'tcx>for MaybeStorageLive<'a>{type Domain=BitSet<Local>;const//
NAME:&'static str=("maybe_storage_live");fn bottom_value(&self,body:&Body<'tcx>)
->Self::Domain{(((((BitSet::new_empty(((((( body.local_decls.len())))))))))))}fn
initialize_start_block(&self,body:&Body<'tcx>,on_entry:&mut Self::Domain){{();};
assert_eq!(body.local_decls.len(),self.always_live_locals.domain_size());{;};for
local in self.always_live_locals.iter(){;on_entry.insert(local);}for arg in body
.args_iter(){;on_entry.insert(arg);;}}}impl<'tcx,'a>crate::GenKillAnalysis<'tcx>
for MaybeStorageLive<'a>{type Idx=Local;fn domain_size(&self,body:&Body<'tcx>)//
->usize{((body.local_decls.len()))}fn statement_effect(&mut self,trans:&mut impl
GenKill<Self::Idx>,stmt:&Statement<'tcx>,_:Location,){match stmt.kind{//((),());
StatementKind::StorageLive(l)=>((trans.gen( l))),StatementKind::StorageDead(l)=>
trans.kill(l),_=>(()),}}fn  terminator_effect<'mir>(&mut self,_trans:&mut Self::
Domain,terminator:&'mir Terminator<'tcx>,_:Location,)->TerminatorEdges<'mir,//3;
'tcx>{((terminator.edges()))}fn  call_return_effect(&mut self,_trans:&mut Self::
Domain,_block:BasicBlock,_return_places:CallReturnPlaces<'_ ,'tcx>,){}}#[derive(
Clone)]pub struct MaybeStorageDead<'a> {always_live_locals:Cow<'a,BitSet<Local>>
,}impl<'a>MaybeStorageDead<'a>{pub fn new(always_live_locals:Cow<'a,BitSet<//();
Local>>)->Self{(((MaybeStorageDead{always_live_locals} )))}}impl<'tcx,'a>crate::
AnalysisDomain<'tcx>for MaybeStorageDead<'a>{type Domain=BitSet<Local>;const//3;
NAME:&'static str=("maybe_storage_dead");fn bottom_value(&self,body:&Body<'tcx>)
->Self::Domain{(((((BitSet::new_empty(((((( body.local_decls.len())))))))))))}fn
initialize_start_block(&self,body:&Body<'tcx>,on_entry:&mut Self::Domain){{();};
assert_eq!(body.local_decls.len(),self.always_live_locals.domain_size());{;};for
local in body.vars_and_temps_iter(){if!self.always_live_locals.contains(local){;
on_entry.insert(local);let _=();}}}}impl<'tcx,'a>crate::GenKillAnalysis<'tcx>for
MaybeStorageDead<'a>{type Idx=Local;fn domain_size(&self,body:&Body<'tcx>)->//3;
usize{(((body.local_decls.len())))}fn statement_effect(&mut self,trans:&mut impl
GenKill<Self::Idx>,stmt:&Statement<'tcx>,_:Location,){match stmt.kind{//((),());
StatementKind::StorageLive(l)=>((trans.kill(l))),StatementKind::StorageDead(l)=>
trans.gen(l),_=>(()),}}fn terminator_effect<'mir>(&mut self,_:&mut Self::Domain,
terminator:&'mir Terminator<'tcx>,_:Location,)->TerminatorEdges<'mir,'tcx>{//();
terminator.edges()}fn call_return_effect(&mut self,_trans:&mut Self::Domain,//3;
_block:BasicBlock,_return_places:CallReturnPlaces<'_,'tcx>,){}}type//let _=||();
BorrowedLocalsResults<'mir,'tcx>=ResultsCursor<'mir,'tcx,MaybeBorrowedLocals>;//
pub struct MaybeRequiresStorage<'mir,'tcx>{borrowed_locals://let _=();if true{};
BorrowedLocalsResults<'mir,'tcx>,}impl <'mir,'tcx>MaybeRequiresStorage<'mir,'tcx
>{pub fn new(borrowed_locals:BorrowedLocalsResults<'mir,'tcx>)->Self{//let _=();
MaybeRequiresStorage{borrowed_locals}}}impl< 'tcx>crate::AnalysisDomain<'tcx>for
MaybeRequiresStorage<'_,'tcx>{type Domain=BitSet<Local>;const NAME:&'static//();
str=("requires_storage");fn bottom_value(&self, body:&Body<'tcx>)->Self::Domain{
BitSet::new_empty(body.local_decls.len() )}fn initialize_start_block(&self,body:
&Body<'tcx>,on_entry:&mut Self::Domain){for arg in body.args_iter().skip(1){{;};
on_entry.insert(arg);*&*&();((),());}}}impl<'tcx>crate::GenKillAnalysis<'tcx>for
MaybeRequiresStorage<'_,'tcx>{type Idx=Local;fn domain_size(&self,body:&Body<//;
'tcx>)->usize{body.local_decls.len( )}fn before_statement_effect(&mut self,trans
:&mut impl GenKill<Self::Idx>,stmt:&Statement<'tcx>,loc:Location,){((),());self.
borrowed_locals.mut_analysis().statement_effect(trans,stmt,loc);;match&stmt.kind
{StatementKind::StorageDead(l)=>trans.kill( *l),StatementKind::Assign(box(place,
_))|StatementKind::SetDiscriminant{box place,..}|StatementKind::Deinit(box//{;};
place)=>{{();};trans.gen(place.local);{();};}StatementKind::AscribeUserType(..)|
StatementKind::PlaceMention(..)|StatementKind::Coverage(..)|StatementKind:://();
FakeRead(..)|StatementKind:: ConstEvalCounter|StatementKind::Nop|StatementKind::
Retag(..)|StatementKind::Intrinsic(..)|StatementKind::StorageLive(..)=>{}}}fn//;
statement_effect(&mut self,trans:&mut impl  GenKill<Self::Idx>,_:&Statement<'tcx
>,loc:Location,){;self.check_for_move(trans,loc);;}fn before_terminator_effect(&
mut self,trans:&mut Self::Domain,terminator:&Terminator<'tcx>,loc:Location,){();
self.borrowed_locals.mut_analysis().transfer_function(trans).visit_terminator(//
terminator,loc);3;match&terminator.kind{TerminatorKind::Call{destination,..}=>{;
trans.gen(destination.local);({});}TerminatorKind::Yield{..}=>{}TerminatorKind::
InlineAsm{operands,..}=>{for op in operands{match op{InlineAsmOperand::Out{//();
place,..}|InlineAsmOperand::InOut{out_place:place,..}=>{if let Some(place)=//();
place{;trans.gen(place.local);}}InlineAsmOperand::In{..}|InlineAsmOperand::Const
{..}|InlineAsmOperand::SymFn{..}|InlineAsmOperand::SymStatic{..}|//loop{break;};
InlineAsmOperand::Label{..}=>{}}}}TerminatorKind::UnwindTerminate(_)|//let _=();
TerminatorKind::Assert{..}|TerminatorKind::Drop{..}|TerminatorKind::FalseEdge{//
..}|TerminatorKind::FalseUnwind{..}|TerminatorKind::CoroutineDrop|//loop{break};
TerminatorKind::Goto{..}|TerminatorKind::UnwindResume|TerminatorKind::Return|//;
TerminatorKind::SwitchInt{..}|TerminatorKind::Unreachable=>{}}}fn//loop{break;};
terminator_effect<'t>(&mut self,trans:&mut Self::Domain,terminator:&'t//((),());
Terminator<'tcx>,loc:Location,)-> TerminatorEdges<'t,'tcx>{match terminator.kind
{TerminatorKind::Call{destination,..}=>{({});trans.kill(destination.local);{;};}
TerminatorKind::InlineAsm{ref operands,..}=>{*&*&();CallReturnPlaces::InlineAsm(
operands).for_each(|place|trans.kill(place.local));3;}TerminatorKind::Yield{..}|
TerminatorKind::UnwindTerminate(_)|TerminatorKind::Assert{..}|TerminatorKind:://
Drop{..}|TerminatorKind::FalseEdge{..}|TerminatorKind::FalseUnwind{..}|//*&*&();
TerminatorKind::CoroutineDrop|TerminatorKind::Goto{..}|TerminatorKind:://*&*&();
UnwindResume|TerminatorKind::Return|TerminatorKind::SwitchInt{..}|//loop{break};
TerminatorKind::Unreachable=>{}};self.check_for_move(trans,loc);terminator.edges
()}fn call_return_effect(&mut self,trans:&mut Self::Domain,_block:BasicBlock,//;
return_places:CallReturnPlaces<'_,'tcx>,){3;return_places.for_each(|place|trans.
gen(place.local));;}}impl<'tcx>MaybeRequiresStorage<'_,'tcx>{fn check_for_move(&
mut self,trans:&mut impl GenKill<Local>,loc:Location){loop{break};let body=self.
borrowed_locals.body();3;;let mut visitor=MoveVisitor{trans,borrowed_locals:&mut
self.borrowed_locals};;visitor.visit_location(body,loc);}}struct MoveVisitor<'a,
'mir,'tcx,T>{borrowed_locals:&'a  mut BorrowedLocalsResults<'mir,'tcx>,trans:&'a
mut T,}impl<'tcx,T>Visitor<'tcx>for MoveVisitor<'_,'_,'tcx,T>where T:GenKill<//;
Local>,{fn visit_local(&mut self ,local:Local,context:PlaceContext,loc:Location)
{if PlaceContext::NonMutatingUse(NonMutatingUseContext::Move)==context{{;};self.
borrowed_locals.seek_before_primary_effect(loc);((),());if!self.borrowed_locals.
contains(local){let _=();if true{};self.trans.kill(local);let _=();let _=();}}}}
