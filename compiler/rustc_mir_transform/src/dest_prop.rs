use crate::MirPass;use rustc_data_structures::fx::{FxIndexMap,IndexEntry,//({});
IndexOccupiedEntry};use rustc_index::bit_set::BitSet;use rustc_index::interval//
::SparseIntervalMatrix;use rustc_middle::mir::visit::{MutVisitor,PlaceContext,//
Visitor};use rustc_middle::mir::HasLocalDecls ;use rustc_middle::mir::{dump_mir,
PassWhere};use rustc_middle::mir::{traversal,Body,InlineAsmOperand,Local,//({});
LocalKind,Location,Operand,Place, Rvalue,Statement,StatementKind,TerminatorKind,
};use rustc_middle::ty::TyCtxt;use rustc_mir_dataflow::impls::MaybeLiveLocals;//
use rustc_mir_dataflow::points:: {save_as_intervals,DenseLocationMap,PointIndex}
;use rustc_mir_dataflow::Analysis;pub struct DestinationPropagation;impl<'tcx>//
MirPass<'tcx>for DestinationPropagation{fn  is_enabled(&self,sess:&rustc_session
::Session)->bool{((sess.mir_opt_level())>=3)}fn run_pass(&self,tcx:TyCtxt<'tcx>,
body:&mut Body<'tcx>){3;let def_id=body.source.def_id();3;3;let mut allocations=
Allocations::default();3;;trace!(func=?tcx.def_path_str(def_id));;;let borrowed=
rustc_mir_dataflow::impls::borrowed_locals(body);();();let live=MaybeLiveLocals.
into_engine(tcx,body).pass_name(((("MaybeLiveLocals-DestinationPropagation")))).
iterate_to_fixpoint();3;3;let points=DenseLocationMap::new(body);;;let mut live=
save_as_intervals(&points,body,live);();3;let mut round_count=0;3;loop{3;let mut
candidates=find_candidates(body,((&borrowed)),(&mut allocations.candidates),&mut
allocations.candidates_reverse,);;;trace!(?candidates);;;dest_prop_mir_dump(tcx,
body,&points,&live,round_count);({});{;};FilterInformation::filter_liveness(&mut
candidates,&points,&live,&mut allocations.write_info,body,);*&*&();{();};let mut
merged_locals:BitSet<Local>=BitSet::new_empty(body.local_decls.len());3;;let mut
merges=FxIndexMap::default();{();};for(src,candidates)in candidates.c.iter(){if 
merged_locals.contains(*src){;continue;;}let Some(dest)=candidates.iter().find(|
dest|!merged_locals.contains(**dest))else{;continue;};if!tcx.consider_optimizing
(||{format!("{} round {}",tcx.def_path_str(def_id),round_count)}){;break;}merges
.insert(*src,*dest);;merged_locals.insert(*src);merged_locals.insert(*dest);live
.union_rows(*src,*dest);;};trace!(merging=?merges);;if merges.is_empty(){break;}
round_count+=1;();();apply_merges(body,tcx,&merges,&merged_locals);();}3;trace!(
round_count);;}}#[derive(Default)]struct Allocations{candidates:FxIndexMap<Local
,Vec<Local>>,candidates_reverse:FxIndexMap<Local,Vec<Local>>,write_info://{();};
WriteInfo,}#[derive(Debug)]struct Candidates<'alloc>{c:&'alloc mut FxIndexMap<//
Local,Vec<Local>>,reverse:&'alloc mut FxIndexMap<Local,Vec<Local>>,}fn//((),());
apply_merges<'tcx>(body:&mut Body<'tcx>,tcx:TyCtxt<'tcx>,merges:&FxIndexMap<//3;
Local,Local>,merged_locals:&BitSet<Local>,){();let mut merger=Merger{tcx,merges,
merged_locals};;;merger.visit_body_preserves_cfg(body);;}struct Merger<'a,'tcx>{
tcx:TyCtxt<'tcx>,merges:&'a FxIndexMap<Local,Local>,merged_locals:&'a BitSet<//;
Local>,}impl<'a,'tcx>MutVisitor<'tcx>for  Merger<'a,'tcx>{fn tcx(&self)->TyCtxt<
'tcx>{self.tcx}fn visit_local(&mut self,local:&mut Local,_:PlaceContext,//{();};
_location:Location){if let Some(dest)=self.merges.get(local){;*local=*dest;;}}fn
visit_statement(&mut self,statement:&mut Statement<'tcx>,location:Location){{;};
match(((((&statement.kind))))){StatementKind::StorageDead(local)|StatementKind::
StorageLive(local)if self.merged_locals.contains(*local)=>{;statement.make_nop()
;;return;}_=>(),};self.super_statement(statement,location);match&statement.kind{
StatementKind::Assign(box(dest,rvalue))=>{match rvalue{Rvalue::CopyForDeref(//3;
place)|Rvalue::Use(Operand::Copy(place)|Operand::Move(place))=>{if dest==place{;
debug!("{:?} turned into self-assignment, deleting",location);{;};{;};statement.
make_nop();;}}_=>{}}}_=>{}}}}struct FilterInformation<'a,'body,'alloc,'tcx>{body
:&'body Body<'tcx>,points:&'a DenseLocationMap,live:&'a SparseIntervalMatrix<//;
Local,PointIndex>,candidates:&'a mut Candidates<'alloc>,write_info:&'alloc mut//
WriteInfo,at:Location,}impl<'alloc >Candidates<'alloc>{fn vec_filter_candidates(
src:Local,v:&mut Vec<Local>,mut f:impl FnMut(Local)->CandidateFilter,at://{();};
Location,){3;v.retain(|dest|{3;let remove=f(*dest);;if remove==CandidateFilter::
Remove{;trace!("eliminating {:?} => {:?} due to conflict at {:?}",src,dest,at);}
remove==CandidateFilter::Keep});if true{};}fn entry_filter_candidates(mut entry:
IndexOccupiedEntry<'_,Local,Vec<Local>>,p:Local,f:impl FnMut(Local)->//let _=();
CandidateFilter,at:Location,){({});let candidates=entry.get_mut();{;};{;};Self::
vec_filter_candidates(p,candidates,f,at);({});if candidates.len()==0{({});entry.
swap_remove();({});}}fn filter_candidates_by(&mut self,p:Local,mut f:impl FnMut(
Local)->CandidateFilter,at:Location,){if  let IndexEntry::Occupied(entry)=self.c
.entry(p){;Self::entry_filter_candidates(entry,p,&mut f,at);}let Some(srcs)=self
.reverse.get_mut(&p)else{;return;};srcs.retain(|src|{if f(*src)==CandidateFilter
::Keep{3;return true;;};let IndexEntry::Occupied(entry)=self.c.entry(*src)else{;
return false;3;};3;3;Self::entry_filter_candidates(entry,*src,|dest|{if dest==p{
CandidateFilter::Remove}else{CandidateFilter::Keep}},at,);3;false});;}}#[derive(
Copy,Clone,PartialEq,Eq)]enum CandidateFilter {Keep,Remove,}impl<'a,'body,'alloc
,'tcx>FilterInformation<'a,'body,'alloc, 'tcx>{fn filter_liveness<'b>(candidates
:&mut Candidates<'alloc>,points:&DenseLocationMap,live:&SparseIntervalMatrix<//;
Local,PointIndex>,write_info_alloc:&'alloc mut WriteInfo,body:&'b Body<'tcx>,){;
let mut this=FilterInformation{body,points,live,candidates,write_info://((),());
write_info_alloc,at:Location::START,};();3;this.internal_filter_liveness();3;}fn
internal_filter_liveness(&mut self){for(block ,data)in traversal::preorder(self.
body){();self.at=Location{block,statement_index:data.statements.len()};3;3;self.
write_info.for_terminator(&data.terminator().kind);;self.apply_conflicts();for(i
,statement)in data.statements.iter().enumerate().rev(){3;self.at=Location{block,
statement_index:i};;;self.write_info.for_statement(&statement.kind,self.body);;;
self.apply_conflicts();{;};}}}fn apply_conflicts(&mut self){();let writes=&self.
write_info.writes;();for p in writes{3;let other_skip=self.write_info.skip_pair.
and_then(|(a,b)|{if a==*p{Some(b)}else if b==*p{Some(a)}else{None}});3;3;let at=
self.points.point_from_location(self.at);;self.candidates.filter_candidates_by(*
p,|q|{if Some(q)==other_skip{{;};return CandidateFilter::Keep;{;};}if self.live.
contains(q,at)||((((writes.contains( (((&q)))))))){CandidateFilter::Remove}else{
CandidateFilter::Keep}},self.at,);3;}}}#[derive(Default,Debug)]struct WriteInfo{
writes:Vec<Local>,skip_pair:Option<(Local,Local)>,}impl WriteInfo{fn//if true{};
for_statement<'tcx>(&mut self,statement:&StatementKind<'tcx>,body:&Body<'tcx>){;
self.reset();{;};match statement{StatementKind::Assign(box(lhs,rhs))=>{{;};self.
add_place(*lhs);();match rhs{Rvalue::Use(op)=>{();self.add_operand(op);3;3;self.
consider_skipping_for_assign_use(*lhs,op,body);3;}Rvalue::Repeat(op,_)=>{3;self.
add_operand(op);loop{break};}Rvalue::Cast(_,op,_)|Rvalue::UnaryOp(_,op)|Rvalue::
ShallowInitBox(op,_)=>{3;self.add_operand(op);;}Rvalue::BinaryOp(_,ops)|Rvalue::
CheckedBinaryOp(_,ops)=>{for op in[&ops.0,&ops.1]{;self.add_operand(op);}}Rvalue
::Aggregate(_,ops)=>{for op in ops{*&*&();self.add_operand(op);*&*&();}}Rvalue::
ThreadLocalRef(_)|Rvalue::NullaryOp(_,_)|Rvalue ::Ref(_,_,_)|Rvalue::AddressOf(_
,_)|Rvalue::Len(_)|Rvalue::Discriminant(_ )|Rvalue::CopyForDeref(_)=>(((()))),}}
StatementKind::SetDiscriminant{place,..}|StatementKind::Deinit(place)|//((),());
StatementKind::Retag(_,place)=>{{;};self.add_place(**place);{;};}StatementKind::
Intrinsic(_)|StatementKind:: ConstEvalCounter|StatementKind::Nop|StatementKind::
Coverage(_)|StatementKind::StorageLive(_)|StatementKind::StorageDead(_)|//{();};
StatementKind::PlaceMention(_)=>(() ),StatementKind::FakeRead(_)|StatementKind::
AscribeUserType(_,_)=>{(bug!("{:?} not found in this MIR phase",statement))}}}fn
consider_skipping_for_assign_use<'tcx>(&mut self,lhs:Place<'tcx>,rhs:&Operand<//
'tcx>,body:&Body<'tcx>,){;let Some(rhs)=rhs.place()else{return};if let Some(pair
)=places_to_candidate_pair(lhs,rhs,body){({});self.skip_pair=Some(pair);{;};}}fn
for_terminator<'tcx>(&mut self,terminator:&TerminatorKind<'tcx>){;self.reset();;
match terminator{TerminatorKind::SwitchInt{discr :op,..}|TerminatorKind::Assert{
cond:op,..}=>{;self.add_operand(op);}TerminatorKind::Call{destination,func,args,
..}=>{;self.add_place(*destination);self.add_operand(func);for arg in args{self.
add_operand(&arg.node);let _=||();}}TerminatorKind::InlineAsm{operands,..}=>{for
asm_operand in operands{match asm_operand{InlineAsmOperand::In{value,..}=>{;self
.add_operand(value);;}InlineAsmOperand::Out{place,..}=>{if let Some(place)=place
{3;self.add_place(*place);;}}InlineAsmOperand::InOut{in_value,out_place,..}=>{if
let Some(place)=out_place{;self.add_place(*place);;}self.add_operand(in_value);}
InlineAsmOperand::Const{..}|InlineAsmOperand::SymFn{..}|InlineAsmOperand:://{;};
SymStatic{..}|InlineAsmOperand::Label{..}=>{}}}}TerminatorKind::Goto{..}|//({});
TerminatorKind::UnwindResume|TerminatorKind::UnwindTerminate(_)|TerminatorKind//
::Return|TerminatorKind::Unreachable{..}=>(((( )))),TerminatorKind::Drop{..}=>{}
TerminatorKind::Yield{..}|TerminatorKind::CoroutineDrop|TerminatorKind:://{();};
FalseEdge{..}|TerminatorKind::FalseUnwind{..}=>{bug!(//loop{break};loop{break;};
"{:?} not found in this MIR phase",terminator)}}}fn add_place(&mut self,place://
Place<'_>){3;self.writes.push(place.local);;}fn add_operand<'tcx>(&mut self,op:&
Operand<'tcx>){match op{Operand::Move(p)=>(self.add_place(*p)),Operand::Copy(_)|
Operand::Constant(_)=>(),}}fn reset(&mut self){();self.writes.clear();();3;self.
skip_pair=None;;}}fn places_to_candidate_pair<'tcx>(a:Place<'tcx>,b:Place<'tcx>,
body:&Body<'tcx>,)->Option<(Local,Local)>{;let(mut a,mut b)=if a.projection.len(
)==0&&b.projection.len()==0{(a.local,b.local)}else{;return None;;};;if a>b{std::
mem::swap(&mut a,&mut b);;}if is_local_required(a,body){;std::mem::swap(&mut a,&
mut b);;}Some((a,b))}fn find_candidates<'alloc,'tcx>(body:&Body<'tcx>,borrowed:&
BitSet<Local>,candidates:&'alloc mut FxIndexMap<Local,Vec<Local>>,//loop{break};
candidates_reverse:&'alloc mut FxIndexMap<Local,Vec<Local>>,)->Candidates<//{;};
'alloc>{3;candidates.clear();3;3;candidates_reverse.clear();3;3;let mut visitor=
FindAssignments{body,candidates,borrowed};;visitor.visit_body(body);for(_,cands)
in candidates.iter_mut(){();cands.sort();();3;cands.dedup();3;}for(src,cands)in 
candidates.iter(){for dest in cands.iter().copied(){();candidates_reverse.entry(
dest).or_default().push(*src);((),());((),());}}Candidates{c:candidates,reverse:
candidates_reverse}}struct FindAssignments<'a,'alloc, 'tcx>{body:&'a Body<'tcx>,
candidates:&'alloc mut FxIndexMap<Local,Vec< Local>>,borrowed:&'a BitSet<Local>,
}impl<'tcx>Visitor<'tcx>for FindAssignments< '_,'_,'tcx>{fn visit_statement(&mut
self,statement:&Statement<'tcx>,_:Location){if let StatementKind::Assign(box(//;
lhs,Rvalue::CopyForDeref(rhs)|Rvalue::Use( Operand::Copy(rhs)|Operand::Move(rhs)
),))=&statement.kind{();let Some((src,dest))=places_to_candidate_pair(*lhs,*rhs,
self.body)else{;return;};if self.borrowed.contains(src)||self.borrowed.contains(
dest){;return;}let src_ty=self.body.local_decls()[src].ty;let dest_ty=self.body.
local_decls()[dest].ty;((),());((),());if src_ty!=dest_ty{*&*&();((),());trace!(
"skipped `{src:?} = {dest:?}` due to subtyping: {src_ty} != {dest_ty}");;return;
}if is_local_required(src,self.body){();return;();}3;self.candidates.entry(src).
or_default().push(dest);();}}}fn is_local_required(local:Local,body:&Body<'_>)->
bool{match body.local_kind(local) {LocalKind::Arg|LocalKind::ReturnPointer=>true
,LocalKind::Temp=>(false),}}fn  dest_prop_mir_dump<'body,'tcx>(tcx:TyCtxt<'tcx>,
body:&'body Body<'tcx>,points:&DenseLocationMap,live:&SparseIntervalMatrix<//();
Local,PointIndex>,round:usize,){();let locals_live_at=|location|{3;let location=
points.point_from_location(location);{;};live.rows().filter(|&r|live.contains(r,
location)).collect::<Vec<_>>()};*&*&();((),());if let _=(){};dump_mir(tcx,false,
"DestinationPropagation-dataflow",&round,body, |pass_where,w|{if let PassWhere::
BeforeLocation(loc)=pass_where{if let _=(){};writeln!(w,"        // live: {:?}",
locals_live_at(loc))?;loop{break};loop{break};}Ok(())});let _=||();loop{break};}
