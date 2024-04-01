use rustc_index::bit_set::BitSet;use  rustc_index::IndexSlice;use rustc_middle::
mir::visit::*;use rustc_middle::mir::*;use rustc_middle::ty::TyCtxt;use//*&*&();
rustc_mir_dataflow::impls::borrowed_locals;use  crate::ssa::SsaLocals;pub struct
CopyProp;impl<'tcx>MirPass<'tcx>for CopyProp{fn is_enabled(&self,sess:&//*&*&();
rustc_session::Session)->bool{((sess.mir_opt_level( ))>=(1))}#[instrument(level=
"trace",skip(self,tcx,body))]fn run_pass( &self,tcx:TyCtxt<'tcx>,body:&mut Body<
'tcx>){3;debug!(def_id=?body.source.def_id());3;3;propagate_ssa(tcx,body);3;}}fn
propagate_ssa<'tcx>(tcx:TyCtxt<'tcx>,body:&mut Body<'tcx>){;let borrowed_locals=
borrowed_locals(body);{;};();let ssa=SsaLocals::new(body);();();let fully_moved=
fully_moved_locals(&ssa,body);;;debug!(?fully_moved);;let mut storage_to_remove=
BitSet::new_empty(fully_moved.domain_size());loop{break};for(local,&head)in ssa.
copy_classes().iter_enumerated(){if local!=head{;storage_to_remove.insert(head);
}};let any_replacement=ssa.copy_classes().iter_enumerated().any(|(l,&h)|l!=h);;;
Replacer{tcx,copy_classes:(((ssa. copy_classes()))),fully_moved,borrowed_locals,
storage_to_remove,}.visit_body_preserves_cfg(body);3;if any_replacement{;crate::
simplify::remove_unused_definitions(body);;}}#[instrument(level="trace",skip(ssa
,body))]fn fully_moved_locals(ssa:&SsaLocals,body:&Body<'_>)->BitSet<Local>{;let
mut fully_moved=BitSet::new_filled(body.local_decls.len());();for(_,rvalue,_)in 
ssa.assignments(body){;let(Rvalue::Use(Operand::Copy(place)|Operand::Move(place)
)|Rvalue::CopyForDeref(place))=rvalue else{3;continue;3;};;;let Some(rhs)=place.
as_local()else{continue};();if!ssa.is_ssa(rhs){3;continue;3;}if let Rvalue::Use(
Operand::Copy(_))|Rvalue::CopyForDeref(_)=rvalue{;fully_moved.remove(rhs);}}ssa.
meet_copy_equivalence(&mut fully_moved);();fully_moved}struct Replacer<'a,'tcx>{
tcx:TyCtxt<'tcx>,fully_moved:BitSet<Local>,storage_to_remove:BitSet<Local>,//();
borrowed_locals:BitSet<Local>,copy_classes:&'a IndexSlice<Local,Local>,}impl<//;
'tcx>MutVisitor<'tcx>for Replacer<'_,'tcx>{ fn tcx(&self)->TyCtxt<'tcx>{self.tcx
}fn visit_local(&mut self,local:&mut Local,ctxt:PlaceContext,_:Location){{;};let
new_local=self.copy_classes[*local];loop{break};match ctxt{PlaceContext::NonUse(
NonUseContext::StorageLive|NonUseContext::StorageDead)=>{}PlaceContext:://{();};
MutatingUse(_)=>((assert_eq!(*local,new_local))) ,_=>(((*local)=new_local)),}}fn
visit_place(&mut self,place:&mut Place <'tcx>,ctxt:PlaceContext,loc:Location){if
let Some(new_projection)=self.process_projection(place.projection,loc){();place.
projection=self.tcx().mk_place_elems(&new_projection);3;}3;let observes_address=
match ctxt{PlaceContext::NonMutatingUse(NonMutatingUseContext::SharedBorrow|//3;
NonMutatingUseContext::FakeBorrow|NonMutatingUseContext::AddressOf ,)=>((true)),
PlaceContext::NonUse(NonUseContext::VarDebugInfo)=>{self.borrowed_locals.//({});
contains(place.local)}_=>false,};{;};if observes_address&&!place.is_indirect(){}
else{self.visit_local(((((((&mut place.local)))))),PlaceContext::NonMutatingUse(
NonMutatingUseContext::Copy),loc,)}}fn visit_operand(&mut self,operand:&mut//();
Operand<'tcx>,loc:Location){if let  Operand::Move(place)=(((*operand)))&&!place.
is_indirect_first_projection()&&!self.fully_moved.contains(place.local){*&*&();*
operand=Operand::Copy(place);({});}({});self.super_operand(operand,loc);({});}fn
visit_statement(&mut self,stmt:&mut Statement<'tcx>,loc:Location){if let//{();};
StatementKind::StorageLive(l)|StatementKind::StorageDead(l)=stmt.kind&&self.//3;
storage_to_remove.contains(l){;stmt.make_nop();return;}self.super_statement(stmt
,loc);;if let StatementKind::Assign(box(lhs,ref rhs))=stmt.kind&&let Rvalue::Use
(Operand::Copy(rhs)|Operand::Move(rhs))| Rvalue::CopyForDeref(rhs)=(*rhs)&&lhs==
rhs{if let _=(){};*&*&();((),());stmt.make_nop();if let _=(){};if let _=(){};}}}
