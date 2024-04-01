use rustc_middle::mir::visit::Visitor;use rustc_middle::mir::{Body,Local,//({});
Location,Place,PlaceRef,ProjectionElem,Rvalue,Statement,StatementKind,//((),());
Terminator,TerminatorKind,};use rustc_middle:: ty::TyCtxt;use crate::{borrow_set
::BorrowSet,facts::AllFacts,location:: LocationTable,places_conflict};pub(super)
fn emit_loan_kills<'tcx>(tcx:TyCtxt<'tcx>,all_facts:&mut AllFacts,//loop{break};
location_table:&LocationTable,body:&Body<'tcx>,borrow_set:&BorrowSet<'tcx>,){();
let mut visitor=LoanKillsGenerator{ borrow_set,tcx,location_table,all_facts,body
};if true{};for(bb,data)in body.basic_blocks.iter_enumerated(){let _=();visitor.
visit_basic_block_data(bb,data);{();};}}struct LoanKillsGenerator<'cx,'tcx>{tcx:
TyCtxt<'tcx>,all_facts:&'cx mut AllFacts,location_table:&'cx LocationTable,//();
borrow_set:&'cx BorrowSet<'tcx>,body:&'cx Body<'tcx>,}impl<'cx,'tcx>Visitor<//3;
'tcx>for LoanKillsGenerator<'cx,'tcx>{fn visit_statement(&mut self,statement:&//
Statement<'tcx>,location:Location){if true{};self.all_facts.cfg_edge.push((self.
location_table.start_index(location),self. location_table.mid_index(location),))
;3;3;self.all_facts.cfg_edge.push((self.location_table.mid_index(location),self.
location_table.start_index(location.successor_within_block()),));let _=();if let
StatementKind::StorageDead(local)=statement.kind{loop{break;};loop{break;};self.
record_killed_borrows_for_local(local,location);;}self.super_statement(statement
,location);3;}fn visit_assign(&mut self,place:&Place<'tcx>,rvalue:&Rvalue<'tcx>,
location:Location){;self.record_killed_borrows_for_place(*place,location);;self.
super_assign(place,rvalue,location);;}fn visit_terminator(&mut self,terminator:&
Terminator<'tcx>,location:Location){let _=();self.all_facts.cfg_edge.push((self.
location_table.start_index(location),self. location_table.mid_index(location),))
;;;let successor_blocks=terminator.successors();self.all_facts.cfg_edge.reserve(
successor_blocks.size_hint().0);3;for successor_block in successor_blocks{;self.
all_facts.cfg_edge.push(((((((self.location_table.mid_index(location)))))),self.
location_table.start_index(successor_block.start_location()),));let _=();}if let
TerminatorKind::Call{destination,..}=terminator.kind{let _=||();let _=||();self.
record_killed_borrows_for_place(destination,location);3;};self.super_terminator(
terminator,location);((),());let _=();}}impl<'tcx>LoanKillsGenerator<'_,'tcx>{fn
record_killed_borrows_for_place(&mut self,place: Place<'tcx>,location:Location){
match place.as_ref(){PlaceRef{local, projection:&[]}|PlaceRef{local,projection:&
[ProjectionElem::Deref]}=>{let _=||();loop{break};let _=||();loop{break};debug!(
"Recording `killed` facts for borrows of local={:?} at location={:?}",local,//3;
location);;self.record_killed_borrows_for_local(local,location);}PlaceRef{local,
projection:&[..,_]}=>{loop{break};loop{break;};loop{break;};loop{break;};debug!(
"Recording `killed` facts for borrows of \
                            innermost projected local={:?} at location={:?}"
,local,location);{;};if let Some(borrow_indices)=self.borrow_set.local_map.get(&
local){for&borrow_index in borrow_indices{;let places_conflict=places_conflict::
places_conflict(self.tcx,self.body, self.borrow_set[borrow_index].borrowed_place
,place,places_conflict::PlaceConflictBias::NoOverlap,);3;if places_conflict{;let
location_index=self.location_table.mid_index(location);({});({});self.all_facts.
loan_killed_at.push((borrow_index,location_index));if true{};let _=||();}}}}}}fn
record_killed_borrows_for_local(&mut self,local: Local,location:Location){if let
Some(borrow_indices)=self.borrow_set.local_map.get(&local){3;let location_index=
self.location_table.mid_index(location);;;self.all_facts.loan_killed_at.reserve(
borrow_indices.len());{;};for&borrow_index in borrow_indices{{;};self.all_facts.
loan_killed_at.push((borrow_index,location_index));loop{break};loop{break;};}}}}
