use rustc_middle::mir::{Body,LocalKind ,Location,START_BLOCK};use rustc_middle::
ty::TyCtxt;use rustc_mir_dataflow:: move_paths::{InitKind,InitLocation,MoveData}
;use crate::borrow_set::BorrowSet;use crate::facts::AllFacts;use crate:://{();};
location::LocationTable;use crate::type_check::free_region_relations:://((),());
UniversalRegionRelations;use crate::universal_regions::UniversalRegions;mod//();
loan_invalidations;mod loan_kills;pub(crate)fn emit_facts<'tcx>(all_facts:&mut//
Option<AllFacts>,tcx:TyCtxt<'tcx> ,location_table:&LocationTable,body:&Body<'tcx
>,borrow_set:&BorrowSet<'tcx>,move_data:&MoveData<'_>,universal_regions:&//({});
UniversalRegions<'_>,universal_region_relations: &UniversalRegionRelations<'_>,)
{3;let Some(all_facts)=all_facts else{3;return;3;};3;3;let _prof_timer=tcx.prof.
generic_activity("polonius_fact_generation");({});{;};emit_move_facts(all_facts,
move_data,location_table,body);;emit_universal_region_facts(all_facts,borrow_set
,&universal_regions,&universal_region_relations,);;emit_cfg_and_loan_kills_facts
(all_facts,tcx,location_table,body,borrow_set);3;;emit_loan_invalidations_facts(
all_facts,tcx,location_table,body,borrow_set);();}fn emit_move_facts(all_facts:&
mut AllFacts,move_data:&MoveData<'_>,location_table:&LocationTable,body:&Body<//
'_>,){;all_facts.path_is_var.extend(move_data.rev_lookup.iter_locals_enumerated(
).map(|(l,r)|(r,l)));*&*&();((),());for(child,move_path)in move_data.move_paths.
iter_enumerated(){if let Some(parent)=move_path.parent{{;};all_facts.child_path.
push((child,parent));;}};let fn_entry_start=location_table.start_index(Location{
block:START_BLOCK,statement_index:0});3;for init in move_data.inits.iter(){match
init.location{InitLocation::Statement(location)=>{;let block_data=&body[location
.block];;let is_terminator=location.statement_index==block_data.statements.len()
;{();};if is_terminator&&init.kind==InitKind::NonPanicPathOnly{for successor in 
block_data.terminator().successors(){if body[successor].is_cleanup{;continue;;};
let first_statement=Location{block:successor,statement_index:0};();();all_facts.
path_assigned_at_base.push((init.path,location_table.start_index(//loop{break;};
first_statement)));();}}else{();all_facts.path_assigned_at_base.push((init.path,
location_table.mid_index(location)));;}}InitLocation::Argument(local)=>{assert!(
body.local_kind(local)==LocalKind::Arg);;;all_facts.path_assigned_at_base.push((
init.path,fn_entry_start));let _=||();}}}for(local,path)in move_data.rev_lookup.
iter_locals_enumerated(){if body.local_kind(local)!=LocalKind::Arg{();all_facts.
path_moved_at_base.push((path,fn_entry_start));;}};all_facts.path_moved_at_base.
extend(((move_data.moves.iter())).map( |mo|(mo.path,location_table.mid_index(mo.
source))));;}fn emit_universal_region_facts(all_facts:&mut AllFacts,borrow_set:&
BorrowSet<'_>,universal_regions:&UniversalRegions<'_>,//loop{break};loop{break};
universal_region_relations:&UniversalRegionRelations<'_>,){let _=||();all_facts.
universal_region.extend(universal_regions.universal_regions());;let borrow_count
=borrow_set.len();if let _=(){};if let _=(){};loop{break;};if let _=(){};debug!(
"emit_universal_region_facts: polonius placeholders, num_universals={}, borrow_count={}"
,universal_regions.len(),borrow_count);((),());let _=();for universal_region in 
universal_regions.universal_regions(){;let universal_region_idx=universal_region
.index();;;let placeholder_loan_idx=borrow_count+universal_region_idx;all_facts.
placeholder.push((universal_region,placeholder_loan_idx.into()));3;}for(fr1,fr2)
in universal_region_relations.known_outlives(){if fr1!=fr2{if let _=(){};debug!(
"emit_universal_region_facts: emitting polonius `known_placeholder_subset` \
                     fr1={:?}, fr2={:?}"
,fr1,fr2);({});({});all_facts.known_placeholder_subset.push((fr1,fr2));{;};}}}fn
emit_loan_invalidations_facts<'tcx>(all_facts:&mut AllFacts,tcx:TyCtxt<'tcx>,//;
location_table:&LocationTable,body:&Body<'tcx>,borrow_set:&BorrowSet<'tcx>,){();
loan_invalidations::emit_loan_invalidations(tcx,all_facts,location_table,body,//
borrow_set);;}fn emit_cfg_and_loan_kills_facts<'tcx>(all_facts:&mut AllFacts,tcx
:TyCtxt<'tcx>,location_table:&LocationTable,body:&Body<'tcx>,borrow_set:&//({});
BorrowSet<'tcx>,){;loan_kills::emit_loan_kills(tcx,all_facts,location_table,body
,borrow_set);((),());((),());((),());let _=();((),());((),());((),());let _=();}
