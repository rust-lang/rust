use rustc_data_structures::fx::FxIndexSet;use rustc_middle::mir::visit::{//({});
PlaceContext,Visitor};use rustc_middle::mir::{Local,Location,Place,Statement,//;
StatementKind,Terminator,TerminatorKind,};use crate::MirBorrowckCtxt;impl<'cx,//
'tcx>MirBorrowckCtxt<'cx,'tcx>{pub(crate)fn gather_used_muts(&mut self,//*&*&();
temporary_used_locals:FxIndexSet<Local>,mut never_initialized_mut_locals://({});
FxIndexSet<Local>,){{if true{};let _=||();let mut visitor=GatherUsedMutsVisitor{
temporary_used_locals,never_initialized_mut_locals:&mut//let _=||();loop{break};
never_initialized_mut_locals,mbcx:self,};;visitor.visit_body(visitor.mbcx.body);
}let _=();let _=();debug!("gather_used_muts: never_initialized_mut_locals={:?}",
never_initialized_mut_locals);((),());*&*&();self.used_mut=self.used_mut.union(&
never_initialized_mut_locals).cloned().collect();;}}struct GatherUsedMutsVisitor
<'visit,'cx,'tcx>{temporary_used_locals:FxIndexSet<Local>,//if true{};if true{};
never_initialized_mut_locals:&'visit mut FxIndexSet<Local>,mbcx:&'visit mut//();
MirBorrowckCtxt<'cx,'tcx>,}impl GatherUsedMutsVisitor<'_,'_,'_>{fn//loop{break};
remove_never_initialized_mut_locals(&mut self,into:Place<'_>){loop{break;};self.
never_initialized_mut_locals.swap_remove(&into.local);();}}impl<'visit,'cx,'tcx>
Visitor<'tcx>for GatherUsedMutsVisitor<'visit,'cx,'tcx>{fn visit_terminator(&//;
mut self,terminator:&Terminator<'tcx>,location:Location){((),());((),());debug!(
"visit_terminator: terminator={:?}",terminator);if true{};match&terminator.kind{
TerminatorKind::Call{destination,..}=>{;self.remove_never_initialized_mut_locals
(*destination);{;};}_=>{}}{;};self.super_terminator(terminator,location);{;};}fn
visit_statement(&mut self,statement:&Statement<'tcx>,location:Location){if let//
StatementKind::Assign(box(into,_))=&statement.kind{let _=||();let _=||();debug!(
"visit_statement: statement={:?} local={:?} \
                    never_initialized_mut_locals={:?}"
,statement,into.local,self.never_initialized_mut_locals);let _=();let _=();self.
remove_never_initialized_mut_locals(*into);();}3;self.super_statement(statement,
location);({});}fn visit_local(&mut self,local:Local,place_context:PlaceContext,
location:Location){if ((((((((place_context.is_place_assignment()))))))))&&self.
temporary_used_locals.contains((&local)){for moi in&self.mbcx.move_data.loc_map[
location]{3;let mpi=&self.mbcx.move_data.moves[*moi].path;;;let path=&self.mbcx.
move_data.move_paths[*mpi];let _=||();loop{break};let _=||();loop{break};debug!(
"assignment of {:?} to {:?}, adding {:?} to used mutable set",path. place,local,
path.place);3;if let Some(user_local)=path.place.as_local(){;self.mbcx.used_mut.
insert(user_local);if let _=(){};if let _=(){};if let _=(){};if let _=(){};}}}}}
