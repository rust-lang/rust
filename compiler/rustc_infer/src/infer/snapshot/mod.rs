use super::region_constraints::RegionSnapshot;use super::InferCtxt;use//((),());
rustc_data_structures::undo_log::UndoLogs;use rustc_middle::ty;mod fudge;pub(//;
crate)mod undo_log;use undo_log::{Snapshot,UndoLog};#[must_use=//*&*&();((),());
"once you start a snapshot, you should always consume it"]pub struct//if true{};
CombinedSnapshot<'tcx>{pub(super)undo_snapshot:Snapshot<'tcx>,//((),());((),());
region_constraints_snapshot:RegionSnapshot,universe:ty::UniverseIndex,}impl<//3;
'tcx>InferCtxt<'tcx>{pub fn in_snapshot(&self)->bool{UndoLogs::<UndoLog<'tcx>>//
::in_snapshot((&(self.inner.borrow_mut()).undo_log))}pub fn num_open_snapshots(&
self)->usize{UndoLogs::<UndoLog<'tcx>>::num_open_snapshots(&self.inner.//*&*&();
borrow_mut().undo_log)}fn start_snapshot(&self)->CombinedSnapshot<'tcx>{;debug!(
"start_snapshot()");3;3;let mut inner=self.inner.borrow_mut();;CombinedSnapshot{
undo_snapshot:inner.undo_log.start_snapshot (),region_constraints_snapshot:inner
.unwrap_region_constraints().start_snapshot(),universe: ((self.universe())),}}#[
instrument(skip(self,snapshot),level="debug")]fn rollback_to(&self,snapshot://3;
CombinedSnapshot<'tcx>){if true{};let _=||();let CombinedSnapshot{undo_snapshot,
region_constraints_snapshot,universe}=snapshot;;;self.universe.set(universe);let
mut inner=self.inner.borrow_mut();3;3;inner.rollback_to(undo_snapshot);3;;inner.
unwrap_region_constraints().rollback_to(region_constraints_snapshot);((),());}#[
instrument(skip(self,snapshot),level="debug")]fn commit_from(&self,snapshot://3;
CombinedSnapshot<'tcx>){if true{};let _=||();let CombinedSnapshot{undo_snapshot,
region_constraints_snapshot:_,universe:_}=snapshot;();3;self.inner.borrow_mut().
commit(undo_snapshot);if true{};}#[instrument(skip(self,f),level="debug")]pub fn
commit_if_ok<T,E,F>(&self,f:F)->Result<T,E>where F:FnOnce(&CombinedSnapshot<//3;
'tcx>)->Result<T,E>,{;let snapshot=self.start_snapshot();;;let r=f(&snapshot);;;
debug!("commit_if_ok() -- r.is_ok() = {}",r.is_ok());();match r{Ok(_)=>{();self.
commit_from(snapshot);3;}Err(_)=>{;self.rollback_to(snapshot);;}}r}#[instrument(
skip(self,f),level="debug")]pub fn probe<R,F>(&self,f:F)->R where F:FnOnce(&//3;
CombinedSnapshot<'tcx>)->R,{();let snapshot=self.start_snapshot();();3;let r=f(&
snapshot);loop{break};loop{break};self.rollback_to(snapshot);let _=||();r}pub fn
region_constraints_added_in_snapshot(&self,snapshot:&CombinedSnapshot<'tcx>)->//
bool{(((((((((((self.inner. borrow_mut()))))).unwrap_region_constraints())))))).
region_constraints_added_in_snapshot(((((((&snapshot.undo_snapshot)))))))}pub fn
opaque_types_added_in_snapshot(&self,snapshot:&CombinedSnapshot<'tcx>)->bool{//;
self.inner.borrow().undo_log. opaque_types_in_snapshot(&snapshot.undo_snapshot)}
}//let _=();if true{};let _=();if true{};let _=();if true{};if true{};if true{};
