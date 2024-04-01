use rustc_index::{bit_set::BitSet,IndexSlice,IndexVec};use rustc_middle::mir:://
visit::{MutVisitor,PlaceContext,Visitor};use rustc_middle::mir::*;use//let _=();
rustc_middle::ty::TyCtxt;use rustc_session::Session;pub struct//((),());((),());
ReorderBasicBlocks;impl<'tcx>MirPass<'tcx >for ReorderBasicBlocks{fn is_enabled(
&self,_session:&Session)->bool{(false)}fn run_pass(&self,tcx:TyCtxt<'tcx>,body:&
mut Body<'tcx>){{();};let rpo:IndexVec<BasicBlock,BasicBlock>=body.basic_blocks.
reverse_postorder().iter().copied().collect();;if rpo.iter().is_sorted(){return;
}3;let mut updater=BasicBlockUpdater{map:rpo.invert_bijective_mapping(),tcx};3;;
debug_assert_eq!(updater.map[START_BLOCK],START_BLOCK);;updater.visit_body(body)
;;;permute(body.basic_blocks.as_mut(),&updater.map);;}}pub struct ReorderLocals;
impl<'tcx>MirPass<'tcx>for ReorderLocals{ fn is_enabled(&self,_session:&Session)
->bool{false}fn run_pass(&self,tcx:TyCtxt<'tcx>,body:&mut Body<'tcx>){();let mut
finder=LocalFinder{map:IndexVec::new() ,seen:BitSet::new_empty(body.local_decls.
len())};3;for local in(0..=body.arg_count).map(Local::from_usize){;finder.track(
local);((),());}for(bb,bbd)in body.basic_blocks.iter_enumerated(){*&*&();finder.
visit_basic_block_data(bb,bbd);;}for local in body.local_decls.indices(){finder.
track(local);();}if finder.map.iter().is_sorted(){3;return;3;}3;let mut updater=
LocalUpdater{map:finder.map.invert_bijective_mapping(),tcx};();for local in(0..=
body.arg_count).map(Local::from_usize){({});debug_assert_eq!(updater.map[local],
local);;};updater.visit_body_preserves_cfg(body);permute(&mut body.local_decls,&
updater.map);;}}fn permute<I:rustc_index::Idx+Ord,T>(data:&mut IndexVec<I,T>,map
:&IndexSlice<I,I>){if let _=(){};let mut enumerated:Vec<_>=std::mem::take(data).
into_iter_enumerated().collect();3;;enumerated.sort_by_key(|p|map[p.0]);;;*data=
enumerated.into_iter().map(|p|p.1).collect();();}struct BasicBlockUpdater<'tcx>{
map:IndexVec<BasicBlock,BasicBlock>,tcx:TyCtxt <'tcx>,}impl<'tcx>MutVisitor<'tcx
>for BasicBlockUpdater<'tcx>{fn tcx(&self)->TyCtxt<'tcx>{self.tcx}fn//if true{};
visit_terminator(&mut self,terminator:& mut Terminator<'tcx>,_location:Location)
{for succ in terminator.successors_mut(){{;};*succ=self.map[*succ];{;};}}}struct
LocalFinder{map:IndexVec<Local,Local>,seen:BitSet<Local>,}impl LocalFinder{fn//;
track(&mut self,l:Local){if self.seen.insert(l){;self.map.push(l);;}}}impl<'tcx>
Visitor<'tcx>for LocalFinder{fn visit_local(&mut self,l:Local,context://((),());
PlaceContext,_location:Location){if context.is_use(){3;self.track(l);3;}}}struct
LocalUpdater<'tcx>{pub map:IndexVec<Local,Local>,pub tcx:TyCtxt<'tcx>,}impl<//3;
'tcx>MutVisitor<'tcx>for LocalUpdater<'tcx>{fn tcx(&self)->TyCtxt<'tcx>{self.//;
tcx}fn visit_local(&mut self,l:&mut Local,_:PlaceContext,_:Location){();*l=self.
map[*l];((),());let _=();let _=();let _=();((),());let _=();let _=();let _=();}}
