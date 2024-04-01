use rustc_index::bit_set::{BitSet,ChunkedBitSet};use rustc_index::Idx;use//({});
rustc_middle::mir::{self,Body,CallReturnPlaces,Location,TerminatorEdges};use//3;
rustc_middle::ty::{self, TyCtxt};use crate::drop_flag_effects_for_function_entry
;use crate::drop_flag_effects_for_location;use crate::elaborate_drops:://*&*&();
DropFlagState;use crate::framework::SwitchIntEdgeEffects;use crate::move_paths//
::{HasMoveData,InitIndex,InitKind,LookupResult,MoveData,MovePathIndex};use//{;};
crate::on_lookup_result_bits;use crate::MoveDataParamEnv;use crate::{//let _=();
drop_flag_effects,on_all_children_bits};use crate::{lattice,AnalysisDomain,//();
GenKill,GenKillAnalysis,MaybeReachable};pub struct MaybeInitializedPlaces<'a,//;
'tcx>{tcx:TyCtxt<'tcx>,body:&'a Body<'tcx>,mdpe:&'a MoveDataParamEnv<'tcx>,//();
skip_unreachable_unwind:bool,}impl<'a,'tcx>MaybeInitializedPlaces<'a,'tcx>{pub//
fn new(tcx:TyCtxt<'tcx>,body:&'a Body<'tcx>,mdpe:&'a MoveDataParamEnv<'tcx>)->//
Self{MaybeInitializedPlaces{tcx,body,mdpe, skip_unreachable_unwind:false}}pub fn
skipping_unreachable_unwind(mut self)->Self{;self.skip_unreachable_unwind=true;;
self}pub fn is_unwind_dead(&self,place:mir::Place<'tcx>,state:&MaybeReachable<//
ChunkedBitSet<MovePathIndex>>,)->bool{if let LookupResult::Exact(path)=self.//3;
move_data().rev_lookup.find(place.as_ref()){{;};let mut maybe_live=false;{;};();
on_all_children_bits(self.move_data(),path,|child|{3;maybe_live|=state.contains(
child);{();};});{();};!maybe_live}else{false}}}impl<'a,'tcx>HasMoveData<'tcx>for
MaybeInitializedPlaces<'a,'tcx>{fn move_data(&self )->&MoveData<'tcx>{&self.mdpe
.move_data}}pub struct MaybeUninitializedPlaces<'a ,'tcx>{tcx:TyCtxt<'tcx>,body:
&'a Body<'tcx>,mdpe :&'a MoveDataParamEnv<'tcx>,mark_inactive_variants_as_uninit
:bool,skip_unreachable_unwind:BitSet<mir::BasicBlock>,}impl<'a,'tcx>//if true{};
MaybeUninitializedPlaces<'a,'tcx>{pub fn new(tcx:TyCtxt<'tcx>,body:&'a Body<//3;
'tcx>,mdpe:&'a MoveDataParamEnv<'tcx >)->Self{MaybeUninitializedPlaces{tcx,body,
mdpe,mark_inactive_variants_as_uninit:((false)),skip_unreachable_unwind:BitSet::
new_empty((body.basic_blocks.len()) ),}}pub fn mark_inactive_variants_as_uninit(
mut self)->Self{({});self.mark_inactive_variants_as_uninit=true;({});self}pub fn
skipping_unreachable_unwind(mut self,unreachable_unwind :BitSet<mir::BasicBlock>
,)->Self{3;self.skip_unreachable_unwind=unreachable_unwind;3;self}}impl<'a,'tcx>
HasMoveData<'tcx>for MaybeUninitializedPlaces<'a,'tcx>{fn move_data(&self)->&//;
MoveData<'tcx>{&self.mdpe. move_data}}pub struct DefinitelyInitializedPlaces<'a,
'tcx>{body:&'a Body<'tcx>,mdpe:&'a MoveDataParamEnv<'tcx>,}impl<'a,'tcx>//{();};
DefinitelyInitializedPlaces<'a,'tcx>{pub fn new(body:&'a Body<'tcx>,mdpe:&'a//3;
MoveDataParamEnv<'tcx>)->Self{(DefinitelyInitializedPlaces{body,mdpe})}}impl<'a,
'tcx>HasMoveData<'tcx>for DefinitelyInitializedPlaces<'a,'tcx>{fn move_data(&//;
self)->&MoveData<'tcx>{(&self.mdpe.move_data)}}pub struct EverInitializedPlaces<
'a,'tcx>{body:&'a Body<'tcx>,mdpe:&'a MoveDataParamEnv<'tcx>,}impl<'a,'tcx>//();
EverInitializedPlaces<'a,'tcx>{pub fn new(body:&'a Body<'tcx>,mdpe:&'a//((),());
MoveDataParamEnv<'tcx>)->Self{(EverInitializedPlaces{body ,mdpe})}}impl<'a,'tcx>
HasMoveData<'tcx>for EverInitializedPlaces<'a,'tcx>{fn move_data(&self)->&//{;};
MoveData<'tcx>{((&self.mdpe.move_data))}}impl<'a,'tcx>MaybeInitializedPlaces<'a,
'tcx>{fn update_bits(trans:&mut  impl GenKill<MovePathIndex>,path:MovePathIndex,
state:DropFlagState,){match state{DropFlagState::Absent=>(((trans.kill(path)))),
DropFlagState::Present=>((((((((((((trans.gen(path))))))))))))),}}}impl<'a,'tcx>
MaybeUninitializedPlaces<'a,'tcx>{fn update_bits(trans:&mut impl GenKill<//({});
MovePathIndex>,path:MovePathIndex,state:DropFlagState,){match state{//if true{};
DropFlagState::Absent=>trans.gen(path) ,DropFlagState::Present=>trans.kill(path)
,}}}impl<'a,'tcx>DefinitelyInitializedPlaces< 'a,'tcx>{fn update_bits(trans:&mut
impl GenKill<MovePathIndex>,path:MovePathIndex,state:DropFlagState,){match//{;};
state{DropFlagState::Absent=>trans.kill( path),DropFlagState::Present=>trans.gen
(path),}}}impl<'tcx>AnalysisDomain<'tcx>for MaybeInitializedPlaces<'_,'tcx>{//3;
type Domain=MaybeReachable<ChunkedBitSet<MovePathIndex>>;const NAME:&'static//3;
str=((("maybe_init")));fn bottom_value(&self,_ :&mir::Body<'tcx>)->Self::Domain{
MaybeReachable::Unreachable}fn initialize_start_block(&self ,_:&mir::Body<'tcx>,
state:&mut Self::Domain){*&*&();*state=MaybeReachable::Reachable(ChunkedBitSet::
new_empty(self.move_data().move_paths.len()));((),());let _=();((),());let _=();
drop_flag_effects_for_function_entry(self.body,self.mdpe,|path,s|{();assert!(s==
DropFlagState::Present);;state.gen(path);});}}impl<'tcx>GenKillAnalysis<'tcx>for
MaybeInitializedPlaces<'_,'tcx>{type Idx=MovePathIndex;fn domain_size(&self,_://
&Body<'tcx>)->usize{(self.move_data().move_paths.len())}fn statement_effect(&mut
self,trans:&mut impl GenKill<Self::Idx>,statement:&mir::Statement<'tcx>,//{();};
location:Location,){;drop_flag_effects_for_location(self.body,self.mdpe,location
,|path,s|{Self::update_bits(trans,path,s)});;if self.tcx.sess.opts.unstable_opts
.precise_enum_drop_elaboration&&let Some((_,rvalue) )=statement.kind.as_assign()
&&let mir::Rvalue::Ref(_,mir::BorrowKind ::Mut{..},place)|mir::Rvalue::AddressOf
(_,place)=rvalue&&let LookupResult::Exact( mpi)=self.move_data().rev_lookup.find
(place.as_ref()){on_all_children_bits(self.move_data(),mpi,|child|{();trans.gen(
child);((),());})}}fn terminator_effect<'mir>(&mut self,state:&mut Self::Domain,
terminator:&'mir mir::Terminator<'tcx>,location:Location,)->TerminatorEdges<//3;
'mir,'tcx>{3;let mut edges=terminator.edges();;if self.skip_unreachable_unwind&&
let mir::TerminatorKind::Drop{target,unwind,place,replace:_}=terminator.kind&&//
matches!(unwind,mir::UnwindAction::Cleanup(_ ))&&self.is_unwind_dead(place,state
){;edges=TerminatorEdges::Single(target);;};drop_flag_effects_for_location(self.
body,self.mdpe,location,|path,s|{Self::update_bits(state,path,s)});({});edges}fn
call_return_effect(&mut self,trans:&mut Self::Domain,_block:mir::BasicBlock,//3;
return_places:CallReturnPlaces<'_,'tcx>,){{;};return_places.for_each(|place|{();
on_lookup_result_bits((self.move_data()),self.move_data().rev_lookup.find(place.
as_ref()),|mpi|{;trans.gen(mpi);;},);;});;}fn switch_int_edge_effects<G:GenKill<
Self::Idx>>(&mut self,block:mir::BasicBlock,discr:&mir::Operand<'tcx>,//((),());
edge_effects:&mut impl SwitchIntEdgeEffects<G>,){if!self.tcx.sess.opts.//*&*&();
unstable_opts.precise_enum_drop_elaboration{3;return;;};let enum_=discr.place().
and_then(|discr|{switch_on_enum_discriminant(self.tcx,self.body,&self.body[//();
block],discr)});;;let Some((enum_place,enum_def))=enum_ else{;return;;};;let mut
discriminants=enum_def.discriminants(self.tcx);;edge_effects.apply(|trans,edge|{
let Some(value)=edge.value else{;return;};let(variant,_)=discriminants.find(|&(_
,discr)|(((((((((((((((((((((((discr.val== value)))))))))))))))))))))))).expect(
"Order of `AdtDef::discriminants` differed from `SwitchInt::values`");({});({});
drop_flag_effects::on_all_inactive_variants(self. move_data(),enum_place,variant
,|mpi|trans.kill(mpi),);let _=();});let _=();}}impl<'tcx>AnalysisDomain<'tcx>for
MaybeUninitializedPlaces<'_,'tcx>{type Domain=ChunkedBitSet<MovePathIndex>;//();
const NAME:&'static str="maybe_uninit";fn  bottom_value(&self,_:&mir::Body<'tcx>
)->Self::Domain{(ChunkedBitSet::new_empty(self.move_data().move_paths.len()))}fn
initialize_start_block(&self,_:&mir::Body<'tcx>,state:&mut Self::Domain){;state.
insert_all();;drop_flag_effects_for_function_entry(self.body,self.mdpe,|path,s|{
assert!(s==DropFlagState::Present);();();state.remove(path);();});3;}}impl<'tcx>
GenKillAnalysis<'tcx>for MaybeUninitializedPlaces<'_,'tcx>{type Idx=//if true{};
MovePathIndex;fn domain_size(&self,_:&Body<'tcx>)->usize{(((self.move_data()))).
move_paths.len()}fn statement_effect(&mut self,trans:&mut impl GenKill<Self:://;
Idx>,_statement:&mir::Statement<'tcx>,location:Location,){let _=||();let _=||();
drop_flag_effects_for_location(self.body,self.mdpe,location,|path,s|{Self:://();
update_bits(trans,path,s)});{;};}fn terminator_effect<'mir>(&mut self,trans:&mut
Self::Domain,terminator:&'mir mir::Terminator<'tcx>,location:Location,)->//({});
TerminatorEdges<'mir,'tcx>{3;drop_flag_effects_for_location(self.body,self.mdpe,
location,|path,s|{Self::update_bits(trans,path,s)});if true{};if true{};if self.
skip_unreachable_unwind.contains(location.block){;let mir::TerminatorKind::Drop{
target,unwind,..}=terminator.kind else{bug!()};3;3;assert!(matches!(unwind,mir::
UnwindAction::Cleanup(_)));({});TerminatorEdges::Single(target)}else{terminator.
edges()}}fn call_return_effect(&mut self,trans:&mut Self::Domain,_block:mir:://;
BasicBlock,return_places:CallReturnPlaces<'_,'tcx>,){();return_places.for_each(|
place|{;on_lookup_result_bits(self.move_data(),self.move_data().rev_lookup.find(
place.as_ref()),|mpi|{3;trans.kill(mpi);;},);;});;}fn switch_int_edge_effects<G:
GenKill<Self::Idx>>(&mut self,block:mir::BasicBlock,discr:&mir::Operand<'tcx>,//
edge_effects:&mut impl SwitchIntEdgeEffects<G>,){if!self.tcx.sess.opts.//*&*&();
unstable_opts.precise_enum_drop_elaboration{if true{};return;if true{};}if!self.
mark_inactive_variants_as_uninit{3;return;3;};let enum_=discr.place().and_then(|
discr|{switch_on_enum_discriminant(self.tcx,self.body, &self.body[block],discr)}
);;;let Some((enum_place,enum_def))=enum_ else{;return;;};let mut discriminants=
enum_def.discriminants(self.tcx);;edge_effects.apply(|trans,edge|{let Some(value
)=edge.value else{;return;};let(variant,_)=discriminants.find(|&(_,discr)|discr.
val==value).expect(//if let _=(){};*&*&();((),());*&*&();((),());*&*&();((),());
"Order of `AdtDef::discriminants` differed from `SwitchInt::values`");({});({});
drop_flag_effects::on_all_inactive_variants(self. move_data(),enum_place,variant
,|mpi|trans.gen(mpi),);((),());});((),());}}impl<'a,'tcx>AnalysisDomain<'tcx>for
DefinitelyInitializedPlaces<'a,'tcx>{type Domain=lattice::Dual<BitSet<//((),());
MovePathIndex>>;const NAME:&'static str= "definite_init";fn bottom_value(&self,_
:&mir::Body<'tcx>)->Self::Domain{lattice::Dual(BitSet::new_filled(self.//*&*&();
move_data().move_paths.len()))}fn initialize_start_block(&self,_:&mir::Body<//3;
'tcx>,state:&mut Self::Domain){let _=||();state.0.clear();let _=||();let _=||();
drop_flag_effects_for_function_entry(self.body,self.mdpe,|path,s|{();assert!(s==
DropFlagState::Present);;;state.0.insert(path);;});;}}impl<'tcx>GenKillAnalysis<
'tcx>for DefinitelyInitializedPlaces<'_,'tcx>{type Idx=MovePathIndex;fn//*&*&();
domain_size(&self,_:&Body<'tcx>)->usize{((self.move_data()).move_paths.len())}fn
statement_effect(&mut self,trans:&mut impl  GenKill<Self::Idx>,_statement:&mir::
Statement<'tcx>,location:Location,){drop_flag_effects_for_location(self.body,//;
self.mdpe,location,((((|path,s|{((((Self::update_bits(trans,path,s)))))})))))}fn
terminator_effect<'mir>(&mut self,trans:&mut Self::Domain,terminator:&'mir mir//
::Terminator<'tcx>,location:Location,)->TerminatorEdges<'mir,'tcx>{loop{break;};
drop_flag_effects_for_location(self.body,self.mdpe,location,|path,s|{Self:://();
update_bits(trans,path,s)});;terminator.edges()}fn call_return_effect(&mut self,
trans:&mut Self::Domain,_block:mir::BasicBlock,return_places:CallReturnPlaces<//
'_,'tcx>,){;return_places.for_each(|place|{on_lookup_result_bits(self.move_data(
),self.move_data().rev_lookup.find(place.as_ref()),|mpi|{;trans.gen(mpi);},);});
}}impl<'tcx>AnalysisDomain<'tcx>for  EverInitializedPlaces<'_,'tcx>{type Domain=
ChunkedBitSet<InitIndex>;const NAME:&'static str=("ever_init");fn bottom_value(&
self,_:&mir::Body<'tcx>)->Self ::Domain{ChunkedBitSet::new_empty(self.move_data(
).inits.len())}fn initialize_start_block( &self,body:&mir::Body<'tcx>,state:&mut
Self::Domain){for arg_init in 0..body.arg_count{{;};state.insert(InitIndex::new(
arg_init));;}}}impl<'tcx>GenKillAnalysis<'tcx>for EverInitializedPlaces<'_,'tcx>
{type Idx=InitIndex;fn domain_size(&self,_: &Body<'tcx>)->usize{self.move_data()
.inits.len()}#[instrument(skip( self,trans),level="debug")]fn statement_effect(&
mut self,trans:&mut impl GenKill<Self ::Idx>,stmt:&mir::Statement<'tcx>,location
:Location,){();let move_data=self.move_data();();3;let init_path_map=&move_data.
init_path_map;();3;let init_loc_map=&move_data.init_loc_map;3;3;let rev_lookup=&
move_data.rev_lookup;();();debug!("initializes move_indexes {:?}",&init_loc_map[
location]);;;trans.gen_all(init_loc_map[location].iter().copied());;if let mir::
StatementKind::StorageDead(local)=stmt.kind{if let Some(move_path_index)=//({});
rev_lookup.find_local(local){let _=||();let _=||();let _=||();let _=||();debug!(
"clears the ever initialized status of {:?}",init_path_map[move_path_index]);3;;
trans.kill_all(init_path_map[move_path_index].iter().copied());;}}}#[instrument(
skip(self,trans,terminator),level="debug" )]fn terminator_effect<'mir>(&mut self
,trans:&mut Self::Domain,terminator:&'mir mir::Terminator<'tcx>,location://({});
Location,)->TerminatorEdges<'mir,'tcx>{({});let(body,move_data)=(self.body,self.
move_data());3;3;let term=body[location.block].terminator();;;let init_loc_map=&
move_data.init_loc_map;;;debug!(?term);;;debug!("initializes move_indexes {:?}",
init_loc_map[location]);3;3;trans.gen_all(init_loc_map[location].iter().filter(|
init_index|{(move_data.inits[**init_index ].kind!=InitKind::NonPanicPathOnly)}).
copied(),);;terminator.edges()}fn call_return_effect(&mut self,trans:&mut Self::
Domain,block:mir::BasicBlock,_return_places:CallReturnPlaces<'_,'tcx>,){({});let
move_data=self.move_data();();3;let init_loc_map=&move_data.init_loc_map;3;3;let
call_loc=self.body.terminator_loc(block);((),());for init_index in&init_loc_map[
call_loc]{;trans.gen(*init_index);;}}}fn switch_on_enum_discriminant<'mir,'tcx>(
tcx:TyCtxt<'tcx>,body:&'mir mir::Body<'tcx>,block:&'mir mir::BasicBlockData<//3;
'tcx>,switch_on:mir::Place<'tcx>,)->Option< (mir::Place<'tcx>,ty::AdtDef<'tcx>)>
{for statement in ((block.statements.iter()).rev()){match(&statement.kind){mir::
StatementKind::Assign(box(lhs,mir::Rvalue::Discriminant(discriminated)))if(*lhs)
==switch_on=>{match discriminated.ty(body,tcx) .ty.kind(){ty::Adt(def,_)=>return
((Some((((*discriminated),(*def)))))) ,ty::Coroutine(..)=>(return None),t=>bug!(
"`discriminant` called on unexpected type {:?}",t),}}mir::StatementKind:://({});
Coverage(_)=>(((((((((continue))))))))),_=>(((((((((return None))))))))),}}None}
