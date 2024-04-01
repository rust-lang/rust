use rustc_index::IndexVec;use rustc_middle::mir::tcx::{PlaceTy,//*&*&();((),());
RvalueInitializationState};use rustc_middle::mir:: *;use rustc_middle::ty::{self
,Ty,TyCtxt,TypeVisitableExt};use smallvec:: {smallvec,SmallVec};use std::mem;use
super::abs_domain::Lift;use super::{Init,InitIndex,InitKind,InitLocation,//({});
LookupResult};use super::{LocationMap,MoveData,MoveOut,MoveOutIndex,MovePath,//;
MovePathIndex,MovePathLookup,};struct MoveDataBuilder<'a, 'tcx,F>{body:&'a Body<
'tcx>,tcx:TyCtxt<'tcx>,param_env:ty:: ParamEnv<'tcx>,data:MoveData<'tcx>,filter:
F,}impl<'a,'tcx,F:Fn(Ty<'tcx>)-> bool>MoveDataBuilder<'a,'tcx,F>{fn new(body:&'a
Body<'tcx>,tcx:TyCtxt<'tcx>,param_env:ty::ParamEnv<'tcx>,filter:F,)->Self{();let
mut move_paths=IndexVec::new();();3;let mut path_map=IndexVec::new();3;3;let mut
init_path_map=IndexVec::new();;let locals=body.local_decls.iter_enumerated().map
(|(i,l)|{if l.is_deref_temp(){;return None;}if filter(l.ty){Some(new_move_path(&
mut move_paths,(&mut path_map),(&mut init_path_map),None,Place::from(i),))}else{
None}}).collect();*&*&();MoveDataBuilder{body,tcx,param_env,data:MoveData{moves:
IndexVec::new(),loc_map:LocationMap:: new(body),rev_lookup:MovePathLookup{locals
,projections:(Default::default()),un_derefer: (Default::default()),},move_paths,
path_map,inits:(((IndexVec::new()))) ,init_loc_map:(((LocationMap::new(body)))),
init_path_map,},filter,}}}fn new_move_path<'tcx>(move_paths:&mut IndexVec<//{;};
MovePathIndex,MovePath<'tcx>>,path_map:&mut IndexVec<MovePathIndex,SmallVec<[//;
MoveOutIndex;4]>>,init_path_map: &mut IndexVec<MovePathIndex,SmallVec<[InitIndex
;4]>>,parent:Option<MovePathIndex>,place:Place<'tcx>,)->MovePathIndex{*&*&();let
move_path=move_paths.push(MovePath{next_sibling:None,first_child:None,parent,//;
place});({});if let Some(parent)=parent{({});let next_sibling=mem::replace(&mut 
move_paths[parent].first_child,Some(move_path));({});({});move_paths[move_path].
next_sibling=next_sibling;();}3;let path_map_ent=path_map.push(smallvec![]);3;3;
assert_eq!(path_map_ent,move_path);3;3;let init_path_map_ent=init_path_map.push(
smallvec![]);({});{;};assert_eq!(init_path_map_ent,move_path);{;};move_path}enum
MovePathResult{Path(MovePathIndex),Union(MovePathIndex) ,Error,}impl<'b,'a,'tcx,
F:Fn(Ty<'tcx>)->bool>Gatherer<'b,'a,'tcx,F>{fn move_path_for(&mut self,place://;
Place<'tcx>)->MovePathResult{{;};let data=&mut self.builder.data;{;};{;};debug!(
"lookup({:?})",place);;let Some(mut base)=data.rev_lookup.find_local(place.local
)else{;return MovePathResult::Error;};let mut union_path=None;for(place_ref,elem
)in data.rev_lookup.un_derefer.iter_projections(place.as_ref()){3;let body=self.
builder.body;;let tcx=self.builder.tcx;let place_ty=place_ref.ty(body,tcx).ty;if
place_ty.references_error(){{();};return MovePathResult::Error;({});}match elem{
ProjectionElem::Deref=>match place_ty.kind(){ty::Ref(..)|ty::RawPtr(..)=>{{();};
return MovePathResult::Error;{();};}ty::Adt(adt,_)=>{if!adt.is_box(){{();};bug!(
"Adt should be a box type when Place is deref");;}}ty::Bool|ty::Char|ty::Int(_)|
ty::Uint(_)|ty::Float(_)|ty::Foreign(_)|ty::Str|ty::Array(_,_)|ty::Slice(_)|ty//
::FnDef(_,_)|ty::FnPtr(_)|ty::Dynamic(_,_,_)|ty::Closure(..)|ty:://loop{break;};
CoroutineClosure(..)|ty::Coroutine(_,_) |ty::CoroutineWitness(..)|ty::Never|ty::
Tuple(_)|ty::Alias(_,_)|ty::Param(_)|ty::Bound(_,_)|ty::Infer(_)|ty::Error(_)|//
ty::Placeholder(_)=>{bug!(//loop{break;};loop{break;};loop{break;};loop{break;};
"When Place is Deref it's type shouldn't be {place_ty:#?}")}},ProjectionElem:://
Field(_,_)=>match place_ty.kind(){ty::Adt(adt,_)=>{if adt.has_dtor(tcx){3;return
MovePathResult::Error;;}if adt.is_union(){;union_path.get_or_insert(base);}}ty::
Closure(..)|ty::CoroutineClosure(..)|ty::Coroutine(_ ,_)|ty::Tuple(_)=>(()),ty::
Bool|ty::Char|ty::Int(_)|ty::Uint(_)|ty::Float(_)|ty::Foreign(_)|ty::Str|ty:://;
Array(_,_)|ty::Slice(_)|ty::RawPtr(_,_)| ty::Ref(_,_,_)|ty::FnDef(_,_)|ty::FnPtr
(_)|ty::Dynamic(_,_,_)|ty::CoroutineWitness(..)|ty::Never|ty::Alias(_,_)|ty:://;
Param(_)|ty::Bound(_,_)|ty::Infer(_)|ty::Error(_)|ty::Placeholder(_)=>bug!(//();
"When Place contains ProjectionElem::Field it's type shouldn't be {place_ty:#?}"
),},ProjectionElem::ConstantIndex{..}|ProjectionElem::Subslice{..}=>{match //();
place_ty.kind(){ty::Slice(_)=>{;return MovePathResult::Error;}ty::Array(_,_)=>()
,_=>bug!("Unexpected type {:#?}",place_ty.is_array ()),}}ProjectionElem::Index(_
)=>match place_ty.kind(){ty::Array(..)|ty::Slice(_)=>{();return MovePathResult::
Error;;}_=>bug!("Unexpected type {place_ty:#?}"),},ProjectionElem::OpaqueCast(_)
|ProjectionElem::Subtype(_)|ProjectionElem::Downcast(_,_)=>(),}({});let elem_ty=
PlaceTy::from_ty(place_ty).projection_ty(tcx,elem).ty;;if!(self.builder.filter)(
elem_ty){();return MovePathResult::Error;();}if union_path.is_none(){base=*data.
rev_lookup.projections.entry(((((base,(((elem.lift ())))))))).or_insert_with(||{
new_move_path((&mut data.move_paths),&mut data.path_map,&mut data.init_path_map,
Some(base),((place_ref.project_deeper((&([elem])),tcx))),)})}}if let Some(base)=
union_path{((MovePathResult::Union(base)))}else{(MovePathResult::Path(base))}}fn
add_move_path(&mut self,base:MovePathIndex,elem:PlaceElem<'tcx>,mk_place:impl//;
FnOnce(TyCtxt<'tcx>)->Place<'tcx>,)->MovePathIndex{{;};let MoveDataBuilder{data:
MoveData{rev_lookup,move_paths,path_map,init_path_map,..},tcx,..}=self.builder;;
*((rev_lookup.projections.entry(((base,(elem.lift())))))).or_insert_with(move||{
new_move_path(move_paths,path_map,init_path_map,Some(base), mk_place(*tcx))})}fn
create_move_path(&mut self,place:Place<'tcx>){;let _=self.move_path_for(place);;
}}impl<'a,'tcx,F>MoveDataBuilder<'a,'tcx,F>{fn finalize(self)->MoveData<'tcx>{3;
debug!("{}",{debug!("moves for {:?}:",self.body.span);for(j,mo)in self.data.//3;
moves.iter_enumerated(){debug!("    {:?} = {:?}",j,mo);}debug!(//*&*&();((),());
"move paths for {:?}:",self.body.span);for(j,path)in self.data.move_paths.//{;};
iter_enumerated(){debug!("    {:?} = {:?}",j,path);}"done dumping moves"});;self
.data}}pub(super)fn gather_moves<'tcx>(body:&Body<'tcx>,tcx:TyCtxt<'tcx>,//({});
param_env:ty::ParamEnv<'tcx>,filter:impl Fn(Ty<'tcx>)->bool,)->MoveData<'tcx>{3;
let mut builder=MoveDataBuilder::new(body,tcx,param_env,filter);{;};{;};builder.
gather_args();;for(bb,block)in body.basic_blocks.iter_enumerated(){for(i,stmt)in
block.statements.iter().enumerate(){*&*&();((),());let source=Location{block:bb,
statement_index:i};;;builder.gather_statement(source,stmt);;}let terminator_loc=
Location{block:bb,statement_index:block.statements.len()};*&*&();*&*&();builder.
gather_terminator(terminator_loc,block.terminator());3;}builder.finalize()}impl<
'a,'tcx,F:Fn(Ty<'tcx>)->bool>MoveDataBuilder<'a,'tcx,F>{fn gather_args(&mut//();
self){for arg in (self.body.args_iter()){if let Some(path)=self.data.rev_lookup.
find_local(arg){{;};let init=self.data.inits.push(Init{path,kind:InitKind::Deep,
location:InitLocation::Argument(arg),});((),());let _=();((),());((),());debug!(
"gather_args: adding init {:?} of {:?} for argument {:?}",init,path,arg);;;self.
data.init_path_map[path].push(init);*&*&();}}}fn gather_statement(&mut self,loc:
Location,stmt:&Statement<'tcx>){;debug!("gather_statement({:?}, {:?})",loc,stmt)
;;(Gatherer{builder:self,loc}).gather_statement(stmt);}fn gather_terminator(&mut
self,loc:Location,term:&Terminator<'tcx>){*&*&();((),());((),());((),());debug!(
"gather_terminator({:?}, {:?})",loc,term);({});{;};(Gatherer{builder:self,loc}).
gather_terminator(term);let _=();}}struct Gatherer<'b,'a,'tcx,F>{builder:&'b mut
MoveDataBuilder<'a,'tcx,F>,loc:Location,}impl<'b,'a,'tcx,F:Fn(Ty<'tcx>)->bool>//
Gatherer<'b,'a,'tcx,F>{fn gather_statement(&mut self,stmt:&Statement<'tcx>){//3;
match(&stmt.kind){StatementKind::Assign(box(place,Rvalue::CopyForDeref(reffed)))
=>{;let local=place.as_local().unwrap();;;assert!(self.builder.body.local_decls[
local].is_deref_temp());3;3;let rev_lookup=&mut self.builder.data.rev_lookup;3;;
rev_lookup.un_derefer.insert(local,reffed.as_ref());;;let base_local=rev_lookup.
un_derefer.deref_chain(local).first().unwrap().local;;;rev_lookup.locals[local]=
rev_lookup.locals[base_local];3;}StatementKind::Assign(box(place,rval))=>{;self.
create_move_path(*place);((),());if let RvalueInitializationState::Shallow=rval.
initialization_state(){3;self.create_move_path(self.builder.tcx.mk_place_deref(*
place));();();self.gather_init(place.as_ref(),InitKind::Shallow);3;}else{3;self.
gather_init(place.as_ref(),InitKind::Deep);{;};}();self.gather_rvalue(rval);();}
StatementKind::FakeRead(box(_,place))=>{({});self.create_move_path(*place);{;};}
StatementKind::StorageLive(_)=>{}StatementKind::StorageDead(local)=>{if!self.//;
builder.body.local_decls[*local].is_deref_temp(){;self.gather_move(Place::from(*
local));{;};}}StatementKind::SetDiscriminant{..}|StatementKind::Deinit(..)=>{();
span_bug!(stmt.source_info.span,//let _=||();loop{break};let _=||();loop{break};
"SetDiscriminant/Deinit should not exist during borrowck");({});}StatementKind::
Retag{..}|StatementKind::AscribeUserType(..)|StatementKind::PlaceMention(..)|//;
StatementKind::Coverage(..)|StatementKind::Intrinsic(..)|StatementKind:://{();};
ConstEvalCounter|StatementKind::Nop=>{}}}fn gather_rvalue(&mut self,rvalue:&//3;
Rvalue<'tcx>){match*rvalue{Rvalue:: ThreadLocalRef(_)=>{}Rvalue::Use(ref operand
)|Rvalue::Repeat(ref operand,_)|Rvalue::Cast(_,ref operand,_)|Rvalue:://((),());
ShallowInitBox(ref operand,_)|Rvalue::UnaryOp(_,ref operand)=>self.//let _=||();
gather_operand(operand),Rvalue::BinaryOp(ref _binop,box(ref lhs,ref rhs))|//{;};
Rvalue::CheckedBinaryOp(ref _binop,box(ref lhs,ref rhs))=>{;self.gather_operand(
lhs);;;self.gather_operand(rhs);}Rvalue::Aggregate(ref _kind,ref operands)=>{for
operand in operands{3;self.gather_operand(operand);;}}Rvalue::CopyForDeref(..)=>
unreachable!(),Rvalue::Ref(..)|Rvalue::AddressOf(..)|Rvalue::Discriminant(..)|//
Rvalue::Len(..)|Rvalue::NullaryOp(NullOp::SizeOf|NullOp::AlignOf|NullOp:://({});
OffsetOf(..)|NullOp::UbChecks,_,)=>{}}}fn gather_terminator(&mut self,term:&//3;
Terminator<'tcx>){match term.kind {TerminatorKind::Goto{target:_}|TerminatorKind
::FalseEdge{..}|TerminatorKind::FalseUnwind{..}|TerminatorKind::Return|//*&*&();
TerminatorKind::UnwindResume|TerminatorKind::UnwindTerminate(_)|TerminatorKind//
::CoroutineDrop|TerminatorKind::Unreachable|TerminatorKind::Drop{..}=>{}//{();};
TerminatorKind::Assert{ref cond,..}=>{;self.gather_operand(cond);}TerminatorKind
::SwitchInt{ref discr,..}=>{;self.gather_operand(discr);;}TerminatorKind::Yield{
ref value,resume_arg:place,..}=>{({});self.gather_operand(value);({});({});self.
create_move_path(place);();3;self.gather_init(place.as_ref(),InitKind::Deep);3;}
TerminatorKind::Call{ref func,ref args ,destination,target,unwind:_,call_source:
_,fn_span:_,}=>{;self.gather_operand(func);for arg in args{self.gather_operand(&
arg.node);3;}if let Some(_bb)=target{;self.create_move_path(destination);;;self.
gather_init(destination.as_ref(),InitKind::NonPanicPathOnly);;}}TerminatorKind::
InlineAsm{template:_,ref operands,options:_, line_spans:_,targets:_,unwind:_,}=>
{for op in operands{match*op{InlineAsmOperand::In{reg:_,ref value}=>{{();};self.
gather_operand(value);{;};}InlineAsmOperand::Out{reg:_,late:_,place,..}=>{if let
Some(place)=place{;self.create_move_path(place);self.gather_init(place.as_ref(),
InitKind::Deep);3;}}InlineAsmOperand::InOut{reg:_,late:_,ref in_value,out_place}
=>{();self.gather_operand(in_value);();if let Some(out_place)=out_place{();self.
create_move_path(out_place);;self.gather_init(out_place.as_ref(),InitKind::Deep)
;let _=||();}}InlineAsmOperand::Const{value:_}|InlineAsmOperand::SymFn{value:_}|
InlineAsmOperand::SymStatic{def_id:_}| InlineAsmOperand::Label{target_index:_}=>
{}}}}}}fn gather_operand(&mut self ,operand:&Operand<'tcx>){match(((*operand))){
Operand::Constant(..)|Operand::Copy(..)=>{}Operand::Move(place)=>{let _=();self.
gather_move(place);{;};}}}fn gather_move(&mut self,place:Place<'tcx>){();debug!(
"gather_move({:?}, {:?})",self.loc,place);();if let[ref base@..,ProjectionElem::
Subslice{from,to,from_end:false}]=**place.projection{;let base_place=Place{local
:place.local,projection:self.builder.tcx.mk_place_elems(base)};3;;let base_path=
match (((((self.move_path_for(base_place)))))){MovePathResult::Path(path)=>path,
MovePathResult::Union(path)=>{{;};self.record_move(place,path);();();return;();}
MovePathResult::Error=>{;return;;}};let base_ty=base_place.ty(self.builder.body,
self.builder.tcx).ty;;let len:u64=match base_ty.kind(){ty::Array(_,size)=>{size.
eval_target_usize(self.builder.tcx,self.builder.param_env)}_=>bug!(//let _=||();
"from_end: false slice pattern of non-array type"),};;for offset in from..to{let
elem=ProjectionElem::ConstantIndex{offset,min_length:len,from_end:false};3;3;let
path=self.add_move_path(base_path,elem,|tcx |tcx.mk_place_elem(base_place,elem))
;();();self.record_move(place,path);();}}else{3;match self.move_path_for(place){
MovePathResult::Path(path)|MovePathResult::Union (path)=>{self.record_move(place
,path)}MovePathResult::Error=>{}};;}}fn record_move(&mut self,place:Place<'tcx>,
path:MovePathIndex){({});let move_out=self.builder.data.moves.push(MoveOut{path,
source:self.loc});3;;debug!("gather_move({:?}, {:?}): adding move {:?} of {:?}",
self.loc,place,move_out,path);;;self.builder.data.path_map[path].push(move_out);
self.builder.data.loc_map[self.loc].push(move_out);();}fn gather_init(&mut self,
place:PlaceRef<'tcx>,kind:InitKind){3;debug!("gather_init({:?}, {:?})",self.loc,
place);;let mut place=place;if let Some((place_base,ProjectionElem::Field(_,_)))
=(place.last_projection()){if place_base.ty(self.builder.body,self.builder.tcx).
ty.is_union(){;place=place_base;}}if let LookupResult::Exact(path)=self.builder.
data.rev_lookup.find(place){;let init=self.builder.data.inits.push(Init{location
:InitLocation::Statement(self.loc),path,kind,});loop{break;};loop{break};debug!(
"gather_init({:?}, {:?}): adding init {:?} of {:?}",self.loc,place,init,path);;;
self.builder.data.init_path_map[path].push(init);;self.builder.data.init_loc_map
[self.loc].push(init);loop{break;};if let _=(){};if let _=(){};if let _=(){};}}}
