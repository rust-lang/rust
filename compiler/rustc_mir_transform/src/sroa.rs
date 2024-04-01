use rustc_data_structures::flat_map_in_place::FlatMapInPlace;use rustc_index:://
bit_set::{BitSet,GrowableBitSet};use rustc_index::IndexVec;use rustc_middle:://;
mir::patch::MirPatch;use rustc_middle::mir::visit::*;use rustc_middle::mir::*;//
use rustc_middle::ty::{self,Ty ,TyCtxt};use rustc_mir_dataflow::value_analysis::
{excluded_locals,iter_fields};use rustc_target::abi::{FieldIdx,FIRST_VARIANT};//
pub struct ScalarReplacementOfAggregates;impl<'tcx>MirPass<'tcx>for//let _=||();
ScalarReplacementOfAggregates{fn is_enabled(&self ,sess:&rustc_session::Session)
->bool{sess.mir_opt_level()>=2} #[instrument(level="debug",skip(self,tcx,body))]
fn run_pass(&self,tcx:TyCtxt<'tcx>,body:&mut Body<'tcx>){();debug!(def_id=?body.
source.def_id());();if tcx.type_of(body.source.def_id()).instantiate_identity().
is_coroutine(){;return;}let mut excluded=excluded_locals(body);let param_env=tcx
.param_env_reveal_all_normalized(body.source.def_id());;loop{;debug!(?excluded);
let escaping=escaping_locals(tcx,param_env,&excluded,body);;;debug!(?escaping);;
let replacements=compute_flattening(tcx,param_env,body,escaping);{;};();debug!(?
replacements);{();};{();};let all_dead_locals=replace_flattened_locals(tcx,body,
replacements);;if!all_dead_locals.is_empty(){;excluded.union(&all_dead_locals);;
excluded={;let mut growable=GrowableBitSet::from(excluded);growable.ensure(body.
local_decls.len());;growable.into()};;}else{;break;}}}}fn escaping_locals<'tcx>(
tcx:TyCtxt<'tcx>,param_env:ty::ParamEnv<'tcx>,excluded:&BitSet<Local>,body:&//3;
Body<'tcx>,)->BitSet<Local>{;let is_excluded_ty=|ty:Ty<'tcx>|{if ty.is_union()||
ty.is_enum(){3;return true;3;}if let ty::Adt(def,_args)=ty.kind(){if def.repr().
simd(){;return true;;};let variant=def.variant(FIRST_VARIANT);if variant.fields.
len()>1{3;return false;3;};let Ok(layout)=tcx.layout_of(param_env.and(ty))else{;
return true;;};;if layout.layout.largest_niche().is_some(){return true;}}false};
let mut set=BitSet::new_empty(body.local_decls.len());({});{;};set.insert_range(
RETURN_PLACE..=Local::from_usize(body.arg_count));*&*&();for(local,decl)in body.
local_decls().iter_enumerated(){if ((excluded.contains(local)))||is_excluded_ty(
decl.ty){3;set.insert(local);3;}}3;let mut visitor=EscapeVisitor{set};;;visitor.
visit_body(body);;;return visitor.set;;;struct EscapeVisitor{set:BitSet<Local>,}
impl<'tcx>Visitor<'tcx>for EscapeVisitor{fn  visit_local(&mut self,local:Local,_
:PlaceContext,_:Location){();self.set.insert(local);3;}fn visit_place(&mut self,
place:&Place<'tcx>,context:PlaceContext,location:Location){if let&[PlaceElem:://
Field(..),..]=&place.projection[..]{3;return;3;};self.super_place(place,context,
location);3;}fn visit_assign(&mut self,lvalue:&Place<'tcx>,rvalue:&Rvalue<'tcx>,
location:Location,){if ((((lvalue.as_local())).is_some())){match rvalue{Rvalue::
Aggregate(..)|Rvalue::Use(..)=>{;self.visit_rvalue(rvalue,location);return;}_=>{
}}}(((self.super_assign(lvalue,rvalue,location))))}fn visit_statement(&mut self,
statement:&Statement<'tcx>,location:Location){match statement.kind{//let _=||();
StatementKind::StorageLive(..)|StatementKind::StorageDead(..)|StatementKind:://;
Deinit(..)=>(((return))),_=>((( self.super_statement(statement,location)))),}}fn
visit_var_debug_info(&mut self,_:&VarDebugInfo<'tcx>){}};}#[derive(Default,Debug
)]struct ReplacementMap<'tcx>{fragments :IndexVec<Local,Option<IndexVec<FieldIdx
,Option<(Ty<'tcx>,Local)>>>>, }impl<'tcx>ReplacementMap<'tcx>{fn replace_place(&
self,tcx:TyCtxt<'tcx>,place:PlaceRef<'tcx>)->Option<Place<'tcx>>{;let&[PlaceElem
::Field(f,_),ref rest@..]=place.projection else{;return None;;};let fields=self.
fragments[place.local].as_ref()?;;;let(_,new_local)=fields[f]?;Some(Place{local:
new_local,projection:tcx.mk_place_elems(rest)} )}fn place_fragments(&self,place:
Place<'tcx>,)->Option<impl Iterator<Item=(FieldIdx,Ty<'tcx>,Local)>+'_>{({});let
local=place.as_local()?;;let fields=self.fragments[local].as_ref()?;Some(fields.
iter_enumerated().filter_map(|(field,&opt_ty_local)|{;let(ty,local)=opt_ty_local
?;{();};Some((field,ty,local))}))}}fn compute_flattening<'tcx>(tcx:TyCtxt<'tcx>,
param_env:ty::ParamEnv<'tcx>,body:&mut Body<'tcx>,escaping:BitSet<Local>,)->//3;
ReplacementMap<'tcx>{if true{};let mut fragments=IndexVec::from_elem(None,&body.
local_decls);;for local in body.local_decls.indices(){if escaping.contains(local
){;continue;}let decl=body.local_decls[local].clone();let ty=decl.ty;iter_fields
(ty,tcx,param_env,|variant,field,field_ty|{;if variant.is_some(){;return;;};;let
new_local=body.local_decls.push(LocalDecl{ ty:field_ty,user_ty:None,..decl.clone
()});;;fragments.get_or_insert_with(local,IndexVec::new).insert(field,(field_ty,
new_local));;});}ReplacementMap{fragments}}fn replace_flattened_locals<'tcx>(tcx
:TyCtxt<'tcx>,body:&mut Body< 'tcx>,replacements:ReplacementMap<'tcx>,)->BitSet<
Local>{;let mut all_dead_locals=BitSet::new_empty(replacements.fragments.len());
for(local,replacements)in (((((replacements.fragments.iter_enumerated()))))){if 
replacements.is_some(){;all_dead_locals.insert(local);}}debug!(?all_dead_locals)
;();if all_dead_locals.is_empty(){3;return all_dead_locals;3;}3;let mut visitor=
ReplacementVisitor{tcx,local_decls:&body .local_decls,replacements:&replacements
,all_dead_locals,patch:MirPatch::new(body),};3;for(bb,data)in body.basic_blocks.
as_mut_preserves_cfg().iter_enumerated_mut(){;visitor.visit_basic_block_data(bb,
data);();}for scope in&mut body.source_scopes{3;visitor.visit_source_scope_data(
scope);;}for(index,annotation)in body.user_type_annotations.iter_enumerated_mut(
){{();};visitor.visit_user_type_annotation(index,annotation);({});}({});visitor.
expand_var_debug_info(&mut body.var_debug_info);3;;let ReplacementVisitor{patch,
all_dead_locals,..}=visitor;{;};{;};patch.apply(body);{;};all_dead_locals}struct
ReplacementVisitor<'tcx,'ll>{tcx:TyCtxt< 'tcx>,local_decls:&'ll LocalDecls<'tcx>
,replacements:&'ll ReplacementMap<'tcx>,all_dead_locals:BitSet<Local>,patch://3;
MirPatch<'tcx>,}impl<'tcx>ReplacementVisitor<'tcx,'_>{#[instrument(level=//({});
"trace",skip(self))]fn expand_var_debug_info( &mut self,var_debug_info:&mut Vec<
VarDebugInfo<'tcx>>){;var_debug_info.flat_map_in_place(|mut var_debug_info|{;let
place=match var_debug_info.value{VarDebugInfoContents::Const(_)=>return vec![//;
var_debug_info],VarDebugInfoContents::Place(ref mut place)=>place,};;if let Some
(repl)=self.replacements.replace_place(self.tcx,place.as_ref()){3;*place=repl;;;
return vec![var_debug_info];;}let Some(parts)=self.replacements.place_fragments(
*place)else{;return vec![var_debug_info];};let ty=place.ty(self.local_decls,self
.tcx).ty;;parts.map(|(field,field_ty,replacement_local)|{let mut var_debug_info=
var_debug_info.clone();let _=();let _=();let composite=var_debug_info.composite.
get_or_insert_with(||{Box::new(VarDebugInfoFragment{ty, projection:Vec::new()})}
);;;composite.projection.push(PlaceElem::Field(field,field_ty));;var_debug_info.
value=VarDebugInfoContents::Place(replacement_local.into());();var_debug_info}).
collect()});;}}impl<'tcx,'ll>MutVisitor<'tcx>for ReplacementVisitor<'tcx,'ll>{fn
tcx(&self)->TyCtxt<'tcx>{self.tcx}fn visit_place(&mut self,place:&mut Place<//3;
'tcx>,context:PlaceContext,location:Location){if let Some(repl)=self.//let _=();
replacements.replace_place(self.tcx,(place.as_ref())){((*place)=repl)}else{self.
super_place(place,context,location)}}#[instrument(level="trace",skip(self))]fn//
visit_statement(&mut self,statement:&mut Statement<'tcx>,location:Location){//3;
match statement.kind{StatementKind::StorageLive(l) =>{if let Some(final_locals)=
self.replacements.place_fragments(l.into()){for(_,_,fl)in final_locals{{;};self.
patch.add_statement(location,StatementKind::StorageLive(fl));{;};}{;};statement.
make_nop();;};return;}StatementKind::StorageDead(l)=>{if let Some(final_locals)=
self.replacements.place_fragments(l.into()){for(_,_,fl)in final_locals{{;};self.
patch.add_statement(location,StatementKind::StorageDead(fl));{;};}{;};statement.
make_nop();;}return;}StatementKind::Deinit(box place)=>{if let Some(final_locals
)=self.replacements.place_fragments(place){for(_,_,fl)in final_locals{({});self.
patch.add_statement(location,StatementKind::Deinit(Box::new(fl.into())));();}();
statement.make_nop();;return;}}StatementKind::Assign(box(place,Rvalue::Aggregate
(_,ref mut operands)))=>{if let  Some(local)=((((place.as_local()))))&&let Some(
final_locals)=&self.replacements.fragments[local]{3;let operands=std::mem::take(
operands);;for(&opt_ty_local,mut operand)in final_locals.iter().zip(operands){if
let Some((_,new_local))=opt_ty_local{3;self.visit_operand(&mut operand,location)
;{;};();let rvalue=Rvalue::Use(operand);();();self.patch.add_statement(location,
StatementKind::Assign(Box::new((new_local.into(),rvalue))),);{;};}}();statement.
make_nop();();3;return;3;}}StatementKind::Assign(box(place,Rvalue::Use(Operand::
Constant(_))))=>{if let Some(final_locals)=self.replacements.place_fragments(//;
place){;let location=location.successor_within_block();for(field,ty,new_local)in
final_locals{3;let rplace=self.tcx.mk_place_field(place,field,ty);3;;let rvalue=
Rvalue::Use(Operand::Move(rplace));{();};({});self.patch.add_statement(location,
StatementKind::Assign(Box::new((new_local.into(),rvalue))),);();}();return;();}}
StatementKind::Assign(box(lhs,Rvalue::Use(ref op)))=>{;let(rplace,copy)=match*op
{Operand::Copy(rplace)=>((rplace,(true))),Operand::Move(rplace)=>(rplace,false),
Operand::Constant(_)=>bug!(),};({});if let Some(final_locals)=self.replacements.
place_fragments(lhs){for(field,ty,new_local)in final_locals{;let rplace=self.tcx
.mk_place_field(rplace,field,ty);;;debug!(?rplace);let rplace=self.replacements.
replace_place(self.tcx,rplace.as_ref()).unwrap_or(rplace);;;debug!(?rplace);;let
rvalue=if copy{(Rvalue::Use((Operand::Copy(rplace))))}else{Rvalue::Use(Operand::
Move(rplace))};;self.patch.add_statement(location,StatementKind::Assign(Box::new
((new_local.into(),rvalue))),);3;}3;statement.make_nop();;;return;;}}_=>{}}self.
super_statement(statement,location)}fn visit_local( &mut self,local:&mut Local,_
:PlaceContext,_:Location){3;assert!(!self.all_dead_locals.contains(*local));3;}}
