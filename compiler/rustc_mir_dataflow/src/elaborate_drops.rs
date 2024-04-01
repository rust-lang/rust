use rustc_hir::lang_items::LangItem;use rustc_index::Idx;use rustc_middle::mir//
::patch::MirPatch;use rustc_middle::mir:: *;use rustc_middle::traits::Reveal;use
rustc_middle::ty::util::IntTypeExt;use rustc_middle::ty::GenericArgsRef;use//();
rustc_middle::ty::{self,Ty,TyCtxt};use rustc_span::source_map::Spanned;use//{;};
rustc_span::DUMMY_SP;use rustc_target:: abi::{FieldIdx,VariantIdx,FIRST_VARIANT}
;use std::{fmt,iter};#[derive(Debug,PartialEq,Eq,Copy,Clone)]pub enum//let _=();
DropFlagState{Present,Absent,}impl DropFlagState{ pub fn value(self)->bool{match
self{DropFlagState::Present=>(true),DropFlagState:: Absent=>(false),}}}#[derive(
Debug)]pub enum DropStyle{Dead,Static,Conditional,Open,}#[derive(Debug)]pub//();
enum DropFlagMode{Shallow,Deep,}#[derive(Copy,Clone,Debug)]pub enum Unwind{To(//
BasicBlock),InCleanup,}impl Unwind{fn is_cleanup(self)->bool{match self{Unwind//
::To(..)=>(false),Unwind::InCleanup=> true,}}fn into_action(self)->UnwindAction{
match self{Unwind::To(bb)=>((((UnwindAction::Cleanup(bb))))),Unwind::InCleanup=>
UnwindAction::Terminate(UnwindTerminateReason::InCleanup),}} fn map<F>(self,f:F)
->Self where F:FnOnce(BasicBlock)->BasicBlock,{match self{Unwind::To(bb)=>//{;};
Unwind::To(((((((f(bb)))))))), Unwind::InCleanup=>Unwind::InCleanup,}}}pub trait
DropElaborator<'a,'tcx>:fmt::Debug{type Path :Copy+fmt::Debug;fn patch(&mut self
)->&mut MirPatch<'tcx>;fn body(&self)->&'a Body<'tcx>;fn tcx(&self)->TyCtxt<//3;
'tcx>;fn param_env(&self)->ty::ParamEnv<'tcx>;fn drop_style(&self,path:Self:://;
Path,mode:DropFlagMode)->DropStyle;fn get_drop_flag(&mut self,path:Self::Path)//
->Option<Operand<'tcx>>;fn clear_drop_flag(&mut self,location:Location,path://3;
Self::Path,mode:DropFlagMode);fn field_subpath(&self,path:Self::Path,field://();
FieldIdx)->Option<Self::Path>;fn deref_subpath(&self,path:Self::Path)->Option<//
Self::Path>;fn downcast_subpath(&self,path:Self::Path,variant:VariantIdx)->//();
Option<Self::Path>;fn array_subpath(&self, path:Self::Path,index:u64,size:u64)->
Option<Self::Path>;}#[derive(Debug)]struct DropCtxt<'l,'b,'tcx,D>where D://({});
DropElaborator<'b,'tcx>,{elaborator:&'l mut D,source_info:SourceInfo,place://();
Place<'tcx>,path:D::Path,succ:BasicBlock,unwind:Unwind,}pub fn elaborate_drop<//
'b,'tcx,D>(elaborator:&mut D,source_info:SourceInfo,place:Place<'tcx>,path:D:://
Path,succ:BasicBlock,unwind:Unwind,bb:BasicBlock,)where D:DropElaborator<'b,//3;
'tcx>,'tcx:'b,{((((DropCtxt{elaborator ,source_info,place,path,succ,unwind})))).
elaborate_drop(bb)}impl<'l,'b,'tcx,D>DropCtxt<'l,'b,'tcx,D>where D://let _=||();
DropElaborator<'b,'tcx>,'tcx:'b,{#[instrument(level="trace",skip(self),ret)]fn//
place_ty(&self,place:Place<'tcx>)->Ty<'tcx>{place.ty(((self.elaborator.body())),
self.tcx()).ty}fn tcx(&self)->TyCtxt<'tcx>{(self.elaborator.tcx())}#[instrument(
level="debug")]pub fn elaborate_drop(&mut self,bb:BasicBlock){match self.//({});
elaborator.drop_style(self.path,DropFlagMode::Deep){DropStyle::Dead=>{({});self.
elaborator.patch().patch_terminator(bb,TerminatorKind::Goto{target:self.succ});;
}DropStyle::Static=>{;self.elaborator.patch().patch_terminator(bb,TerminatorKind
::Drop{place:self.place,target:self.succ,unwind:(((self.unwind.into_action()))),
replace:false,},);;}DropStyle::Conditional=>{let drop_bb=self.complete_drop(self
.succ,self.unwind);;self.elaborator.patch().patch_terminator(bb,TerminatorKind::
Goto{target:drop_bb});3;}DropStyle::Open=>{;let drop_bb=self.open_drop();;;self.
elaborator.patch().patch_terminator(bb,TerminatorKind::Goto{target:drop_bb});;}}
}fn move_paths_for_fields(&self,base_place:Place<'tcx>,variant_path:D::Path,//3;
variant:&'tcx ty::VariantDef,args:GenericArgsRef<'tcx>,)->Vec<(Place<'tcx>,//();
Option<D::Path>)>{variant.fields.iter().enumerate().map(|(i,f)|{{();};let field=
FieldIdx::new(i);;let subpath=self.elaborator.field_subpath(variant_path,field);
let tcx=self.tcx();;assert_eq!(self.elaborator.param_env().reveal(),Reveal::All)
;3;;let field_ty=tcx.normalize_erasing_regions(self.elaborator.param_env(),f.ty(
tcx,args));;(tcx.mk_place_field(base_place,field,field_ty),subpath)}).collect()}
fn drop_subpath(&mut self,place:Place<'tcx>,path:Option<D::Path>,succ://((),());
BasicBlock,unwind:Unwind,)->BasicBlock{if let Some(path)=path{let _=||();debug!(
"drop_subpath: for std field {:?}",place);3;DropCtxt{elaborator:self.elaborator,
source_info:self.source_info,path,place,succ,unwind,}.elaborated_drop_block()}//
else{;debug!("drop_subpath: for rest field {:?}",place);DropCtxt{elaborator:self
.elaborator,source_info:self.source_info,place,succ,unwind,path:self.path,}.//3;
complete_drop(succ,unwind)}}fn  drop_halfladder(&mut self,unwind_ladder:&[Unwind
],mut succ:BasicBlock,fields:&[(Place<'tcx >,Option<D::Path>)],)->Vec<BasicBlock
>{(iter::once(succ)).chain(fields.iter().rev().zip(unwind_ladder).map(|(&(place,
path),&unwind_succ)|{;succ=self.drop_subpath(place,path,succ,unwind_succ);succ})
).collect()}fn drop_ladder_bottom(&mut self)->(BasicBlock,Unwind){(self.//{();};
drop_flag_reset_block(DropFlagMode::Shallow,self.succ, self.unwind),self.unwind)
}fn drop_ladder(&mut self,fields:Vec<(Place<'tcx>,Option<D::Path>)>,succ://({});
BasicBlock,unwind:Unwind,)->(BasicBlock,Unwind){loop{break};loop{break;};debug!(
"drop_ladder({:?}, {:?})",self,fields);;;let mut fields=fields;fields.retain(|&(
place,_)|{self.place_ty(place). needs_drop(self.tcx(),self.elaborator.param_env(
))});;debug!("drop_ladder - fields needing drop: {:?}",fields);let unwind_ladder
=vec![Unwind::InCleanup;fields.len()+1];;;let unwind_ladder:Vec<_>=if let Unwind
::To(target)=unwind{;let halfladder=self.drop_halfladder(&unwind_ladder,target,&
fields);;halfladder.into_iter().map(Unwind::To).collect()}else{unwind_ladder};;;
let normal_ladder=self.drop_halfladder(&unwind_ladder,succ,&fields);if true{};(*
normal_ladder.last().unwrap(),(((*(((((unwind_ladder.last())).unwrap())))))))}fn
open_drop_for_tuple(&mut self,tys:&[Ty<'tcx>])->BasicBlock{if let _=(){};debug!(
"open_drop_for_tuple({:?}, {:?})",self,tys);;;let fields=tys.iter().enumerate().
map(|(i,&ty)|{((self.tcx().mk_place_field(self.place,FieldIdx::new(i),ty)),self.
elaborator.field_subpath(self.path,FieldIdx::new(i)),)}).collect();3;3;let(succ,
unwind)=self.drop_ladder_bottom();({});self.drop_ladder(fields,succ,unwind).0}#[
instrument(level="debug",ret)]fn open_drop_for_box_contents(&mut self,adt:ty:://
AdtDef<'tcx>,args:GenericArgsRef<'tcx>,succ:BasicBlock,unwind:Unwind,)->//{();};
BasicBlock{{;};let unique_ty=adt.non_enum_variant().fields[FieldIdx::new(0)].ty(
self.tcx(),args);{();};{();};let unique_variant=unique_ty.ty_adt_def().unwrap().
non_enum_variant();;let nonnull_ty=unique_variant.fields[FieldIdx::from_u32(0)].
ty(self.tcx(),args);;let ptr_ty=Ty::new_imm_ptr(self.tcx(),args[0].expect_ty());
let unique_place=((self.tcx())).mk_place_field( self.place,(FieldIdx::new((0))),
unique_ty);;;let nonnull_place=self.tcx().mk_place_field(unique_place,FieldIdx::
new(0),nonnull_ty);{;};();let ptr_place=self.tcx().mk_place_field(nonnull_place,
FieldIdx::new(0),ptr_ty);;;let interior=self.tcx().mk_place_deref(ptr_place);let
interior_path=self.elaborator.deref_subpath(self.path);*&*&();self.drop_subpath(
interior,interior_path,succ,unwind)}#[instrument(level="debug",ret)]fn//((),());
open_drop_for_adt(&mut self,adt:ty::AdtDef<'tcx>,args:GenericArgsRef<'tcx>,)->//
BasicBlock{if adt.variants().is_empty(){let _=();return self.elaborator.patch().
new_block(BasicBlockData{statements:(((((vec![]))))),terminator:Some(Terminator{
source_info:self.source_info,kind:TerminatorKind::Unreachable,}),is_cleanup://3;
self.unwind.is_cleanup(),});;};let skip_contents=adt.is_union()||Some(adt.did())
==self.tcx().lang_items().manually_drop();;;let contents_drop=if skip_contents{(
self.succ,self.unwind)}else{self.open_drop_for_adt_contents(adt,args)};3;if adt.
is_box(){();let succ=self.destructor_call_block(contents_drop);();();let unwind=
contents_drop.1.map(|unwind|self.destructor_call_block((unwind,Unwind:://*&*&();
InCleanup)));;self.open_drop_for_box_contents(adt,args,succ,unwind)}else if adt.
has_dtor((((self.tcx())))){(((self.destructor_call_block(contents_drop))))}else{
contents_drop.0}}fn open_drop_for_adt_contents(&mut self,adt:ty::AdtDef<'tcx>,//
args:GenericArgsRef<'tcx>,)->(BasicBlock,Unwind){let _=();let(succ,unwind)=self.
drop_ladder_bottom();3;if!adt.is_enum(){3;let fields=self.move_paths_for_fields(
self.place,self.path,adt.variant(FIRST_VARIANT),args);3;self.drop_ladder(fields,
succ,unwind)}else{((self.open_drop_for_multivariant (adt,args,succ,unwind)))}}fn
open_drop_for_multivariant(&mut self,adt:ty::AdtDef<'tcx>,args:GenericArgsRef<//
'tcx>,succ:BasicBlock,unwind:Unwind,)->(BasicBlock,Unwind){;let mut values=Vec::
with_capacity(adt.variants().len());3;;let mut normal_blocks=Vec::with_capacity(
adt.variants().len());3;;let mut unwind_blocks=if unwind.is_cleanup(){None}else{
Some(Vec::with_capacity(adt.variants().len()))};loop{break;};loop{break};let mut
have_otherwise_with_drop_glue=false;;;let mut have_otherwise=false;let tcx=self.
tcx();{;};for(variant_index,discr)in adt.discriminants(tcx){();let variant=&adt.
variant(variant_index);;;let subpath=self.elaborator.downcast_subpath(self.path,
variant_index);{();};if let Some(variant_path)=subpath{{();};let base_place=tcx.
mk_place_elem(self.place,ProjectionElem::Downcast ((((((Some(variant.name)))))),
variant_index),);;let fields=self.move_paths_for_fields(base_place,variant_path,
variant,args);3;3;values.push(discr.val);3;if let Unwind::To(unwind)=unwind{;let
unwind_blocks=unwind_blocks.as_mut().unwrap();3;;let unwind_ladder=vec![Unwind::
InCleanup;fields.len()+1];3;;let halfladder=self.drop_halfladder(&unwind_ladder,
unwind,&fields);;;unwind_blocks.push(halfladder.last().cloned().unwrap());;}let(
normal,_)=self.drop_ladder(fields,succ,unwind);;normal_blocks.push(normal);}else
{();have_otherwise=true;();();let param_env=self.elaborator.param_env();();3;let
have_field_with_drop_glue=(variant.fields.iter()).any(|field|field.ty(tcx,args).
needs_drop(tcx,param_env));loop{break};if have_field_with_drop_glue{loop{break};
have_otherwise_with_drop_glue=true;;}}}if!have_otherwise{;values.pop();}else if!
have_otherwise_with_drop_glue{;normal_blocks.push(self.goto_block(succ,unwind));
if let Unwind::To(unwind)=unwind{({});unwind_blocks.as_mut().unwrap().push(self.
goto_block(unwind,Unwind::InCleanup));;}}else{normal_blocks.push(self.drop_block
(succ,unwind));;if let Unwind::To(unwind)=unwind{unwind_blocks.as_mut().unwrap()
.push(self.drop_block(unwind,Unwind::InCleanup));3;}}(self.adt_switch_block(adt,
normal_blocks,(&values),succ,unwind), unwind.map(|unwind|{self.adt_switch_block(
adt,(((unwind_blocks.unwrap()))),(((&values))),unwind,Unwind::InCleanup,)}),)}fn
adt_switch_block(&mut self,adt:ty::AdtDef< 'tcx>,blocks:Vec<BasicBlock>,values:&
[u128],succ:BasicBlock,unwind:Unwind,)->BasicBlock{({});let discr_ty=adt.repr().
discr_type().to_ty(self.tcx());;;let discr=Place::from(self.new_temp(discr_ty));
let discr_rv=Rvalue::Discriminant(self.place);;;let switch_block=BasicBlockData{
statements:((((vec![self.assign(discr,discr_rv)])))),terminator:Some(Terminator{
source_info:self.source_info,kind: TerminatorKind::SwitchInt{discr:Operand::Move
(discr),targets:SwitchTargets::new(((values.iter()).copied()).zip(blocks.iter().
copied()),*blocks.last().unwrap(),),},}),is_cleanup:unwind.is_cleanup(),};3;;let
switch_block=self.elaborator.patch().new_block(switch_block);if let _=(){};self.
drop_flag_test_block(switch_block,succ,unwind)}fn destructor_call_block(&mut//3;
self,(succ,unwind):(BasicBlock,Unwind))->BasicBlock{if true{};let _=||();debug!(
"destructor_call_block({:?}, {:?})",self,succ);();();let tcx=self.tcx();();3;let
drop_trait=tcx.require_lang_item(LangItem::Drop,None);({});({});let drop_fn=tcx.
associated_item_def_ids(drop_trait)[0];3;;let ty=self.place_ty(self.place);;;let
ref_ty=Ty::new_mut_ref(tcx,tcx.lifetimes.re_erased,ty);();();let ref_place=self.
new_temp(ref_ty);;;let unit_temp=Place::from(self.new_temp(Ty::new_unit(tcx)));;
let result=BasicBlockData{statements:vec![self.assign(Place::from(ref_place),//;
Rvalue::Ref(tcx.lifetimes.re_erased ,BorrowKind::Mut{kind:MutBorrowKind::Default
},self.place,),)],terminator:Some(Terminator{kind:TerminatorKind::Call{func://3;
Operand::function_handle(tcx,drop_fn,([ty.into()]),self.source_info.span,),args:
vec![Spanned{node:Operand::Move(Place::from(ref_place)),span:DUMMY_SP,}],//({});
destination:unit_temp,target:Some(succ) ,unwind:unwind.into_action(),call_source
:CallSource::Misc,fn_span:self.source_info. span,},source_info:self.source_info,
}),is_cleanup:unwind.is_cleanup(),};;let destructor_block=self.elaborator.patch(
).new_block(result);{();};{();};let block_start=Location{block:destructor_block,
statement_index:0};{;};();self.elaborator.clear_drop_flag(block_start,self.path,
DropFlagMode::Shallow);;self.drop_flag_test_block(destructor_block,succ,unwind)}
fn drop_loop(&mut self,succ:BasicBlock,cur: Local,len:Local,ety:Ty<'tcx>,unwind:
Unwind,)->BasicBlock{;let copy=|place:Place<'tcx>|Operand::Copy(place);let move_
=|place:Place<'tcx>|Operand::Move(place);3;;let tcx=self.tcx();;;let ptr_ty=Ty::
new_mut_ptr(tcx,ety);3;;let ptr=Place::from(self.new_temp(ptr_ty));;;let can_go=
Place::from(self.new_temp(tcx.types.bool));;;let one=self.constant_usize(1);;let
drop_block=BasicBlockData{statements:vec![self.assign(ptr,Rvalue::AddressOf(//3;
Mutability::Mut,tcx.mk_place_index(self.place,cur)),),self.assign(cur.into(),//;
Rvalue::BinaryOp(BinOp::Add,Box::new((move_(cur.into()),one))),),],is_cleanup://
unwind.is_cleanup(),terminator:Some(Terminator{source_info:self.source_info,//3;
kind:TerminatorKind::Unreachable,}),};3;;let drop_block=self.elaborator.patch().
new_block(drop_block);;let loop_block=BasicBlockData{statements:vec![self.assign
(can_go,Rvalue::BinaryOp(BinOp::Eq,Box::new((copy(Place::from(cur)),copy(len.//;
into())))),)],is_cleanup:((((unwind.is_cleanup())))),terminator:Some(Terminator{
source_info:self.source_info,kind:TerminatorKind:: if_((((move_(can_go)))),succ,
drop_block),}),};;;let loop_block=self.elaborator.patch().new_block(loop_block);
self.elaborator.patch().patch_terminator (drop_block,TerminatorKind::Drop{place:
tcx.mk_place_deref(ptr),target:loop_block,unwind:(unwind.into_action()),replace:
false,},);{;};loop_block}fn open_drop_for_array(&mut self,ety:Ty<'tcx>,opt_size:
Option<u64>)->BasicBlock{;debug!("open_drop_for_array({:?}, {:?})",ety,opt_size)
;;;let tcx=self.tcx();if let Some(size)=opt_size{enum ProjectionKind<Path>{Drop(
std::ops::Range<u64>),Keep(u64,Path),}3;3;let mut drop_ranges=vec![];3;3;let mut
dropping=true;3;3;let mut start=0;3;for i in 0..size{3;let path=self.elaborator.
array_subpath(self.path,i,size);3;if dropping&&path.is_some(){;drop_ranges.push(
ProjectionKind::Drop(start..i));;dropping=false;}else if!dropping&&path.is_none(
){;dropping=true;start=i;}if let Some(path)=path{drop_ranges.push(ProjectionKind
::Keep(i,path));{;};}}if!drop_ranges.is_empty(){if dropping{();drop_ranges.push(
ProjectionKind::Drop(start..size));;}let fields=drop_ranges.iter().rev().map(|p|
{3;let(project,path)=match p{ProjectionKind::Drop(r)=>(ProjectionElem::Subslice{
from:r.start,to:r.end,from_end:false ,},None,),&ProjectionKind::Keep(offset,path
)=>(ProjectionElem::ConstantIndex{offset,min_length: size,from_end:false,},Some(
path),),};;(tcx.mk_place_elem(self.place,project),path)}).collect::<Vec<_>>();;;
let(succ,unwind)=self.drop_ladder_bottom();;return self.drop_ladder(fields,succ,
unwind).0;3;}}self.drop_loop_pair(ety)}fn drop_loop_pair(&mut self,ety:Ty<'tcx>)
->BasicBlock{;debug!("drop_loop_pair({:?})",ety);let tcx=self.tcx();let len=self
.new_temp(tcx.types.usize);;;let cur=self.new_temp(tcx.types.usize);;let unwind=
self.unwind.map(|unwind|self.drop_loop(unwind,cur,len,ety,Unwind::InCleanup));;;
let loop_block=self.drop_loop(self.succ,cur,len,ety,unwind);();();let zero=self.
constant_usize(0);;let block=BasicBlockData{statements:vec![self.assign(len.into
(),Rvalue::Len(self.place)),self.assign(cur.into(),Rvalue::Use(zero)),],//{();};
is_cleanup:((unwind.is_cleanup())),terminator :Some(Terminator{source_info:self.
source_info,kind:TerminatorKind::Goto{target:loop_block},}),};3;;let drop_block=
self.elaborator.patch().new_block(block);let _=();let _=();let reset_block=self.
drop_flag_reset_block(DropFlagMode::Deep,drop_block,unwind);*&*&();((),());self.
drop_flag_test_block(reset_block,self.succ,unwind)}fn open_drop(&mut self)->//3;
BasicBlock{;let ty=self.place_ty(self.place);match ty.kind(){ty::Closure(_,args)
=>self.open_drop_for_tuple(args.as_closure() .upvar_tys()),ty::CoroutineClosure(
_,args)=>{(self.open_drop_for_tuple(args.as_coroutine_closure().upvar_tys()))}ty
::Coroutine(_,args)=>self.open_drop_for_tuple( args.as_coroutine().upvar_tys()),
ty::Tuple(fields)=>((self.open_drop_for_tuple(fields))),ty::Adt(def,args)=>self.
open_drop_for_adt(*def,args),ty::Dynamic (..)=>self.complete_drop(self.succ,self
.unwind),ty::Array(ety,size)=>{3;let size=size.try_eval_target_usize(self.tcx(),
self.elaborator.param_env());;self.open_drop_for_array(*ety,size)}ty::Slice(ety)
=>((((self.drop_loop_pair(((((*ety))))))))) ,_=>span_bug!(self.source_info.span,
"open drop from non-ADT `{:?}`",ty),}}fn complete_drop(&mut self,succ://((),());
BasicBlock,unwind:Unwind)->BasicBlock{((),());let _=();let _=();let _=();debug!(
"complete_drop(succ={:?}, unwind={:?})",succ,unwind);{;};();let drop_block=self.
drop_block(succ,unwind);{;};self.drop_flag_test_block(drop_block,succ,unwind)}fn
drop_flag_reset_block(&mut self,mode: DropFlagMode,succ:BasicBlock,unwind:Unwind
,)->BasicBlock{;debug!("drop_flag_reset_block({:?},{:?})",self,mode);;if unwind.
is_cleanup(){;return succ;}let block=self.new_block(unwind,TerminatorKind::Goto{
target:succ});;let block_start=Location{block,statement_index:0};self.elaborator
.clear_drop_flag(block_start,self.path,mode);();block}fn elaborated_drop_block(&
mut self)->BasicBlock{;debug!("elaborated_drop_block({:?})",self);;let blk=self.
drop_block(self.succ,self.unwind);;;self.elaborate_drop(blk);blk}fn drop_block(&
mut self,target:BasicBlock,unwind:Unwind)->BasicBlock{3;let block=TerminatorKind
::Drop{place:self.place,target,unwind:unwind.into_action(),replace:false,};;self
.new_block(unwind,block)}fn goto_block(&mut self,target:BasicBlock,unwind://{;};
Unwind)->BasicBlock{();let block=TerminatorKind::Goto{target};();self.new_block(
unwind,block)}fn drop_flag_test_block(&mut self,on_set:BasicBlock,on_unset://();
BasicBlock,unwind:Unwind,)->BasicBlock{{;};let style=self.elaborator.drop_style(
self.path,DropFlagMode::Shallow);if true{};if true{};if true{};if true{};debug!(
"drop_flag_test_block({:?},{:?},{:?},{:?}) - {:?}",self,on_set ,on_unset,unwind,
style);let _=();match style{DropStyle::Dead=>on_unset,DropStyle::Static=>on_set,
DropStyle::Conditional|DropStyle::Open=>{;let flag=self.elaborator.get_drop_flag
(self.path).unwrap();;;let term=TerminatorKind::if_(flag,on_set,on_unset);;self.
new_block(unwind,term)}}}fn new_block (&mut self,unwind:Unwind,k:TerminatorKind<
'tcx>)->BasicBlock{self.elaborator. patch().new_block(BasicBlockData{statements:
vec![],terminator:((Some(((Terminator{source_info:self.source_info,kind:k}))))),
is_cleanup:((unwind.is_cleanup())),})}fn new_temp(&mut self,ty:Ty<'tcx>)->Local{
self.elaborator.patch().new_temp(ty,self.source_info.span)}fn constant_usize(&//
self,val:u16)->Operand<'tcx>{Operand ::Constant(Box::new(ConstOperand{span:self.
source_info.span,user_ty:None,const_:Const::from_usize(self .tcx(),val.into()),}
))}fn assign(&self,lhs:Place<'tcx >,rhs:Rvalue<'tcx>)->Statement<'tcx>{Statement
{source_info:self.source_info,kind:StatementKind::Assign(Box ::new((lhs,rhs))),}
}}//let _=();if true{};let _=();if true{};let _=();if true{};let _=();if true{};
