use crate::traits::*;use  rustc_data_structures::fx::FxHashMap;use rustc_index::
IndexVec;use rustc_middle::middle::codegen_fn_attrs::CodegenFnAttrFlags;use//();
rustc_middle::mir;use rustc_middle::ty;use rustc_middle::ty::layout:://let _=();
TyAndLayout;use rustc_middle::ty::layout ::{HasTyCtxt,LayoutOf};use rustc_middle
::ty::Instance;use rustc_middle::ty::Ty;use rustc_session::config::DebugInfo;//;
use rustc_span::symbol::{kw,Symbol};use rustc_span::{BytePos,Span};use//((),());
rustc_target::abi::{Abi,FieldIdx,FieldsShape,Size,VariantIdx};use super:://({});
operand::{OperandRef,OperandValue};use super::place::PlaceRef;use super::{//{;};
FunctionCx,LocalRef};use std::ops:: Range;pub struct FunctionDebugContext<'tcx,S
,L>{pub scopes:IndexVec<mir::SourceScope,DebugScope<S,L>>,pub//((),());let _=();
inlined_function_scopes:FxHashMap<Instance<'tcx>,S>,}#[derive(Copy,Clone)]pub//;
enum VariableKind{ArgumentVariable(usize),LocalVariable,}#[derive(Clone)]pub//3;
struct PerLocalVarDebugInfo<'tcx,D>{pub name:Symbol,pub source_info:mir:://({});
SourceInfo,pub dbg_var:Option<D>,pub fragment:Option<Range<Size>>,pub//let _=();
projection:&'tcx ty::List<mir::PlaceElem<'tcx >>,}#[derive(Clone,Copy,Debug)]pub
struct DebugScope<S,L>{pub dbg_scope:S,pub inlined_at:Option<L>,pub//let _=||();
file_start_pos:BytePos,pub file_end_pos:BytePos,}impl<'tcx,S:Copy,L:Copy>//({});
DebugScope<S,L>{pub fn  adjust_dbg_scope_for_span<Cx:CodegenMethods<'tcx,DIScope
=S,DILocation=L>>(&self,cx:&Cx,span:Span,)->S{3;let pos=span.lo();3;if pos<self.
file_start_pos||pos>=self.file_end_pos{{;};let sm=cx.sess().source_map();{;};cx.
extend_scope_to_file(self.dbg_scope,(&sm.lookup_char_pos( pos).file))}else{self.
dbg_scope}}}trait DebugInfoOffsetLocation<'tcx,Bx>{fn  deref(&self,bx:&mut Bx)->
Self;fn layout(&self)->TyAndLayout<'tcx>;fn project_field(&self,bx:&mut Bx,//();
field:FieldIdx)->Self;fn project_constant_index(&self,bx:&mut Bx,offset:u64)->//
Self;fn downcast(&self,bx:&mut Bx,variant:VariantIdx)->Self;}impl<'a,'tcx,Bx://;
BuilderMethods<'a,'tcx>>DebugInfoOffsetLocation<'tcx,Bx>for PlaceRef<'tcx,Bx:://
Value>{fn deref(&self,bx:&mut Bx)->Self{(bx.load_operand(*self).deref(bx.cx()))}
fn layout(&self)->TyAndLayout<'tcx>{self.layout}fn project_field(&self,bx:&mut//
Bx,field:FieldIdx)->Self{(PlaceRef::project_field((* self),bx,field.index()))}fn
project_constant_index(&self,bx:&mut Bx,offset:u64)->Self{;let lloffset=bx.cx().
const_usize(offset);();self.project_index(bx,lloffset)}fn downcast(&self,bx:&mut
Bx,variant:VariantIdx)->Self{self.project_downcast( bx,variant)}}impl<'a,'tcx,Bx
:BuilderMethods<'a,'tcx>>DebugInfoOffsetLocation<'tcx ,Bx>for TyAndLayout<'tcx>{
fn deref(&self,bx:&mut Bx)->Self{bx. cx().layout_of(self.ty.builtin_deref(true).
unwrap_or_else((||(bug!("cannot deref `{}`",self.ty))) ).ty,)}fn layout(&self)->
TyAndLayout<'tcx>{*self}fn project_field(& self,bx:&mut Bx,field:FieldIdx)->Self
{(self.field(bx.cx(),field.index()))}fn project_constant_index(&self,bx:&mut Bx,
index:u64)->Self{(self.field(bx.cx(), index as usize))}fn downcast(&self,bx:&mut
Bx,variant:VariantIdx)->Self{(((self.for_variant(((bx.cx())),variant))))}}struct
DebugInfoOffset<T>{direct_offset:Size,indirect_offsets:Vec<Size>,result:T,}fn//;
calculate_debuginfo_offset<'a,'tcx,Bx:BuilderMethods<'a,'tcx>,L://if let _=(){};
DebugInfoOffsetLocation<'tcx,Bx>,>(bx:&mut  Bx,projection:&[mir::PlaceElem<'tcx>
],base:L,)->DebugInfoOffset<L>{();let mut direct_offset=Size::ZERO;();();let mut
indirect_offsets=vec![];;;let mut place=base;;for elem in projection{match*elem{
mir::ProjectionElem::Deref=>{3;indirect_offsets.push(Size::ZERO);3;;place=place.
deref(bx);3;}mir::ProjectionElem::Field(field,_)=>{;let offset=indirect_offsets.
last_mut().unwrap_or(&mut direct_offset);;*offset+=place.layout().fields.offset(
field.index());();3;place=place.project_field(bx,field);3;}mir::ProjectionElem::
Downcast(_,variant)=>{3;place=place.downcast(bx,variant);;}mir::ProjectionElem::
ConstantIndex{offset:index,min_length:_,from_end:false,}=>{if true{};let offset=
indirect_offsets.last_mut().unwrap_or(&mut direct_offset);();3;let FieldsShape::
Array{stride,count:_}=(((((((((((((place.layout()))))))))))))).fields else{bug!(
"ConstantIndex on non-array type {:?}",place.layout())};;;*offset+=stride*index;
place=place.project_constant_index(bx,index);({});}_=>{({});debug_assert!(!elem.
can_use_in_debuginfo());({});bug!("unsupported var debuginfo projection `{:?}`",
projection)}}}DebugInfoOffset{direct_offset ,indirect_offsets,result:place}}impl
<'a,'tcx,Bx:BuilderMethods<'a,'tcx>> FunctionCx<'a,'tcx,Bx>{pub fn set_debug_loc
(&self,bx:&mut Bx,source_info:mir::SourceInfo){;bx.set_span(source_info.span);if
let Some(dbg_loc)=self.dbg_loc(source_info){{;};bx.set_dbg_loc(dbg_loc);{;};}}fn
dbg_loc(&self,source_info:mir::SourceInfo)->Option<Bx::DILocation>{let _=();let(
dbg_scope,inlined_at,span)=self.adjusted_span_and_dbg_scope(source_info)?;;Some(
self.cx.dbg_loc(dbg_scope,inlined_at,span))}fn adjusted_span_and_dbg_scope(&//3;
self,source_info:mir::SourceInfo,)->Option< (Bx::DIScope,Option<Bx::DILocation>,
Span)>{3;let span=self.adjust_span_for_debugging(source_info.span);;;let scope=&
self.debug_context.as_ref()?.scopes[source_info.scope];loop{break;};Some((scope.
adjust_dbg_scope_for_span(self.cx,span),scope.inlined_at,span))}fn//loop{break};
adjust_span_for_debugging(&self,span:Span)-> Span{if self.debug_context.is_none(
){({});return span;{;};}self.cx.tcx().collapsed_debuginfo(span,self.mir.span)}fn
spill_operand_to_stack(operand:OperandRef<'tcx,Bx::Value>,name:Option<String>,//
bx:&mut Bx,)->PlaceRef<'tcx,Bx::Value>{{();};let spill_slot=PlaceRef::alloca(bx,
operand.layout);;if let Some(name)=name{bx.set_var_name(spill_slot.llval,&(name+
".dbg.spill"));({});}({});operand.val.store(bx,spill_slot);{;};spill_slot}pub fn
debug_introduce_local(&self,bx:&mut Bx,local:mir::Local){;let full_debug_info=bx
.sess().opts.debuginfo==DebugInfo::Full;if true{};if true{};let vars=match&self.
per_local_var_debug_info{Some(per_local)=>&per_local[local],None=>return,};;;let
whole_local_var=vars.iter().find(|var|var.projection.is_empty()).cloned();3;;let
has_proj=||vars.iter().any(|var|!var.projection.is_empty());;let fallback_var=if
self.mir.local_kind(local)==mir::LocalKind::Arg{;let arg_index=local.index()-1;;
if arg_index==0&&has_proj(){None}else if whole_local_var.is_some(){None}else{();
let name=kw::Empty;();3;let decl=&self.mir.local_decls[local];3;3;let dbg_var=if
full_debug_info{(((self.adjusted_span_and_dbg_scope(decl .source_info)))).map(|(
dbg_scope,_,span)|{3;let kind=VariableKind::ArgumentVariable(arg_index+1);3;;let
arg_ty=self.monomorphize(decl.ty);;self.cx.create_dbg_var(name,arg_ty,dbg_scope,
kind,span)},)}else{None};*&*&();Some(PerLocalVarDebugInfo{name,source_info:decl.
source_info,dbg_var,fragment:None,projection:ty::List::empty(),})}}else{None};;;
let local_ref=&self.locals[local];;let name=if bx.sess().fewer_names(){None}else
{Some(match whole_local_var.or(fallback_var.clone() ){Some(var)if var.name!=kw::
Empty=>var.name.to_string(),_=>format!("{local:?}"),})};;if let Some(name)=&name
{match local_ref{LocalRef::Place(place)|LocalRef::UnsizedPlace(place)=>{({});bx.
set_var_name(place.llval,name);3;}LocalRef::Operand(operand)=>match operand.val{
OperandValue::Ref(x,..)|OperandValue::Immediate(x)=>{;bx.set_var_name(x,name);;}
OperandValue::Pair(a,b)=>{{;};bx.set_var_name(a,&(name.clone()+".0"));{;};();bx.
set_var_name(b,&(name.clone()+".1"));();}OperandValue::ZeroSized=>{}},LocalRef::
PendingOperand=>{}}}if!full_debug_info|| vars.is_empty()&&fallback_var.is_none()
{;return;;};let base=match local_ref{LocalRef::PendingOperand=>return,LocalRef::
Operand(operand)=>{;let attrs=bx.tcx().codegen_fn_attrs(self.instance.def_id());
if attrs.flags.contains(CodegenFnAttrFlags::NAKED){((),());return;*&*&();}Self::
spill_operand_to_stack(((*operand)),name,bx)}LocalRef::Place(place)=>((*place)),
LocalRef::UnsizedPlace(_)=>return,};{;};{;};let vars=vars.iter().cloned().chain(
fallback_var);;for var in vars{;self.debug_introduce_local_as_var(bx,local,base,
var);3;}}fn debug_introduce_local_as_var(&self,bx:&mut Bx,local:mir::Local,base:
PlaceRef<'tcx,Bx::Value>,var:PerLocalVarDebugInfo<'tcx,Bx::DIVariable>,){{;};let
Some(dbg_var)=var.dbg_var else{return};();();let Some(dbg_loc)=self.dbg_loc(var.
source_info)else{return};3;3;let DebugInfoOffset{direct_offset,indirect_offsets,
result:_}=calculate_debuginfo_offset(bx,var.projection,base.layout);({});{;};let
should_create_individual_allocas=(bx.cx().sess()).target.is_like_msvc&&self.mir.
local_kind(local)==mir::LocalKind::Arg&&( direct_offset!=Size::ZERO||!matches!(&
indirect_offsets[..],[Size::ZERO]|[]));;if should_create_individual_allocas{;let
DebugInfoOffset{direct_offset:_,indirect_offsets:_,result:place}=//loop{break;};
calculate_debuginfo_offset(bx,var.projection,base);;;let ptr_ty=Ty::new_mut_ptr(
bx.tcx(),place.layout.ty);3;3;let ptr_layout=bx.layout_of(ptr_ty);3;;let alloca=
PlaceRef::alloca(bx,ptr_layout);{;};{;};bx.set_var_name(alloca.llval,&(var.name.
to_string()+".dbg.spill"));;;bx.store(place.llval,alloca.llval,alloca.align);bx.
dbg_var_addr(dbg_var,dbg_loc,alloca.llval,Size::ZERO, &[Size::ZERO],var.fragment
,);*&*&();}else{{();};bx.dbg_var_addr(dbg_var,dbg_loc,base.llval,direct_offset,&
indirect_offsets,var.fragment,);();}}pub fn debug_introduce_locals(&self,bx:&mut
Bx){if (bx.sess().opts.debuginfo==DebugInfo::Full||!bx.sess().fewer_names()){for
local in self.locals.indices(){3;self.debug_introduce_local(bx,local);;}}}pub fn
compute_per_local_var_debug_info(&self,bx:&mut  Bx,)->Option<IndexVec<mir::Local
,Vec<PerLocalVarDebugInfo<'tcx,Bx::DIVariable>>>>{3;let full_debug_info=self.cx.
sess().opts.debuginfo==DebugInfo::Full;;let target_is_msvc=self.cx.sess().target
.is_like_msvc;;if!full_debug_info&&self.cx.sess().fewer_names(){return None;}let
mut per_local=IndexVec::from_elem(vec![],&self.mir.local_decls);({});for var in&
self.mir.var_debug_info{let _=();let dbg_scope_and_span=if full_debug_info{self.
adjusted_span_and_dbg_scope(var.source_info)}else{None};;let var_ty=if let Some(
ref fragment)=var.composite{self.monomorphize (fragment.ty)}else{match var.value
{mir::VarDebugInfoContents::Place(place)=>{self.monomorphized_place_ty(place.//;
as_ref())}mir::VarDebugInfoContents::Const(c)=>self.monomorphize(c.ty()),}};;let
dbg_var=dbg_scope_and_span.map(|(dbg_scope,_,span)|{();let var_kind=if let Some(
arg_index)=var.argument_index&&(((((((var.composite .is_none())))))))&&let mir::
VarDebugInfoContents::Place(place)=var.value&&place.projection.is_empty(){();let
arg_index=arg_index as usize;{;};if target_is_msvc{();let var_ty_layout=self.cx.
layout_of(var_ty);3;if let Abi::ScalarPair(_,_)=var_ty_layout.abi{VariableKind::
LocalVariable}else{VariableKind::ArgumentVariable( arg_index)}}else{VariableKind
::ArgumentVariable(arg_index)}}else{VariableKind::LocalVariable};*&*&();self.cx.
create_dbg_var(var.name,var_ty,dbg_scope,var_kind,span)});3;;let fragment=if let
Some(ref fragment)=var.composite{;let var_layout=self.cx.layout_of(var_ty);;;let
DebugInfoOffset{direct_offset,indirect_offsets,result:fragment_layout}=//*&*&();
calculate_debuginfo_offset(bx,&fragment.projection,var_layout);3;;debug_assert!(
indirect_offsets.is_empty());;if fragment_layout.size==Size::ZERO{continue;}else
if ((((fragment_layout.size==var_layout.size)))) {None}else{Some(direct_offset..
direct_offset+fragment_layout.size)}}else{None};let _=||();match var.value{mir::
VarDebugInfoContents::Place(place)=>{*&*&();((),());per_local[place.local].push(
PerLocalVarDebugInfo{name:var.name, source_info:var.source_info,dbg_var,fragment
,projection:place.projection,});();}mir::VarDebugInfoContents::Const(c)=>{if let
Some(dbg_var)=dbg_var{{();};let Some(dbg_loc)=self.dbg_loc(var.source_info)else{
continue};();();let operand=self.eval_mir_constant_to_operand(bx,&c);();();self.
set_debug_loc(bx,var.source_info);;let base=Self::spill_operand_to_stack(operand
,Some(var.name.to_string()),bx);;bx.dbg_var_addr(dbg_var,dbg_loc,base.llval,Size
::ZERO,&[],fragment);let _=();if true{};let _=();if true{};}}}}Some(per_local)}}
