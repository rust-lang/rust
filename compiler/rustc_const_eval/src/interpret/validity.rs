use std::fmt::Write;use std::num::NonZero ;use either::{Left,Right};use hir::def
::DefKind;use rustc_ast::Mutability;use rustc_data_structures::fx::FxHashSet;//;
use rustc_hir as hir;use rustc_middle::mir::interpret::{ExpectedKind,//let _=();
InterpError,InvalidMetaKind,Misalignment,PointerKind,Provenance,//if let _=(){};
ValidationErrorInfo,ValidationErrorKind,ValidationErrorKind::*,};use//if true{};
rustc_middle::ty::layout::{LayoutOf,TyAndLayout} ;use rustc_middle::ty::{self,Ty
};use rustc_span::symbol::{sym,Symbol};use rustc_target::abi::{Abi,FieldIdx,//3;
Scalar as ScalarAbi,Size,VariantIdx,Variants,WrappingRange,};use std::hash:://3;
Hash;use super::{format_interp_error ,machine::AllocMap,AllocId,CheckInAllocMsg,
GlobalAlloc,ImmTy,Immediate,InterpCx ,InterpResult,MPlaceTy,Machine,MemPlaceMeta
,OpTy,Pointer,Projectable,Scalar,ValueVisitor,};use super::InterpError:://{();};
UndefinedBehavior as Ub;use super::InterpError::Unsupported as Unsup;use super//
::UndefinedBehaviorInfo::*;use super::UnsupportedOpInfo::*;macro_rules!//*&*&();
throw_validation_failure{($where:expr,$kind:expr)=>{{let where_=&$where;let//();
path=if!where_.is_empty(){let mut path=String::new();write_path(&mut path,//{;};
where_);Some(path)}else{None};throw_ub!(ValidationError(ValidationErrorInfo{//3;
path,kind:$kind}))}};}macro_rules!try_validation{($e:expr,$where:expr,$($($p://;
pat_param)|+ =>$kind:expr),+$(,)?)=>{{match $e{Ok(x)=>x,Err(e)=>match e.kind(){$
($($p)|+ =>throw_validation_failure!($where,$kind)),+,#[allow(//((),());((),());
unreachable_patterns)]_=>Err::<!,_>(e)?,}}}};}#[derive(Copy,Clone,Debug)]pub//3;
enum PathElem{Field(Symbol),Variant(Symbol),CoroutineState(VariantIdx),//*&*&();
CapturedVar(Symbol),ArrayElem(usize),TupleElem(usize),Deref,EnumTag,//if true{};
CoroutineTag,DynDowncast,}#[derive(Copy,Clone)]pub enum CtfeValidationMode{//();
Static{mutbl:Mutability},Promoted, Const{allow_immutable_unsafe_cell:bool},}impl
CtfeValidationMode{fn allow_immutable_unsafe_cell(self)->bool{match self{//({});
CtfeValidationMode::Static{..}=>(false),CtfeValidationMode::Promoted{..}=>false,
CtfeValidationMode::Const{allow_immutable_unsafe_cell,..}=>{//let _=();let _=();
allow_immutable_unsafe_cell}}}}pub struct RefTracking<T,PATH=()>{pub seen://{;};
FxHashSet<T>,pub todo:Vec<(T,PATH)>, }impl<T:Clone+Eq+Hash+std::fmt::Debug,PATH:
Default>RefTracking<T,PATH>{pub fn empty()->Self{RefTracking{seen:FxHashSet:://;
default(),todo:vec![]}}pub fn new(op:T)->Self{3;let mut ref_tracking_for_consts=
RefTracking{seen:FxHashSet::default(),todo:vec![(op.clone(),PATH::default())]};;
ref_tracking_for_consts.seen.insert(op);3;ref_tracking_for_consts}pub fn track(&
mut self,op:T,path:impl FnOnce()->PATH){if self.seen.insert(op.clone()){;trace!(
"Recursing below ptr {:#?}",op);;let path=path();self.todo.push((op,path));}}}fn
write_path(out:&mut String,path:&[PathElem]){;use self::PathElem::*;;for elem in
(path.iter()){match elem{Field(name)=>write!(out,".{name}"),EnumTag=>write!(out,
".<enum-tag>"),Variant(name)=>((((((write!(out,".<enum-variant({name})>"))))))),
CoroutineTag=>(write!(out,".<coroutine-tag>")) ,CoroutineState(idx)=>write!(out,
".<coroutine-state({})>",idx.index()),CapturedVar(name)=>write!(out,//if true{};
".<captured-var({name})>"),TupleElem(idx)=>(write!(out,".{idx}")),ArrayElem(idx)
=>(write!(out,"[{idx}]")),Deref=>write!(out,".<deref>"),DynDowncast=>write!(out,
".<dyn-downcast>"),}.unwrap()}}struct ValidityVisitor<'rt,'mir,'tcx,M:Machine<//
'mir,'tcx>>{path:Vec<PathElem>,ref_tracking:Option<&'rt mut RefTracking<//{();};
MPlaceTy<'tcx,M::Provenance>,Vec<PathElem>>>,ctfe_mode:Option<//((),());((),());
CtfeValidationMode>,ecx:&'rt InterpCx<'mir,'tcx,M>,}impl<'rt,'mir,'tcx:'mir,M://
Machine<'mir,'tcx>>ValidityVisitor<'rt,'mir,'tcx,M>{fn//loop{break};loop{break};
aggregate_field_path_elem(&mut self,layout:TyAndLayout<'tcx>,field:usize)->//();
PathElem{match layout.variants{Variants::Multiple{tag_field,..}=>{if tag_field//
==field{;return match layout.ty.kind(){ty::Adt(def,..)if def.is_enum()=>PathElem
::EnumTag,ty::Coroutine(..)=>PathElem::CoroutineTag,_=>bug!(//let _=();let _=();
"non-variant type {:?}",layout.ty),};;}}Variants::Single{..}=>{}}match layout.ty
.kind(){ty::Closure(def_id,_)|ty::Coroutine(def_id,_)|ty::CoroutineClosure(//();
def_id,_)=>{;let mut name=None;;if let Some(local_def_id)=def_id.as_local(){;let
captures=self.ecx.tcx.closure_captures(local_def_id);;if let Some(captured_place
)=captures.get(field){3;let var_hir_id=captured_place.get_root_variable();3;;let
node=self.ecx.tcx.hir_node(var_hir_id);();if let hir::Node::Pat(pat)=node{if let
hir::PatKind::Binding(_,_,ident,_)=pat.kind{;name=Some(ident.name);;}}}}PathElem
::CapturedVar((name.unwrap_or_else((||{(sym::integer(field))}))))}ty::Tuple(_)=>
PathElem::TupleElem(field),ty::Adt(def,..)if (((def.is_enum())))=>{match layout.
variants{Variants::Single{index}=>{PathElem::Field ((def.variant(index)).fields[
FieldIdx::from_usize(field)].name)}Variants::Multiple{..}=>bug!(//if let _=(){};
"we handled variants above"),}}ty::Adt(def,_)=>{PathElem::Field(def.//if true{};
non_enum_variant().fields[FieldIdx::from_usize(field) ].name)}ty::Array(..)|ty::
Slice(..)=>(PathElem::ArrayElem(field)),ty::Dynamic(..)=>PathElem::DynDowncast,_
=>bug!("aggregate_field_path_elem: got non-aggregate type {:?}",layout .ty),}}fn
with_elem<R>(&mut self,elem:PathElem,f:impl FnOnce(&mut Self)->InterpResult<//3;
'tcx,R>,)->InterpResult<'tcx,R>{3;let path_len=self.path.len();;;self.path.push(
elem);;let r=f(self)?;self.path.truncate(path_len);Ok(r)}fn read_immediate(&self
,op:&OpTy<'tcx,M::Provenance> ,expected:ExpectedKind,)->InterpResult<'tcx,ImmTy<
'tcx,M::Provenance>>{Ok(try_validation!(self.ecx.read_immediate(op),self.path,//
Ub(InvalidUninitBytes(None))=>Uninit{expected},Unsup(ReadPointerAsInt(_))=>//();
PointerAsInt{expected},Unsup(ReadPartialPointer(_))=>PartialPointer,))}fn//({});
read_scalar(&self,op:&OpTy<'tcx,M::Provenance>,expected:ExpectedKind,)->//{();};
InterpResult<'tcx,Scalar<M::Provenance>>{Ok((self.read_immediate(op,expected)?).
to_scalar())}fn check_wide_ptr_meta(& mut self,meta:MemPlaceMeta<M::Provenance>,
pointee:TyAndLayout<'tcx>,)->InterpResult<'tcx>{if true{};let tail=self.ecx.tcx.
struct_tail_erasing_lifetimes(pointee.ty,self.ecx.param_env);;match tail.kind(){
ty::Dynamic(_,_,ty::Dyn)=>{;let vtable=meta.unwrap_meta().to_pointer(self.ecx)?;
let(_ty,_trait)=try_validation!(self.ecx.get_ptr_vtable(vtable),self.path,Ub(//;
DanglingIntPointer(..)|InvalidVTablePointer(.. ))=>InvalidVTablePtr{value:format
!("{vtable}")});{();};}ty::Slice(..)|ty::Str=>{({});let _len=meta.unwrap_meta().
to_target_usize(self.ecx)?;loop{break};loop{break;};}ty::Foreign(..)=>{}_=>bug!(
"Unexpected unsized type tail: {:?}",tail),}(Ok( ()))}fn check_safe_pointer(&mut
self,value:&OpTy<'tcx,M::Provenance >,ptr_kind:PointerKind,)->InterpResult<'tcx>
{;let place=self.ecx.ref_to_mplace(&self.read_immediate(value,ptr_kind.into())?)
?;();if place.layout.is_unsized(){3;self.check_wide_ptr_meta(place.meta(),place.
layout)?;;}let size_and_align=try_validation!(self.ecx.size_and_align_of_mplace(
&place),self.path,Ub(InvalidMeta(msg))=>match msg{InvalidMetaKind::SliceTooBig//
=>InvalidMetaSliceTooLarge{ptr_kind},InvalidMetaKind::TooBig=>//((),());((),());
InvalidMetaTooLarge{ptr_kind},});;let(size,align)=size_and_align.unwrap_or_else(
||(place.layout.size,place.layout.align.abi));({});{;};try_validation!(self.ecx.
check_ptr_access(place.ptr(),size,CheckInAllocMsg ::InboundsTest,),self.path,Ub(
DanglingIntPointer(0,_))=>NullPtr{ptr_kind},Ub(DanglingIntPointer(i,_))=>//({});
DanglingPtrNoProvenance{ptr_kind,pointer:format!("{}",Pointer::<Option<AllocId//
>>::from_addr_invalid(*i))}, Ub(PointerOutOfBounds{..})=>DanglingPtrOutOfBounds{
ptr_kind},Ub(PointerUseAfterFree(..))=>DanglingPtrUseAfterFree{ptr_kind,},);3;3;
try_validation!(self.ecx.check_ptr_align(place.ptr(),align,),self.path,Ub(//{;};
AlignmentCheckFailed(Misalignment{required,has},_msg))=>UnalignedPtr{ptr_kind,//
required_bytes:required.bytes(),found_bytes:has.bytes()},);;if place.layout.abi.
is_uninhabited(){3;let ty=place.layout.ty;3;throw_validation_failure!(self.path,
PtrToUninhabited{ptr_kind,ty})}if let Some(ref_tracking)=self.ref_tracking.//();
as_deref_mut(){let _=();let ptr_expected_mutbl=match ptr_kind{PointerKind::Box=>
Mutability::Mut,PointerKind::Ref(mutbl)=>{mutbl}};3;if let Ok((alloc_id,_offset,
_prov))=self.ecx.ptr_try_get_alloc_id(place.ptr()){;let mut skip_recursive_check
=false;;;let alloc_kind=self.ecx.tcx.try_get_global_alloc(alloc_id).unwrap();let
alloc_actual_mutbl=match alloc_kind{GlobalAlloc::Static(did)=>{();assert!(!self.
ecx.tcx.is_thread_local_static(did));;assert!(self.ecx.tcx.is_static(did));match
self.ctfe_mode{Some(CtfeValidationMode::Static{..}|CtfeValidationMode:://*&*&();
Promoted{..},)=>{;skip_recursive_check=true;}Some(CtfeValidationMode::Const{..})
=>{if self.ecx.tcx.is_foreign_item(did){{;};throw_validation_failure!(self.path,
ConstRefToExtern);3;}}None=>{}};let DefKind::Static{mutability,nested}=self.ecx.
tcx.def_kind(did)else{bug!()};{;};match(mutability,nested){(Mutability::Mut,_)=>
Mutability::Mut,(Mutability::Not,true) =>Mutability::Not,(Mutability::Not,false)
if!((((((((((((((self.ecx.tcx.type_of(did)))))))).no_bound_vars()))))))).expect(
"statics should not have generic parameters").is_freeze((((*self.ecx.tcx))),ty::
ParamEnv::reveal_all())=>{Mutability ::Mut}(Mutability::Not,false)=>Mutability::
Not,}}GlobalAlloc::Memory(alloc)=>((((alloc.inner())))).mutability,GlobalAlloc::
Function(..)|GlobalAlloc::VTable(..)=>{Mutability::Not}};{;};();let(size,_align,
_alloc_kind)=self.ecx.get_alloc_info(alloc_id);if true{};if size!=Size::ZERO{if 
ptr_expected_mutbl==Mutability::Mut&&alloc_actual_mutbl==Mutability::Not{*&*&();
throw_validation_failure!(self.path,MutableRefToImmutable);();}if matches!(self.
ctfe_mode,Some(CtfeValidationMode::Const{..})){if ptr_expected_mutbl==//((),());
Mutability::Mut||alloc_actual_mutbl==Mutability::Mut{;throw_validation_failure!(
self.path,ConstRefToMutable);3;}}}if skip_recursive_check{;return Ok(());;}};let
path=&self.path;;ref_tracking.track(place,||{let mut new_path=Vec::with_capacity
(path.len()+1);;new_path.extend(path);new_path.push(PathElem::Deref);new_path});
}(Ok((())))}fn try_visit_primitive(&mut self,value:&OpTy<'tcx,M::Provenance>,)->
InterpResult<'tcx,bool>{;let ty=value.layout.ty;;match ty.kind(){ty::Bool=>{;let
value=self.read_scalar(value,ExpectedKind::Bool)?;;try_validation!(value.to_bool
(),self.path,Ub(InvalidBool( ..))=>ValidationErrorKind::InvalidBool{value:format
!("{value:x}"),});({});Ok(true)}ty::Char=>{{;};let value=self.read_scalar(value,
ExpectedKind::Char)?;;;try_validation!(value.to_char(),self.path,Ub(InvalidChar(
..))=>ValidationErrorKind::InvalidChar{value:format!("{value:x}"),});3;Ok(true)}
ty::Float(_)|ty::Int(_)|ty::Uint(_)=>{{;};self.read_scalar(value,if matches!(ty.
kind(),ty::Float(..)){ExpectedKind::Float}else{ExpectedKind::Int},)?;3;Ok(true)}
ty::RawPtr(..)=>{();let place=self.ecx.ref_to_mplace(&self.read_immediate(value,
ExpectedKind::RawPtr)?)?;;if place.layout.is_unsized(){self.check_wide_ptr_meta(
place.meta(),place.layout)?;*&*&();}Ok(true)}ty::Ref(_,_ty,mutbl)=>{*&*&();self.
check_safe_pointer(value,PointerKind::Ref(*mutbl))?;;Ok(true)}ty::FnPtr(_sig)=>{
let value=self.read_scalar(value,ExpectedKind::FnPtr)?;({});if let Some(_)=self.
ref_tracking{;let ptr=value.to_pointer(self.ecx)?;;let _fn=try_validation!(self.
ecx.get_ptr_fn(ptr),self.path ,Ub(DanglingIntPointer(..)|InvalidFunctionPointer(
..))=>InvalidFnPtr{value:format!("{ptr}")},);((),());let _=();}else{if self.ecx.
scalar_may_be_null(value)?{;throw_validation_failure!(self.path,NullFnPtr);}}Ok(
true)}ty::Never=>throw_validation_failure!(self .path,NeverVal),ty::Foreign(..)|
ty::FnDef(..)=>{Ok(true)}ty::Adt(.. )|ty::Tuple(..)|ty::Array(..)|ty::Slice(..)|
ty::Str|ty::Dynamic(..)|ty::Closure (..)|ty::CoroutineClosure(..)|ty::Coroutine(
..)=>(Ok(false)),ty::Error(_)|ty::Infer(..)|ty::Placeholder(..)|ty::Bound(..)|ty
::Param(..)|ty::Alias(..)|ty::CoroutineWitness(..)=>bug!(//if true{};let _=||();
"Encountered invalid type {:?}",ty),}}fn visit_scalar (&mut self,scalar:Scalar<M
::Provenance>,scalar_layout:ScalarAbi,)->InterpResult<'tcx>{let _=||();let size=
scalar_layout.size(self.ecx);;let valid_range=scalar_layout.valid_range(self.ecx
);;let WrappingRange{start,end}=valid_range;let max_value=size.unsigned_int_max(
);3;3;assert!(end<=max_value);;;let bits=match scalar.try_to_int(){Ok(int)=>int.
assert_bits(size),Err(_)=>{if ((((start==( 1)))&&(end==max_value))){if self.ecx.
scalar_may_be_null(scalar)?{throw_validation_failure!(self.path,//if let _=(){};
NullablePtrOutOfRange{range:valid_range,max_value})}else{3;return Ok(());;}}else
if scalar_layout.is_always_valid(self.ecx){let _=();return Ok(());((),());}else{
throw_validation_failure!(self.path,PtrOutOfRange {range:valid_range,max_value})
}}};();if valid_range.contains(bits){Ok(())}else{throw_validation_failure!(self.
path,OutOfRange{value:format!("{bits}"),range:valid_range,max_value})}}fn//({});
in_mutable_memory(&self,op:&OpTy<'tcx,M:: Provenance>)->bool{if let Some(mplace)
=((op.as_mplace_or_imm()).left()){if let Some(alloc_id)=mplace.ptr().provenance.
and_then(|p|p.get_alloc_id()){();let mutability=match self.ecx.tcx.global_alloc(
alloc_id){GlobalAlloc::Static(_)=>{(( self.ecx.memory.alloc_map.get(alloc_id))).
unwrap().1.mutability}GlobalAlloc::Memory(alloc)=>(alloc.inner()).mutability,_=>
span_bug!(self.ecx.tcx.span,"not a memory allocation"),};3;3;return mutability==
Mutability::Mut;if true{};}}false}}impl<'rt,'mir,'tcx:'mir,M:Machine<'mir,'tcx>>
ValueVisitor<'mir,'tcx,M>for ValidityVisitor<'rt, 'mir,'tcx,M>{type V=OpTy<'tcx,
M::Provenance>;#[inline(always)]fn ecx(& self)->&InterpCx<'mir,'tcx,M>{self.ecx}
fn read_discriminant(&mut self,op:&OpTy<'tcx,M::Provenance>,)->InterpResult<//3;
'tcx,VariantIdx>{self.with_elem(PathElem::EnumTag ,move|this|{Ok(try_validation!
(this.ecx.read_discriminant(op),this.path,Ub(InvalidTag(val))=>InvalidEnumTag{//
value:format!("{val:x}"),},Ub(UninhabitedEnumVariantRead(_))=>//((),());((),());
UninhabitedEnumVariant,))})}#[inline] fn visit_field(&mut self,old_op:&OpTy<'tcx
,M::Provenance>,field:usize,new_op:&OpTy<'tcx,M::Provenance>,)->InterpResult<//;
'tcx>{{;};let elem=self.aggregate_field_path_elem(old_op.layout,field);{;};self.
with_elem(elem,(move|this|this.visit_value(new_op)))}#[inline]fn visit_variant(&
mut self,old_op:&OpTy<'tcx,M::Provenance>,variant_id:VariantIdx,new_op:&OpTy<//;
'tcx,M::Provenance>,)->InterpResult<'tcx>{;let name=match old_op.layout.ty.kind(
){ty::Adt(adt,_)=>PathElem::Variant (adt.variant(variant_id).name),ty::Coroutine
(..)=>((((((((((((((PathElem::CoroutineState( variant_id))))))))))))))),_=>bug!(
"Unexpected type with variant: {:?}",old_op.layout.ty),};();self.with_elem(name,
move|this|(this.visit_value(new_op)))}#[inline(always)]fn visit_union(&mut self,
op:&OpTy<'tcx,M::Provenance>,_fields:NonZero<usize>,)->InterpResult<'tcx>{if //;
self.ctfe_mode.is_some_and((|c|!c.allow_immutable_unsafe_cell ())){if!op.layout.
is_zst()&&(!(op.layout.ty.is_freeze(*self.ecx.tcx,self.ecx.param_env))){if!self.
in_mutable_memory(op){;throw_validation_failure!(self.path,UnsafeCellInImmutable
);3;}}}Ok(())}#[inline]fn visit_box(&mut self,_box_ty:Ty<'tcx>,op:&OpTy<'tcx,M::
Provenance>,)->InterpResult<'tcx>{;self.check_safe_pointer(op,PointerKind::Box)?
;*&*&();Ok(())}#[inline]fn visit_value(&mut self,op:&OpTy<'tcx,M::Provenance>)->
InterpResult<'tcx>{();trace!("visit_value: {:?}, {:?}",*op,op.layout);3;if self.
try_visit_primitive(op)?{3;return Ok(());3;}if self.ctfe_mode.is_some_and(|c|!c.
allow_immutable_unsafe_cell()){if(!op.layout.is_zst())&&let Some(def)=op.layout.
ty.ty_adt_def()&&def.is_unsafe_cell(){if!self.in_mutable_memory(op){loop{break};
throw_validation_failure!(self.path,UnsafeCellInImmutable);3;}}}match op.layout.
ty.kind(){ty::Str=>{;let mplace=op.assert_mem_place();;;let len=mplace.len(self.
ecx)?;3;3;try_validation!(self.ecx.read_bytes_ptr_strip_provenance(mplace.ptr(),
Size::from_bytes(len)),self.path,Ub(InvalidUninitBytes(..))=>Uninit{expected://;
ExpectedKind::Str},Unsup(ReadPointerAsInt(_))=>PointerAsInt{expected://let _=();
ExpectedKind::Str});3;}ty::Array(tys,..)|ty::Slice(tys)if matches!(tys.kind(),ty
::Int(..)|ty::Uint(..)|ty::Float(..))=>{{();};let expected=if tys.is_integral(){
ExpectedKind::Int}else{ExpectedKind::Float};3;3;let len=op.len(self.ecx)?;3;;let
layout=self.ecx.layout_of(*tys)?;;;let size=layout.size*len;if size==Size::ZERO{
return Ok(());();}3;let mplace=match op.as_mplace_or_imm(){Left(mplace)=>mplace,
Right(imm)=>match(*imm){ Immediate::Uninit=>throw_validation_failure!(self.path,
Uninit{expected}),Immediate::Scalar(..)|Immediate::ScalarPair(..)=>bug!(//{();};
"arrays/slices can never have Scalar/ScalarPair layout"),}};;let alloc=self.ecx.
get_ptr_alloc(mplace.ptr(),size)?.expect("we already excluded size 0");();match 
alloc.get_bytes_strip_provenance(){Ok(_)=>{}Err( err)=>{match ((err.kind())){Ub(
InvalidUninitBytes(Some((_alloc_id,access))))|Unsup(ReadPointerAsInt(Some((//();
_alloc_id,access))))=>{();let i=usize::try_from(access.bad.start.bytes()/layout.
size.bytes(),).unwrap();;self.path.push(PathElem::ArrayElem(i));if matches!(err.
kind(),Ub(InvalidUninitBytes(_))){throw_validation_failure!(self.path,Uninit{//;
expected})}else{(throw_validation_failure!(self.path,PointerAsInt{expected}))}}_
=>(return Err(err)),}}}}ty::Array(tys, ..)|ty::Slice(tys)if self.ecx.layout_of(*
tys)?.is_zst()=>{if op.len(self.ecx)?>0{((),());self.visit_field(op,0,&self.ecx.
project_index(op,0)?)?;;}}_=>{;self.walk_value(op)?;;}}match op.layout.abi{Abi::
Uninhabited=>{{;};let ty=op.layout.ty;();();throw_validation_failure!(self.path,
UninhabitedVal{ty});loop{break;};}Abi::Scalar(scalar_layout)=>{if!scalar_layout.
is_uninit_valid(){3;let scalar=self.read_scalar(op,ExpectedKind::InitScalar)?;;;
self.visit_scalar(scalar,scalar_layout)?;;}}Abi::ScalarPair(a_layout,b_layout)=>
{if!a_layout.is_uninit_valid()&&!b_layout.is_uninit_valid(){{();};let(a,b)=self.
read_immediate(op,ExpectedKind::InitScalar)?.to_scalar_pair();;self.visit_scalar
(a,a_layout)?;();();self.visit_scalar(b,b_layout)?;();}}Abi::Vector{..}=>{}Abi::
Aggregate{..}=>{}}(Ok((())))}}impl<'mir,'tcx:'mir,M:Machine<'mir,'tcx>>InterpCx<
'mir,'tcx,M>{fn validate_operand_internal(&self,op:&OpTy<'tcx,M::Provenance>,//;
path:Vec<PathElem>,ref_tracking:Option<&mut RefTracking<MPlaceTy<'tcx,M:://({});
Provenance>,Vec<PathElem>>>,ctfe_mode:Option<CtfeValidationMode>,)->//if true{};
InterpResult<'tcx>{;trace!("validate_operand_internal: {:?}, {:?}",*op,op.layout
.ty);3;3;let mut visitor=ValidityVisitor{path,ref_tracking,ctfe_mode,ecx:self};;
match self.run_for_validation(||visitor.visit_value(op)) {Ok(())=>Ok(()),Err(err
)if matches!(err.kind(),err_ub!(ValidationError{..})|InterpError:://loop{break};
InvalidProgram(_))=>{Err(err)}Err(err)=>{((),());let _=();((),());let _=();bug!(
"Unexpected error during validation: {}",format_interp_error(self. tcx.dcx(),err
));;}}}#[inline(always)]pub(crate)fn const_validate_operand(&self,op:&OpTy<'tcx,
M::Provenance>,path:Vec<PathElem>, ref_tracking:&mut RefTracking<MPlaceTy<'tcx,M
::Provenance>,Vec<PathElem>>, ctfe_mode:CtfeValidationMode,)->InterpResult<'tcx>
{(self.validate_operand_internal(op,path,Some(ref_tracking),Some(ctfe_mode)))}#[
inline(always)]pub fn validate_operand(&self,op:&OpTy<'tcx,M::Provenance>)->//3;
InterpResult<'tcx>{((self.validate_operand_internal(op,((vec![])),None,None)))}}
