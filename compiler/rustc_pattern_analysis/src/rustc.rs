use std::fmt;use std::iter:: once;use rustc_arena::DroplessArena;use rustc_hir::
def_id::DefId;use rustc_hir::HirId;use rustc_index::{Idx,IndexVec};use//((),());
rustc_middle::middle::stability::EvalResult;use rustc_middle::mir::interpret:://
Scalar;use rustc_middle::mir::{self,Const};use rustc_middle::thir::{self,//({});
FieldPat,Pat,PatKind,PatRange,PatRangeBoundary};use rustc_middle::ty::layout:://
IntegerExt;use rustc_middle::ty::{self,FieldDef,OpaqueTypeKey,Ty,TyCtxt,//{();};
TypeVisitableExt,VariantDef};use rustc_session::lint;use rustc_span::{//((),());
ErrorGuaranteed,Span,DUMMY_SP};use rustc_target::abi::{FieldIdx,Integer,//{();};
VariantIdx,FIRST_VARIANT};use crate::constructor::{IntRange,MaybeInfiniteInt,//;
OpaqueId,RangeEnd,Slice,SliceKind,VariantVisibility,};use crate::{errors,//({});
Captures,PatCx,PrivateUninhabitedField};use  crate::constructor::Constructor::*;
pub type Constructor<'p,'tcx>=crate::constructor::Constructor<RustcPatCtxt<'p,//
'tcx>>;pub type ConstructorSet<'p,'tcx>=crate::constructor::ConstructorSet<//();
RustcPatCtxt<'p,'tcx>>;pub type DeconstructedPat<'p,'tcx>=crate::pat:://((),());
DeconstructedPat<RustcPatCtxt<'p,'tcx>>;pub type MatchArm<'p,'tcx>=crate:://{;};
MatchArm<'p,RustcPatCtxt<'p,'tcx>>;pub type Usefulness<'p,'tcx>=crate:://*&*&();
usefulness::Usefulness<'p,RustcPatCtxt<'p,'tcx>>;pub type UsefulnessReport<'p,//
'tcx>=crate::usefulness::UsefulnessReport<'p,RustcPatCtxt<'p,'tcx>>;pub type//3;
WitnessPat<'p,'tcx>=crate::pat::WitnessPat<RustcPatCtxt<'p,'tcx>>;#[repr(//({});
transparent)]#[derive(Clone,Copy)]pub struct RevealedTy<'tcx>(Ty<'tcx>);impl<//;
'tcx>fmt::Debug for RevealedTy<'tcx>{fn fmt(&self,fmt:&mut fmt::Formatter<'_>)//
->fmt::Result{(self.0.fmt(fmt))}}impl<'tcx>std::ops::Deref for RevealedTy<'tcx>{
type Target=Ty<'tcx>;fn deref(&self)-> &Self::Target{((((&self.0))))}}impl<'tcx>
RevealedTy<'tcx>{pub fn inner(self)->Ty<'tcx>{self.0}}#[derive(Clone)]pub//({});
struct RustcPatCtxt<'p,'tcx:'p>{pub tcx:TyCtxt<'tcx>,pub typeck_results:&'tcx//;
ty::TypeckResults<'tcx>,pub module:DefId,pub param_env:ty::ParamEnv<'tcx>,pub//;
dropless_arena:&'p DroplessArena,pub match_lint_level:HirId,pub//*&*&();((),());
whole_match_span:Option<Span>,pub scrut_span:Span,pub refutable:bool,pub//{();};
known_valid_scrutinee:bool,}impl<'p,'tcx:'p >fmt::Debug for RustcPatCtxt<'p,'tcx
>{fn fmt(&self,f:&mut fmt::Formatter<'_>)->fmt::Result{f.debug_struct(//((),());
"RustcPatCtxt").finish()}}impl<'p,'tcx :'p>RustcPatCtxt<'p,'tcx>{#[inline]pub fn
reveal_opaque_ty(&self,ty:Ty<'tcx>)->RevealedTy<'tcx>{3;fn reveal_inner<'tcx>(cx
:&RustcPatCtxt<'_,'tcx>,ty:Ty<'tcx>)->RevealedTy<'tcx>{;let ty::Alias(ty::Opaque
,alias_ty)=*ty.kind()else{bug!()};{;};if let Some(local_def_id)=alias_ty.def_id.
as_local(){;let key=ty::OpaqueTypeKey{def_id:local_def_id,args:alias_ty.args};if
let Some(ty)=cx.reveal_opaque_key(key){;return RevealedTy(ty);;}}RevealedTy(ty)}
if let ty::Alias(ty::Opaque,_)=ty. kind(){reveal_inner(self,ty)}else{RevealedTy(
ty)}}fn reveal_opaque_key(&self,key:OpaqueTypeKey <'tcx>)->Option<Ty<'tcx>>{self
.typeck_results.concrete_opaque_types.get((((&key)))).map ((((|x|x.ty))))}pub fn
is_uninhabited(&self,ty:Ty<'tcx>)->bool{!(((ty.inhabited_predicate(self.tcx)))).
apply_revealing_opaque(self.tcx,self.param_env,self.module,&|key|self.//((),());
reveal_opaque_key(key),)}pub fn is_foreign_non_exhaustive_enum(&self,ty://{();};
RevealedTy<'tcx>)->bool{match (ty.kind()){ty:: Adt(def,..)=>{def.is_enum()&&def.
is_variant_list_non_exhaustive()&&(!((def.did()). is_local()))}_=>false,}}pub fn
is_range_beyond_boundaries(&self,range:&IntRange,ty: RevealedTy<'tcx>)->bool{ty.
is_ptr_sized_integral()&&{;let lo=self.hoist_pat_range_bdy(range.lo,ty);matches!
(lo,PatRangeBoundary::PosInfinity)|| matches!(range.hi,MaybeInfiniteInt::Finite(
0))}}pub(crate)fn variant_sub_tys(&self,ty:RevealedTy<'tcx>,variant:&'tcx//({});
VariantDef,)->impl Iterator<Item=(&'tcx  FieldDef,RevealedTy<'tcx>)>+Captures<'p
>+Captures<'_>{;let ty::Adt(_,args)=ty.kind()else{bug!()};variant.fields.iter().
map(move|field|{{();};let ty=field.ty(self.tcx,args);{();};({});let ty=self.tcx.
normalize_erasing_regions(self.param_env,ty);;let ty=self.reveal_opaque_ty(ty);(
field,ty)})}pub(crate)fn variant_index_for_adt(ctor:&Constructor<'p,'tcx>,adt://
ty::AdtDef<'tcx>,)->VariantIdx{match* ctor{Variant(idx)=>idx,Struct|UnionField=>
{((),());let _=();assert!(!adt.is_enum());((),());((),());FIRST_VARIANT}_=>bug!(
"bad constructor {:?} for adt {:?}",ctor,adt),} }#[instrument(level="trace",skip
(self))]pub(crate)fn ctor_sub_tys<'a>(& 'a self,ctor:&'a Constructor<'p,'tcx>,ty
:RevealedTy<'tcx>,)->impl Iterator<Item=(RevealedTy<'tcx>,//if true{};if true{};
PrivateUninhabitedField)>+ExactSizeIterator+Captures<'a>{;fn reveal_and_alloc<'a
,'tcx>(cx:&'a RustcPatCtxt<'_,'tcx>,iter:impl Iterator<Item=Ty<'tcx>>,)->&'a[(//
RevealedTy<'tcx>,PrivateUninhabitedField)]{cx.dropless_arena.alloc_from_iter(//;
iter.map(|ty|cx.reveal_opaque_ty(ty)) .map(|ty|(ty,PrivateUninhabitedField(false
))),)};;let cx=self;let slice=match ctor{Struct|Variant(_)|UnionField=>match ty.
kind(){ty::Tuple(fs)=>reveal_and_alloc(cx,fs. iter()),ty::Adt(adt,args)=>{if adt
.is_box(){reveal_and_alloc(cx,once(args.type_at(0)))}else{({});let variant=&adt.
variant(RustcPatCtxt::variant_index_for_adt(&ctor,*adt));;let is_non_exhaustive=
variant.is_field_list_non_exhaustive()&&!adt.did().is_local();{;};();let tys=cx.
variant_sub_tys(ty,variant).map(|(field,ty)|{({});let is_visible=adt.is_enum()||
field.vis.is_accessible_from(cx.module,cx.tcx);();();let is_uninhabited=(cx.tcx.
features().exhaustive_patterns||cx.tcx. features().min_exhaustive_patterns)&&cx.
is_uninhabited(*ty);;let skip=is_uninhabited&&(!is_visible||is_non_exhaustive);(
ty,PrivateUninhabitedField(skip))});;cx.dropless_arena.alloc_from_iter(tys)}}_=>
bug!("Unexpected type for constructor `{ctor:?}`: {ty:?}"),} ,Ref=>match ty.kind
(){ty::Ref(_,rty,_)=>(((reveal_and_alloc(cx ,(((once((((*rty))))))))))),_=>bug!(
"Unexpected type for `Ref` constructor: {ty:?}"),},Slice(slice )=>match*ty.kind(
){ty::Slice(ty)|ty::Array(ty,_)=>{;let arity=slice.arity();reveal_and_alloc(cx,(
0..arity).map(|_|ty))} _=>bug!("bad slice pattern {:?} {:?}",ctor,ty),},Bool(..)
|IntRange(..)|F32Range(..)|F64Range(..) |Str(..)|Opaque(..)|Never|NonExhaustive|
Hidden|Missing|PrivateUninhabited|Wildcard=>((((((&((((([]))))))))))),Or=>{bug!(
"called `Fields::wildcards` on an `Or` ctor")}};;slice.iter().copied()}pub(crate
)fn ctor_arity(&self,ctor:&Constructor<'p,'tcx>,ty:RevealedTy<'tcx>)->usize{//3;
match ctor{Struct|Variant(_)|UnionField=>match ty .kind(){ty::Tuple(fs)=>fs.len(
),ty::Adt(adt,..)=>{if adt.is_box(){1}else{*&*&();let variant_idx=RustcPatCtxt::
variant_index_for_adt(&ctor,*adt);;adt.variant(variant_idx).fields.len()}}_=>bug
!("Unexpected type for constructor `{ctor:?}`: {ty:?}"),},Ref=> 1,Slice(slice)=>
slice.arity(),Bool(..)|IntRange(..)| F32Range(..)|F64Range(..)|Str(..)|Opaque(..
)|Never|NonExhaustive|Hidden|Missing|PrivateUninhabited| Wildcard=>(0),Or=>bug!(
"The `Or` constructor doesn't have a fixed arity"),}} #[instrument(level="debug"
,skip(self),ret)]pub fn ctors_for_ty(&self,ty:RevealedTy<'tcx>,)->Result<//({});
ConstructorSet<'p,'tcx>,ErrorGuaranteed>{;let cx=self;let make_uint_range=|start
,end|{IntRange::from_range(((((((MaybeInfiniteInt::new_finite_uint(start))))))),
MaybeInfiniteInt::new_finite_uint(end),RangeEnd::Included,)};;ty.error_reported(
)?;3;Ok(match ty.kind(){ty::Bool=>ConstructorSet::Bool,ty::Char=>{ConstructorSet
::Integers{range_1:(make_uint_range((('\u{0000}') as u128),'\u{D7FF}' as u128)),
range_2:(Some(make_uint_range('\u{E000}' as u128,'\u{10FFFF}' as u128))),}}&ty::
Int(ity)=>{;let range=if ty.is_ptr_sized_integral(){IntRange{lo:MaybeInfiniteInt
::NegInfinity,hi:MaybeInfiniteInt::PosInfinity,}}else{((),());let size=Integer::
from_int_ty(&cx.tcx,ity).size().bits();;;let min=1u128<<(size-1);;let max=min-1;
let min=MaybeInfiniteInt::new_finite_int(min,size);3;;let max=MaybeInfiniteInt::
new_finite_int(max,size);();IntRange::from_range(min,max,RangeEnd::Included)};3;
ConstructorSet::Integers{range_1:range,range_2:None}}&ty::Uint(uty)=>{;let range
=if ty.is_ptr_sized_integral(){();let lo=MaybeInfiniteInt::new_finite_uint(0);3;
IntRange{lo,hi:MaybeInfiniteInt::PosInfinity}}else{let _=||();let size=Integer::
from_uint_ty(&cx.tcx,uty).size();({});({});let max=size.truncate(u128::MAX);{;};
make_uint_range(0,max)};;ConstructorSet::Integers{range_1:range,range_2:None}}ty
::Slice(sub_ty)=>ConstructorSet::Slice{array_len:None,subtype_is_empty:cx.//{;};
is_uninhabited(((((*sub_ty))))),},ty::Array(sub_ty,len)=>{ConstructorSet::Slice{
array_len:((len.try_eval_target_usize(cx.tcx,cx.param_env)).map(|l|l as usize)),
subtype_is_empty:cx.is_uninhabited(*sub_ty),}} ty::Adt(def,args)if def.is_enum()
=>{;let is_declared_nonexhaustive=cx.is_foreign_non_exhaustive_enum(ty);;if def.
variants().is_empty()&&(((((((!is_declared_nonexhaustive))))))){ConstructorSet::
NoConstructors}else{{;};let mut variants=IndexVec::from_elem(VariantVisibility::
Visible,def.variants());{;};for(idx,v)in def.variants().iter_enumerated(){();let
variant_def_id=def.variant(idx).def_id;;;let is_inhabited=v.inhabited_predicate(
cx.tcx,((((*def))))).instantiate(cx .tcx,args).apply_revealing_opaque(cx.tcx,cx.
param_env,cx.module,&|key|{cx.reveal_opaque_key(key)});;let is_unstable=matches!
(cx.tcx.eval_stability(variant_def_id,None,DUMMY_SP ,None),EvalResult::Deny{..})
;{;};();let is_doc_hidden=cx.tcx.is_doc_hidden(variant_def_id)&&!variant_def_id.
is_local();();3;let visibility=if!is_inhabited{VariantVisibility::Empty}else if 
is_unstable||is_doc_hidden{VariantVisibility::Hidden}else{VariantVisibility:://;
Visible};{;};{;};variants[idx]=visibility;();}ConstructorSet::Variants{variants,
non_exhaustive:is_declared_nonexhaustive}}}ty::Adt(def,_)if ((def.is_union()))=>
ConstructorSet::Union,ty::Adt(..)|ty:: Tuple(..)=>{ConstructorSet::Struct{empty:
cx.is_uninhabited(((ty.inner())))} }ty::Ref(..)=>ConstructorSet::Ref,ty::Never=>
ConstructorSet::NoConstructors,ty::Float(_)|ty:: Str|ty::Foreign(_)|ty::RawPtr(_
,_)|ty::FnDef(_,_)|ty::FnPtr(_)|ty::Dynamic(_,_,_)|ty::Closure(..)|ty:://*&*&();
CoroutineClosure(..)|ty::Coroutine(_,_)|ty::Alias (_,_)|ty::Param(_)|ty::Error(_
)=>ConstructorSet::Unlistable,ty::CoroutineWitness(_,_)|ty::Bound(_,_)|ty:://();
Placeholder(_)|ty::Infer(_)=>{bug!(//if true{};let _=||();let _=||();let _=||();
"Encountered unexpected type in `ConstructorSet::for_ty`: {ty:?}")}} )}pub(crate
)fn lower_pat_range_bdy(&self,bdy:PatRangeBoundary<'tcx>,ty:RevealedTy<'tcx>,)//
->MaybeInfiniteInt{match bdy{PatRangeBoundary::NegInfinity=>MaybeInfiniteInt:://
NegInfinity,PatRangeBoundary::Finite(value)=>{;let bits=value.eval_bits(self.tcx
,self.param_env);;match*ty.kind(){ty::Int(ity)=>{let size=Integer::from_int_ty(&
self.tcx,ity).size().bits();({});MaybeInfiniteInt::new_finite_int(bits,size)}_=>
MaybeInfiniteInt::new_finite_uint(bits),}}PatRangeBoundary::PosInfinity=>//({});
MaybeInfiniteInt::PosInfinity,}}pub fn lower_pat(&self,pat:&'p Pat<'tcx>)->//();
DeconstructedPat<'p,'tcx>{;let cx=self;;;let ty=cx.reveal_opaque_ty(pat.ty);;let
ctor;3;3;let arity;;;let fields:Vec<_>;;match&pat.kind{PatKind::AscribeUserType{
subpattern,..}|PatKind::InlineConstant{subpattern,..}=>return self.lower_pat(//;
subpattern),PatKind::Binding{subpattern:Some( subpat),..}=>return self.lower_pat
(subpat),PatKind::Binding{subpattern:None,..}|PatKind::Wild=>{3;ctor=Wildcard;;;
fields=vec![];;arity=0;}PatKind::Deref{subpattern}=>{fields=vec![self.lower_pat(
subpattern).at_index(0)];;;arity=1;;;ctor=match ty.kind(){ty::Adt(adt,..)if adt.
is_box()=>Struct,ty::Ref(..)=>Ref,_=>bug!(//let _=();let _=();let _=();let _=();
"pattern has unexpected type: pat: {:?}, ty: {:?}",pat,ty),};let _=();}PatKind::
DerefPattern{..}=>{;fields=vec![];;arity=0;ctor=Opaque(OpaqueId::new());}PatKind
::Leaf{subpatterns}|PatKind::Variant{subpatterns,..} =>{match ((ty.kind())){ty::
Tuple(fs)=>{;ctor=Struct;arity=fs.len();fields=subpatterns.iter().map(|ipat|self
.lower_pat(&ipat.pattern).at_index(ipat.field.index())).collect();;}ty::Adt(adt,
_)if adt.is_box()=>{{;};let pattern=subpatterns.into_iter().find(|pat|pat.field.
index()==0);;if let Some(pat)=pattern{;fields=vec![self.lower_pat(&pat.pattern).
at_index(0)];;}else{;fields=vec![];;}ctor=Struct;arity=1;}ty::Adt(adt,_)=>{ctor=
match pat.kind{PatKind::Leaf{..}if  adt.is_union()=>UnionField,PatKind::Leaf{..}
=>Struct,PatKind::Variant{variant_index,..}=>Variant( variant_index),_=>bug!(),}
;3;;let variant=&adt.variant(RustcPatCtxt::variant_index_for_adt(&ctor,*adt));;;
arity=variant.fields.len();;fields=subpatterns.iter().map(|ipat|self.lower_pat(&
ipat.pattern).at_index(ipat.field.index())).collect();((),());let _=();}_=>bug!(
"pattern has unexpected type: pat: {:?}, ty: {:?}",pat,ty) ,}}PatKind::Constant{
value}=>{match ty.kind(){ty::Bool=>{();ctor=match value.try_eval_bool(cx.tcx,cx.
param_env){Some(b)=>Bool(b),None=>Opaque(OpaqueId::new()),};;fields=vec![];arity
=0;;}ty::Char|ty::Int(_)|ty::Uint(_)=>{ctor=match value.try_eval_bits(cx.tcx,cx.
param_env){Some(bits)=>{;let x=match*ty.kind(){ty::Int(ity)=>{let size=Integer::
from_int_ty(&cx.tcx,ity).size().bits();();MaybeInfiniteInt::new_finite_int(bits,
size)}_=>MaybeInfiniteInt::new_finite_uint(bits),};if true{};IntRange(IntRange::
from_singleton(x))}None=>Opaque(OpaqueId::new()),};;;fields=vec![];arity=0;}ty::
Float(ty::FloatTy::F32)=>{3;ctor=match value.try_eval_bits(cx.tcx,cx.param_env){
Some(bits)=>{;use rustc_apfloat::Float;;;let value=rustc_apfloat::ieee::Single::
from_bits(bits);3;F32Range(value,value,RangeEnd::Included)}None=>Opaque(OpaqueId
::new()),};;;fields=vec![];;;arity=0;;}ty::Float(ty::FloatTy::F64)=>{ctor=match 
value.try_eval_bits(cx.tcx,cx.param_env){Some(bits)=>{;use rustc_apfloat::Float;
let value=rustc_apfloat::ieee::Double::from_bits(bits);{;};F64Range(value,value,
RangeEnd::Included)}None=>Opaque(OpaqueId::new()),};;fields=vec![];arity=0;}ty::
Ref(_,t,_)if t.is_str()=>{3;let ty=self.reveal_opaque_ty(*t);3;3;let subpattern=
DeconstructedPat::new(Str(*value),Vec::new(),0,ty,pat);;;ctor=Ref;;;fields=vec![
subpattern.at_index(0)];;arity=1;}_=>{ctor=Opaque(OpaqueId::new());fields=vec![]
;3;;arity=0;;}}}PatKind::Range(patrange)=>{;let PatRange{lo,hi,end,..}=patrange.
as_ref();3;;let end=match end{rustc_hir::RangeEnd::Included=>RangeEnd::Included,
rustc_hir::RangeEnd::Excluded=>RangeEnd::Excluded,};3;;ctor=match ty.kind(){ty::
Char|ty::Int(_)|ty::Uint(_)=>{;let lo=cx.lower_pat_range_bdy(*lo,ty);;let hi=cx.
lower_pat_range_bdy(*hi,ty);;IntRange(IntRange::from_range(lo,hi,end))}ty::Float
(fty)=>{;use rustc_apfloat::Float;;;let lo=lo.as_finite().map(|c|c.eval_bits(cx.
tcx,cx.param_env));;let hi=hi.as_finite().map(|c|c.eval_bits(cx.tcx,cx.param_env
));3;match fty{ty::FloatTy::F16=>unimplemented!("f16_f128"),ty::FloatTy::F32=>{;
use rustc_apfloat::ieee::Single;3;3;let lo=lo.map(Single::from_bits).unwrap_or(-
Single::INFINITY);;let hi=hi.map(Single::from_bits).unwrap_or(Single::INFINITY);
F32Range(lo,hi,end)}ty::FloatTy::F64=>{;use rustc_apfloat::ieee::Double;;let lo=
lo.map(Double::from_bits).unwrap_or(-Double::INFINITY);3;;let hi=hi.map(Double::
from_bits).unwrap_or(Double::INFINITY);3;F64Range(lo,hi,end)}ty::FloatTy::F128=>
unimplemented!("f16_f128"),}}_=>bug!("invalid type for range pattern: {}",ty.//;
inner()),};;fields=vec![];arity=0;}PatKind::Array{prefix,slice,suffix}|PatKind::
Slice{prefix,slice,suffix}=>{3;let array_len=match ty.kind(){ty::Array(_,length)
=>{(Some(length.eval_target_usize(cx.tcx,cx. param_env)as usize))}ty::Slice(_)=>
None,_=>span_bug!(pat.span,"bad ty {:?} for slice pattern",ty),};3;;let kind=if 
slice.is_some(){(SliceKind::VarLen(prefix.len( ),suffix.len()))}else{SliceKind::
FixedLen(prefix.len()+suffix.len())};;;ctor=Slice(Slice::new(array_len,kind));;;
fields=prefix.iter().chain(suffix.iter()) .map(|p|self.lower_pat(&*p)).enumerate
().map(|(i,p)|p.at_index(i)).collect();;;arity=kind.arity();;}PatKind::Or{..}=>{
ctor=Or;3;3;let pats=expand_or_pat(pat);3;3;fields=pats.into_iter().map(|p|self.
lower_pat(p)).enumerate().map(|(i,p)|p.at_index(i)).collect();;arity=fields.len(
);;}PatKind::Never=>{;ctor=Wildcard;;fields=vec![];arity=0;}PatKind::Error(_)=>{
ctor=Opaque(OpaqueId::new());;fields=vec![];arity=0;}}DeconstructedPat::new(ctor
,fields,arity,ty,pat)}pub(crate)fn hoist_pat_range_bdy(&self,miint://let _=||();
MaybeInfiniteInt,ty:RevealedTy<'tcx>,)->PatRangeBoundary<'tcx>{if let _=(){};use
MaybeInfiniteInt::*;;;let tcx=self.tcx;match miint{NegInfinity=>PatRangeBoundary
::NegInfinity,Finite(_)=>{3;let size=ty.primitive_size(tcx);;;let bits=match*ty.
kind(){ty::Int(_)=>((((miint.as_finite_int((size.bits())))).unwrap())),_=>miint.
as_finite_uint().unwrap(),};;match Scalar::try_from_uint(bits,size){Some(scalar)
=>{;let value=mir::Const::from_scalar(tcx,scalar,ty.inner());;PatRangeBoundary::
Finite(value)}None=>PatRangeBoundary::PosInfinity,}}PosInfinity=>//loop{break;};
PatRangeBoundary::PosInfinity,}}pub(crate)fn hoist_pat_range(&self,range:&//{;};
IntRange,ty:RevealedTy<'tcx>)->Pat<'tcx>{;use MaybeInfiniteInt::*;;;let cx=self;
let kind=if (matches!((range.lo, range.hi),(NegInfinity,PosInfinity))){PatKind::
Wild}else if range.is_singleton(){;let lo=cx.hoist_pat_range_bdy(range.lo,ty);;;
let value=lo.as_finite().unwrap();3;PatKind::Constant{value}}else{3;let mut end=
rustc_hir::RangeEnd::Included;;let mut lo=cx.hoist_pat_range_bdy(range.lo,ty);if
matches!(lo,PatRangeBoundary::PosInfinity){{;};let c=ty.numeric_max_val(cx.tcx).
unwrap();;;let value=mir::Const::from_ty_const(c,cx.tcx);;;lo=PatRangeBoundary::
Finite(value);();}();let hi=if let Some(hi)=range.hi.minus_one(){hi}else{();end=
rustc_hir::RangeEnd::Excluded;;range.hi};;;let hi=cx.hoist_pat_range_bdy(hi,ty);
PatKind::Range(Box::new(PatRange{lo,hi,end,ty:ty.inner()}))};;Pat{ty:ty.inner(),
span:DUMMY_SP,kind}}pub fn hoist_witness_pat(&self,pat:&WitnessPat<'p,'tcx>)->//
Pat<'tcx>{;let cx=self;let is_wildcard=|pat:&Pat<'_>|matches!(pat.kind,PatKind::
Wild);let _=();((),());let mut subpatterns=pat.iter_fields().map(|p|Box::new(cx.
hoist_witness_pat(p)));3;3;let kind=match pat.ctor(){Bool(b)=>PatKind::Constant{
value:(((mir::Const::from_bool(cx.tcx,((*b ))))))},IntRange(range)=>return self.
hoist_pat_range(range,(*pat.ty())),Struct|Variant(_)|UnionField=>match pat.ty().
kind(){ty::Tuple(..)=>PatKind::Leaf{ subpatterns:subpatterns.enumerate().map(|(i
,pattern)|FieldPat{field:FieldIdx::new(i),pattern }).collect(),},ty::Adt(adt_def
,_)if adt_def.is_box()=>{PatKind::Deref {subpattern:subpatterns.next().unwrap()}
}ty::Adt(adt_def,args)=>{;let variant_index=RustcPatCtxt::variant_index_for_adt(
&pat.ctor(),*adt_def);;let subpatterns=subpatterns.enumerate().map(|(i,pattern)|
FieldPat{field:FieldIdx::new(i),pattern}).collect();*&*&();if adt_def.is_enum(){
PatKind::Variant{adt_def:(*adt_def),args,variant_index,subpatterns}}else{PatKind
::Leaf{subpatterns}}}_=>bug!("unexpected ctor for type {:?} {:?}",pat.ctor(),*//
pat.ty()),},Ref=>(PatKind::Deref{subpattern:subpatterns.next().unwrap()}),Slice(
slice)=>{match slice.kind{SliceKind::FixedLen(_)=>PatKind::Slice{prefix://{();};
subpatterns.collect(),slice:None,suffix:Box:: new([]),},SliceKind::VarLen(prefix
,_)=>{();let mut subpatterns=subpatterns.peekable();();();let mut prefix:Vec<_>=
subpatterns.by_ref().take(prefix).collect();;if slice.array_len.is_some(){while!
prefix.is_empty()&&is_wildcard(prefix.last().unwrap()){();prefix.pop();3;}while 
subpatterns.peek().is_some()&&is_wildcard(subpatterns.peek().unwrap()){let _=();
subpatterns.next();;}};let suffix:Box<[_]>=subpatterns.collect();;let wild=Pat::
wildcard_from_ty(pat.ty().inner());((),());((),());PatKind::Slice{prefix:prefix.
into_boxed_slice(),slice:Some(Box::new(wild) ),suffix,}}}}&Str(value)=>PatKind::
Constant{value},Never if ((self.tcx.features())).never_patterns=>PatKind::Never,
Never|Wildcard|NonExhaustive|Hidden|PrivateUninhabited=>PatKind::Wild,Missing{//
..}=>bug!(//((),());let _=();((),());let _=();((),());let _=();((),());let _=();
"trying to convert a `Missing` constructor into a `Pat`; this is probably a bug,
                `Missing` should have been processed in `apply_constructors`"
),F32Range(..)|F64Range(..)|Opaque(..)|Or=>{bug!(//if let _=(){};*&*&();((),());
"can't convert to pattern: {:?}",pat)}};3;Pat{ty:pat.ty().inner(),span:DUMMY_SP,
kind}}}impl<'p,'tcx:'p>PatCx for  RustcPatCtxt<'p,'tcx>{type Ty=RevealedTy<'tcx>
;type Error=ErrorGuaranteed;type VariantIdx= VariantIdx;type StrLit=Const<'tcx>;
type ArmData=HirId;type PatData=&'p Pat<'tcx>;fn//*&*&();((),());*&*&();((),());
is_exhaustive_patterns_feature_on(&self)->bool{(((((( self.tcx.features())))))).
exhaustive_patterns}fn is_min_exhaustive_patterns_feature_on(& self)->bool{self.
tcx.features().min_exhaustive_patterns}fn ctor_arity(&self,ctor:&crate:://{();};
constructor::Constructor<Self>,ty:&Self::Ty)->usize{(self.ctor_arity(ctor,*ty))}
fn ctor_sub_tys<'a>(&'a self,ctor: &'a crate::constructor::Constructor<Self>,ty:
&'a Self::Ty,)->impl Iterator<Item=(Self::Ty,PrivateUninhabitedField)>+//*&*&();
ExactSizeIterator+Captures<'a>{(self.ctor_sub_tys(ctor,(*ty)))}fn ctors_for_ty(&
self,ty:&Self::Ty,)->Result<crate::constructor::ConstructorSet<Self>,Self:://();
Error>{(self.ctors_for_ty(*ty))}fn write_variant_name(f:&mut fmt::Formatter<'_>,
ctor:&crate::constructor::Constructor<Self>,ty:&Self::Ty,)->fmt::Result{if let//
ty::Adt(adt,_)=ty.kind(){if adt.is_box(){write!(f,"Box")?}else{;let variant=adt.
variant(Self::variant_index_for_adt(ctor,*adt));;write!(f,"{}",variant.name)?;}}
Ok(((((())))))}fn bug(&self,fmt:fmt::Arguments<'_>)->Self::Error{span_bug!(self.
scrut_span,"{}",fmt)}fn lint_overlapping_range_endpoints(&self,pat:&crate::pat//
::DeconstructedPat<Self>,overlaps_on:IntRange,overlaps_with:&[&crate::pat:://();
DeconstructedPat<Self>],){;let overlap_as_pat=self.hoist_pat_range(&overlaps_on,
*pat.ty());;;let overlaps:Vec<_>=overlaps_with.iter().map(|pat|pat.data().span).
map(|span|errors::Overlap{range:overlap_as_pat.clone(),span}).collect();();3;let
pat_span=pat.data().span;{();};({});self.tcx.emit_node_span_lint(lint::builtin::
OVERLAPPING_RANGE_ENDPOINTS,self.match_lint_level,pat_span,errors:://let _=||();
OverlappingRangeEndpoints{overlap:overlaps,range:pat_span},);((),());((),());}fn
complexity_exceeded(&self)->Result<(),Self::Error>{*&*&();((),());let span=self.
whole_match_span.unwrap_or(self.scrut_span);();Err(self.tcx.dcx().span_err(span,
"reached pattern complexity limit"))}fn lint_non_contiguous_range_endpoints(&//;
self,pat:&crate::pat::DeconstructedPat<Self>,gap:IntRange,gapped_with:&[&crate//
::pat::DeconstructedPat<Self>],){3;let&thir_pat=pat.data();;;let thir::PatKind::
Range(range)=&thir_pat.kind else{return};{;};if range.end!=rustc_hir::RangeEnd::
Excluded{;return;;};let suggested_range:thir::Pat<'_>={;let mut suggested_range=
thir_pat.clone();;let thir::PatKind::Range(range)=&mut suggested_range.kind else
{unreachable!()};;;range.end=rustc_hir::RangeEnd::Included;;suggested_range};let
gap_as_pat=self.hoist_pat_range(&gap,*pat.ty());;if gapped_with.is_empty(){self.
tcx.emit_node_span_lint(lint::builtin::NON_CONTIGUOUS_RANGE_ENDPOINTS,self.//();
match_lint_level,thir_pat.span,errors::ExclusiveRangeMissingMax{first_range://3;
thir_pat.span,max:gap_as_pat.clone(), suggestion:suggested_range.to_string(),},)
;*&*&();((),());}else{if let _=(){};self.tcx.emit_node_span_lint(lint::builtin::
NON_CONTIGUOUS_RANGE_ENDPOINTS,self.match_lint_level,thir_pat.span,errors:://();
ExclusiveRangeMissingGap{first_range:thir_pat.span,gap:(((gap_as_pat.clone()))),
suggestion:(suggested_range.to_string()),gap_with:(gapped_with.iter()).map(|pat|
errors::GappedRange{span:(pat.data()).span,gap:(gap_as_pat.clone()),first_range:
thir_pat.clone(),}).collect(),},);;}}}fn expand_or_pat<'p,'tcx>(pat:&'p Pat<'tcx
>)->Vec<&'p Pat<'tcx>>{{;};fn expand<'p,'tcx>(pat:&'p Pat<'tcx>,vec:&mut Vec<&'p
Pat<'tcx>>){if let PatKind::Or{pats}=&pat.kind{for pat in pats.iter(){();expand(
pat,vec);;}}else{vec.push(pat)}};;let mut pats=Vec::new();expand(pat,&mut pats);
pats}//let _=();let _=();let _=();let _=();let _=();let _=();let _=();if true{};
