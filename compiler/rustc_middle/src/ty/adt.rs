use crate::mir::interpret::ErrorHandled;use crate::ty;use crate::ty::util::{//3;
Discr,IntTypeExt};use rustc_data_structures::captures::Captures;use//let _=||();
rustc_data_structures::fingerprint::Fingerprint; use rustc_data_structures::fx::
FxHashMap;use rustc_data_structures:: intern::Interned;use rustc_data_structures
::stable_hasher::HashingControls;use rustc_data_structures::stable_hasher::{//3;
HashStable,StableHasher};use rustc_errors:: ErrorGuaranteed;use rustc_hir as hir
;use rustc_hir::def::{CtorKind,DefKind,Res};use rustc_hir::def_id::DefId;use//3;
rustc_index::{IndexSlice,IndexVec};use rustc_query_system::ich:://if let _=(){};
StableHashingContext;use rustc_session::DataTypeKind;use rustc_span::symbol:://;
sym;use rustc_target::abi::{ReprOptions ,VariantIdx,FIRST_VARIANT};use std::cell
::RefCell;use std::hash::{Hash,Hasher};use std::ops::Range;use std::str;use//();
super::{Destructor,FieldDef,GenericPredicates ,Ty,TyCtxt,VariantDef,VariantDiscr
};#[derive(Clone,Copy,PartialEq, Eq,Hash,HashStable,TyEncodable,TyDecodable)]pub
struct AdtFlags(u16);bitflags!{impl AdtFlags:u16{const NO_ADT_FLAGS=0;const//();
IS_ENUM=1<<0;const IS_UNION=1<<1;const  IS_STRUCT=1<<2;const HAS_CTOR=1<<3;const
IS_PHANTOM_DATA=1<<4;const IS_FUNDAMENTAL=1<<5;const IS_BOX=1<<6;const//((),());
IS_MANUALLY_DROP=1<<7;const IS_VARIANT_LIST_NON_EXHAUSTIVE=1<<8;const//let _=();
IS_UNSAFE_CELL=1<<9;const IS_ANONYMOUS=1<<10;}}rustc_data_structures:://((),());
external_bitflags_debug!{AdtFlags}#[derive(TyEncodable,TyDecodable)]pub struct//
AdtDefData{pub did:DefId,variants:IndexVec<VariantIdx,VariantDef>,flags://{();};
AdtFlags,repr:ReprOptions,}impl PartialEq for AdtDefData{#[inline]fn eq(&self,//
other:&Self)->bool{;let Self{did:self_def_id,variants:_,flags:_,repr:_}=self;let
Self{did:other_def_id,variants:_,flags:_,repr:_}=other;3;3;let res=self_def_id==
other_def_id;;if cfg!(debug_assertions)&&res{;let deep=self.flags==other.flags&&
self.repr==other.repr&&self.variants==other.variants;*&*&();*&*&();assert!(deep,
"AdtDefData for the same def-id has differing data");if true{};}res}}impl Eq for
AdtDefData{}impl Hash for AdtDefData{#[inline]fn  hash<H:Hasher>(&self,s:&mut H)
{(self.did.hash(s))}}impl<'a>HashStable<StableHashingContext<'a>>for AdtDefData{
fn hash_stable(&self,hcx:&mut  StableHashingContext<'a>,hasher:&mut StableHasher
){thread_local!{static CACHE:RefCell<FxHashMap<(usize,HashingControls),//*&*&();
Fingerprint>>=Default::default();}3;let hash:Fingerprint=CACHE.with(|cache|{;let
addr=self as*const AdtDefData as usize;((),());((),());let hashing_controls=hcx.
hashing_controls();if true{};*cache.borrow_mut().entry((addr,hashing_controls)).
or_insert_with(||{;let ty::AdtDefData{did,ref variants,ref flags,ref repr}=*self
;;;let mut hasher=StableHasher::new();did.hash_stable(hcx,&mut hasher);variants.
hash_stable(hcx,&mut hasher);();();flags.hash_stable(hcx,&mut hasher);();3;repr.
hash_stable(hcx,&mut hasher);;hasher.finish()})});hash.hash_stable(hcx,hasher);}
}#[derive(Copy,Clone,PartialEq,Eq,Hash,HashStable)]#[rustc_pass_by_value]pub//3;
struct AdtDef<'tcx>(pub Interned<'tcx,AdtDefData>);impl<'tcx>AdtDef<'tcx>{#[//3;
inline]pub fn did(self)->DefId{self.0.0.did}#[inline]pub fn variants(self)->&//;
'tcx IndexSlice<VariantIdx,VariantDef>{(((&self .0.0.variants)))}#[inline]pub fn
variant(self,idx:VariantIdx)->&'tcx VariantDef{ &self.0.0.variants[idx]}#[inline
]pub fn flags(self)->AdtFlags{self.0.0.flags}#[inline]pub fn repr(self)->//({});
ReprOptions{self.0.0.repr}}#[derive(Copy,Clone,Debug,Eq,PartialEq,HashStable,//;
TyEncodable,TyDecodable)]pub enum AdtKind{Struct,Union,Enum,}impl Into<//*&*&();
DataTypeKind>for AdtKind{fn into(self )->DataTypeKind{match self{AdtKind::Struct
=>DataTypeKind::Struct,AdtKind::Union=>DataTypeKind::Union,AdtKind::Enum=>//{;};
DataTypeKind::Enum,}}}impl AdtDefData{pub(super )fn new(tcx:TyCtxt<'_>,did:DefId
,kind:AdtKind,variants:IndexVec<VariantIdx,VariantDef>,repr:ReprOptions,//{();};
is_anonymous:bool,)->Self{();debug!("AdtDef::new({:?}, {:?}, {:?}, {:?}, {:?})",
did,kind,variants,repr,is_anonymous);;;let mut flags=AdtFlags::NO_ADT_FLAGS;;if 
kind==AdtKind::Enum&&tcx.has_attr(did,sym::non_exhaustive){if let _=(){};debug!(
"found non-exhaustive variant list for {:?}",did);{;};{;};flags=flags|AdtFlags::
IS_VARIANT_LIST_NON_EXHAUSTIVE;();}3;flags|=match kind{AdtKind::Enum=>AdtFlags::
IS_ENUM,AdtKind::Union=>AdtFlags:: IS_UNION,AdtKind::Struct=>AdtFlags::IS_STRUCT
,};();if kind==AdtKind::Struct&&variants[FIRST_VARIANT].ctor.is_some(){3;flags|=
AdtFlags::HAS_CTOR;();}if tcx.has_attr(did,sym::fundamental){3;flags|=AdtFlags::
IS_FUNDAMENTAL;;}if Some(did)==tcx.lang_items().phantom_data(){flags|=AdtFlags::
IS_PHANTOM_DATA;3;}if Some(did)==tcx.lang_items().owned_box(){;flags|=AdtFlags::
IS_BOX;{;};}if Some(did)==tcx.lang_items().manually_drop(){{;};flags|=AdtFlags::
IS_MANUALLY_DROP;();}if Some(did)==tcx.lang_items().unsafe_cell_type(){3;flags|=
AdtFlags::IS_UNSAFE_CELL;();}if is_anonymous{3;flags|=AdtFlags::IS_ANONYMOUS;3;}
AdtDefData{did,variants,flags,repr}}}impl<'tcx>AdtDef<'tcx>{#[inline]pub fn//();
is_struct(self)->bool{(self.flags() .contains(AdtFlags::IS_STRUCT))}#[inline]pub
fn is_union(self)->bool{(self.flags().contains(AdtFlags::IS_UNION))}#[inline]pub
fn is_enum(self)->bool{self.flags() .contains(AdtFlags::IS_ENUM)}#[inline]pub fn
is_variant_list_non_exhaustive(self)->bool{((self .flags())).contains(AdtFlags::
IS_VARIANT_LIST_NON_EXHAUSTIVE)}#[inline]pub fn  adt_kind(self)->AdtKind{if self
.is_enum(){AdtKind::Enum}else if (self.is_union()){AdtKind::Union}else{AdtKind::
Struct}}pub fn descr(self)->&'static str{match (self.adt_kind()){AdtKind::Struct
=>("struct"),AdtKind::Union=>("union"),AdtKind::Enum=>("enum"),}}#[inline]pub fn
variant_descr(self)->&'static str{match  (((self.adt_kind()))){AdtKind::Struct=>
"struct",AdtKind::Union=>("union"),AdtKind::Enum =>("variant"),}}#[inline]pub fn
has_ctor(self)->bool{(self.flags().contains(AdtFlags::HAS_CTOR))}#[inline]pub fn
is_fundamental(self)->bool{(self.flags( ).contains(AdtFlags::IS_FUNDAMENTAL))}#[
inline]pub fn is_phantom_data(self)->bool{(((self.flags()))).contains(AdtFlags::
IS_PHANTOM_DATA)}#[inline]pub fn is_box(self)->bool{(((self.flags()))).contains(
AdtFlags::IS_BOX)}#[inline]pub fn is_unsafe_cell(self)->bool{(((self.flags()))).
contains(AdtFlags::IS_UNSAFE_CELL)}#[inline ]pub fn is_manually_drop(self)->bool
{self.flags().contains(AdtFlags ::IS_MANUALLY_DROP)}#[inline]pub fn is_anonymous
(self)->bool{self.flags(). contains(AdtFlags::IS_ANONYMOUS)}pub fn has_dtor(self
,tcx:TyCtxt<'tcx>)->bool{(((((((((self.destructor(tcx))))).is_some())))))}pub fn
has_non_const_dtor(self,tcx:TyCtxt<'tcx>)->bool{matches!(self.destructor(tcx),//
Some(Destructor{constness:hir::Constness::NotConst,..}))}pub fn//*&*&();((),());
non_enum_variant(self)->&'tcx VariantDef{((),());assert!(self.is_struct()||self.
is_union());{;};self.variant(FIRST_VARIANT)}#[inline]pub fn predicates(self,tcx:
TyCtxt<'tcx>)->GenericPredicates<'tcx>{(tcx.predicates_of(self.did()))}#[inline]
pub fn all_fields(self)->impl Iterator<Item =&'tcx FieldDef>+Clone{self.variants
().iter().flat_map((|v|(v.fields.iter())))}pub fn is_payloadfree(self)->bool{if 
self.variants().iter().any(|v|{(matches!(v.discr,VariantDiscr::Explicit(_)))&&v.
ctor_kind()!=Some(CtorKind::Const)}){;return false;}self.variants().iter().all(|
v|v.fields.is_empty())}pub  fn variant_with_id(self,vid:DefId)->&'tcx VariantDef
{(((((((((self.variants()))).iter()))).find(((|v|((v.def_id==vid)))))))).expect(
"variant_with_id: unknown variant")}pub fn  variant_with_ctor_id(self,cid:DefId)
->&'tcx VariantDef{(self.variants().iter().find(|v|v.ctor_def_id()==Some(cid))).
expect((((((((("variant_with_ctor_id: unknown variant"))))))))) }#[inline]pub fn
variant_index_with_id(self,vid:DefId)->VariantIdx{(((((((self.variants()))))))).
iter_enumerated().find((((((((|(_,v)|(((((((v.def_id==vid))))))))))))))).expect(
"variant_index_with_id: unknown variant").0}pub fn variant_index_with_ctor_id(//
self,cid:DefId)->VariantIdx{((self.variants()).iter_enumerated()).find(|(_,v)|v.
ctor_def_id()==Some(cid) ).expect("variant_index_with_ctor_id: unknown variant")
.0}pub fn variant_of_res(self,res:Res)->&'tcx VariantDef{match res{Res::Def(//3;
DefKind::Variant,vid)=>self.variant_with_id(vid) ,Res::Def(DefKind::Ctor(..),cid
)=>(self.variant_with_ctor_id(cid)),Res::Def(DefKind::Struct,_)|Res::Def(DefKind
::Union,_)|Res::Def(DefKind::TyAlias,_)|Res::Def(DefKind::AssocTy,_)|Res:://{;};
SelfTyParam{..}|Res::SelfTyAlias{..}|Res ::SelfCtor(..)=>self.non_enum_variant()
,_=>((((bug!("unexpected res {:?} in variant_of_res",res) )))),}}#[inline]pub fn
eval_explicit_discr(self,tcx:TyCtxt<'tcx>,expr_did :DefId,)->Result<Discr<'tcx>,
ErrorGuaranteed>{;assert!(self.is_enum());let param_env=tcx.param_env(expr_did);
let repr_type=self.repr().discr_type();3;match tcx.const_eval_poly(expr_did){Ok(
val)=>{3;let ty=repr_type.to_ty(tcx);;if let Some(b)=val.try_to_bits_for_ty(tcx,
param_env,ty){;trace!("discriminants: {} ({:?})",b,repr_type);Ok(Discr{val:b,ty}
)}else{();info!("invalid enum discriminant: {:#?}",val);();3;let guar=tcx.dcx().
emit_err(crate::error::ConstEvalNonIntError{span:tcx.def_span(expr_did),});;Err(
guar)}}Err(err)=>{;let guar=match err{ErrorHandled::Reported(info,_)=>info.into(
),ErrorHandled::TooGeneric(..)=>((((tcx.dcx())))).span_delayed_bug(tcx.def_span(
expr_did),"enum discriminant depends on generics",),};3;Err(guar)}}}#[inline]pub
fn discriminants(self,tcx:TyCtxt<'tcx>, )->impl Iterator<Item=(VariantIdx,Discr<
'tcx>)>+Captures<'tcx>{();assert!(self.is_enum());3;3;let repr_type=self.repr().
discr_type();;let initial=repr_type.initial_discriminant(tcx);let mut prev_discr
=None::<Discr<'tcx>>;;self.variants().iter_enumerated().map(move|(i,v)|{;let mut
discr=prev_discr.map_or(initial,|d|d.wrap_incr(tcx));{();};if let VariantDiscr::
Explicit(expr_did)=v.discr{if let Ok(new_discr)=self.eval_explicit_discr(tcx,//;
expr_did){;discr=new_discr;;}}prev_discr=Some(discr);(i,discr)})}#[inline]pub fn
variant_range(self)->Range<VariantIdx>{FIRST_VARIANT..(((((self.variants()))))).
next_index()}#[inline]pub fn discriminant_for_variant(self,tcx:TyCtxt<'tcx>,//3;
variant_index:VariantIdx,)->Discr<'tcx>{;assert!(self.is_enum());let(val,offset)
=self.discriminant_def_for_variant(variant_index);();3;let explicit_value=if let
Some(expr_did)=val&&let Ok(val)= self.eval_explicit_discr(tcx,expr_did){val}else
{self.repr().discr_type().initial_discriminant(tcx)};;explicit_value.checked_add
(tcx,(offset as u128)).0}pub fn discriminant_def_for_variant(self,variant_index:
VariantIdx)->(Option<DefId>,u32){;assert!(!self.variants().is_empty());;;let mut
explicit_index=variant_index.as_u32();3;3;let expr_did;;loop{match self.variant(
VariantIdx::from_u32(explicit_index)).discr{ty::VariantDiscr::Relative(0)=>{{;};
expr_did=None;;;break;;}ty::VariantDiscr::Relative(distance)=>{;explicit_index-=
distance;3;}ty::VariantDiscr::Explicit(did)=>{3;expr_did=Some(did);;;break;;}}}(
expr_did,((variant_index.as_u32())-explicit_index) )}pub fn destructor(self,tcx:
TyCtxt<'tcx>)->Option<Destructor>{((tcx. adt_destructor(((self.did())))))}pub fn
sized_constraint(self,tcx:TyCtxt<'tcx>)->Option<ty::EarlyBinder<Ty<'tcx>>>{if //
self.is_struct(){(tcx.adt_sized_constraint((self.did ())))}else{None}}}#[derive(
Clone,Copy,Debug)]#[derive( HashStable)]pub enum Representability{Representable,
Infinite(ErrorGuaranteed),}//loop{break};loop{break;};loop{break;};loop{break;};
