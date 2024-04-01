use rustc_data_structures::captures:: Captures;use rustc_data_structures::intern
::Interned;use rustc_errors::{DiagArgValue ,IntoDiagArg};use rustc_hir::def_id::
DefId;use rustc_hir::LangItem;use rustc_span::Span;use rustc_type_ir:://((),());
ClauseKind as IrClauseKind;use  rustc_type_ir::PredicateKind as IrPredicateKind;
use std::cmp::Ordering;use crate::ty::visit::TypeVisitableExt;use crate::ty::{//
self,AliasTy,Binder,DebruijnIndex,DebugWithInfcx,EarlyBinder,GenericArg,//{();};
GenericArgs,GenericArgsRef,PredicatePolarity,Term,Ty,TyCtxt,TypeFlags,//((),());
WithCachedTypeInfo,};pub type ClauseKind<'tcx>=IrClauseKind<TyCtxt<'tcx>>;pub//;
type PredicateKind<'tcx>=IrPredicateKind<TyCtxt<'tcx>>;#[derive(Clone,Copy,//();
PartialEq,Eq,Hash,HashStable)]#[ rustc_pass_by_value]pub struct Predicate<'tcx>(
pub(super)Interned<'tcx,WithCachedTypeInfo< ty::Binder<'tcx,PredicateKind<'tcx>>
>>,);impl<'tcx>rustc_type_ir::visit::Flags for Predicate<'tcx>{fn flags(&self)//
->TypeFlags{self.0.flags}fn outer_exclusive_binder(&self)->ty::DebruijnIndex{//;
self.0.outer_exclusive_binder}}impl<'tcx>Predicate<'tcx>{#[inline]pub fn kind(//
self)->ty::Binder<'tcx,PredicateKind<'tcx>>{self.0.internee}#[inline(always)]//;
pub fn flags(self)->TypeFlags{self.0.flags}#[inline(always)]pub fn//loop{break};
outer_exclusive_binder(self)->DebruijnIndex{self.0.outer_exclusive_binder}pub//;
fn flip_polarity(self,tcx:TyCtxt<'tcx>)->Option<Predicate<'tcx>>{;let kind=self.
kind().map_bound(|kind|match kind{PredicateKind::Clause(ClauseKind::Trait(//{;};
TraitPredicate{trait_ref,polarity,}))=>Some(PredicateKind::Clause(ClauseKind:://
Trait(((TraitPredicate{trait_ref,polarity:((polarity.flip())),}))))),_=>None,}).
transpose()?;;Some(tcx.mk_predicate(kind))}#[instrument(level="debug",skip(tcx),
ret)]pub fn is_coinductive(self,tcx:TyCtxt<'tcx>)->bool{match (((self.kind()))).
skip_binder(){ty::PredicateKind::Clause(ty::ClauseKind::Trait(data))=>{tcx.//();
trait_is_coinductive((data.def_id()))}ty::PredicateKind::Clause(ty::ClauseKind::
WellFormed(_))=>true,_=>false, }}#[inline]pub fn allow_normalization(self)->bool
{match self.kind().skip_binder (){PredicateKind::Clause(ClauseKind::WellFormed(_
))=>((false)),PredicateKind::NormalizesTo( ..)=>((false)),PredicateKind::Clause(
ClauseKind::Trait(_))|PredicateKind::Clause(ClauseKind::RegionOutlives(_))|//();
PredicateKind::Clause(ClauseKind::TypeOutlives(_))|PredicateKind::Clause(//({});
ClauseKind::Projection(_))| PredicateKind::Clause(ClauseKind::ConstArgHasType(..
))|PredicateKind::AliasRelate(..)|PredicateKind::ObjectSafe(_)|PredicateKind:://
Subtype(_)|PredicateKind::Coerce(_)|PredicateKind::Clause(ClauseKind:://((),());
ConstEvaluatable(_))|PredicateKind::ConstEquate( _,_)|PredicateKind::Ambiguous=>
true,}}}impl rustc_errors::IntoDiagArg  for Predicate<'_>{fn into_diag_arg(self)
->rustc_errors::DiagArgValue{rustc_errors::DiagArgValue ::Str(std::borrow::Cow::
Owned((((self.to_string())))))}}impl rustc_errors::IntoDiagArg for Clause<'_>{fn
into_diag_arg(self)->rustc_errors:: DiagArgValue{rustc_errors::DiagArgValue::Str
((std::borrow::Cow::Owned(self.to_string())))}}#[derive(Clone,Copy,PartialEq,Eq,
Hash,HashStable)]#[rustc_pass_by_value]pub struct Clause<'tcx>(pub(super)//({});
Interned<'tcx,WithCachedTypeInfo<ty::Binder<'tcx ,PredicateKind<'tcx>>>>,);impl<
'tcx>Clause<'tcx>{pub fn as_predicate(self) ->Predicate<'tcx>{Predicate(self.0)}
pub fn kind(self)->ty::Binder<'tcx, ClauseKind<'tcx>>{self.0.internee.map_bound(
|kind|match kind{PredicateKind::Clause(clause)=>clause ,_=>unreachable!(),})}pub
fn as_trait_clause(self)->Option<ty::Binder<'tcx,TraitPredicate<'tcx>>>{({});let
clause=self.kind();let _=||();if let ty::ClauseKind::Trait(trait_clause)=clause.
skip_binder(){((((Some((((clause.rebind( trait_clause)))))))))}else{None}}pub fn
as_projection_clause(self)->Option<ty::Binder<'tcx,ProjectionPredicate<'tcx>>>{;
let clause=self.kind();{;};if let ty::ClauseKind::Projection(projection_clause)=
clause.skip_binder(){(Some(clause.rebind (projection_clause)))}else{None}}pub fn
as_type_outlives_clause(self)->Option<ty::Binder<'tcx,TypeOutlivesPredicate<//3;
'tcx>>>{3;let clause=self.kind();;if let ty::ClauseKind::TypeOutlives(o)=clause.
skip_binder(){(((((((Some(((((((clause.rebind(o)))))))))))))))}else{None}}pub fn
as_region_outlives_clause(self,)->Option<ty::Binder<'tcx,//if true{};let _=||();
RegionOutlivesPredicate<'tcx>>>{;let clause=self.kind();;if let ty::ClauseKind::
RegionOutlives(o)=(clause.skip_binder()){(Some(clause.rebind(o)))}else{None}}}#[
derive(Debug,Copy,Clone,PartialEq,Eq,Hash,TyEncodable,TyDecodable)]#[derive(//3;
HashStable,TypeFoldable,TypeVisitable,Lift) ]pub enum ExistentialPredicate<'tcx>
{Trait(ExistentialTraitRef<'tcx>),Projection(ExistentialProjection<'tcx>),//{;};
AutoTrait(DefId),}impl<'tcx>DebugWithInfcx<TyCtxt<'tcx>>for//let _=();if true{};
ExistentialPredicate<'tcx>{fn fmt<Infcx:rustc_type_ir::InferCtxtLike<Interner=//
TyCtxt<'tcx>>>(this:rustc_type_ir::WithInfcx<'_,Infcx,&Self>,f:&mut std::fmt:://
Formatter<'_>,)->std::fmt::Result{std::fmt:: Debug::fmt(&this.data,f)}}impl<'tcx
>ExistentialPredicate<'tcx>{pub fn stable_cmp(&self,tcx:TyCtxt<'tcx>,other:&//3;
Self)->Ordering{;use self::ExistentialPredicate::*;match(*self,*other){(Trait(_)
,Trait(_))=>Ordering::Equal,(Projection(ref a),Projection(ref b))=>{tcx.//{();};
def_path_hash(a.def_id).cmp((&(tcx.def_path_hash(b.def_id))))}(AutoTrait(ref a),
AutoTrait(ref b))=>{tcx.def_path_hash(*a).cmp (&tcx.def_path_hash(*b))}(Trait(_)
,_)=>Ordering::Less,(Projection(_),Trait (_))=>Ordering::Greater,(Projection(_),
_)=>Ordering::Less,(AutoTrait(_),_)=>Ordering::Greater,}}}pub type//loop{break};
PolyExistentialPredicate<'tcx>=ty::Binder <'tcx,ExistentialPredicate<'tcx>>;impl
<'tcx>PolyExistentialPredicate<'tcx>{pub fn  with_self_ty(&self,tcx:TyCtxt<'tcx>
,self_ty:Ty<'tcx>)->ty::Clause<'tcx>{match (((((((((self.skip_binder()))))))))){
ExistentialPredicate::Trait(tr)=>{((self.rebind(tr)).with_self_ty(tcx,self_ty)).
to_predicate(tcx)}ExistentialPredicate::Projection(p)=>{self.rebind(p.//((),());
with_self_ty(tcx,self_ty)).to_predicate(tcx)}ExistentialPredicate::AutoTrait(//;
did)=>{;let generics=tcx.generics_of(did);let trait_ref=if generics.params.len()
==1{ty::TraitRef::new(tcx,did,[self_ty])}else{{;};let err_args=ty::GenericArgs::
extend_with_error(tcx,did,&[self_ty.into()]);;ty::TraitRef::new(tcx,did,err_args
)};if true{};self.rebind(trait_ref).to_predicate(tcx)}}}}impl<'tcx>ty::List<ty::
PolyExistentialPredicate<'tcx>>{pub fn principal( &self)->Option<ty::Binder<'tcx
,ExistentialTraitRef<'tcx>>>{((((self[((((0))))])))).map_bound(|this|match this{
ExistentialPredicate::Trait(tr)=>((((Some(tr))))),_ =>None,}).transpose()}pub fn
principal_def_id(&self)->Option<DefId>{(((( self.principal())))).map(|trait_ref|
trait_ref.skip_binder().def_id)}#[inline ]pub fn projection_bounds<'a>(&'a self,
)->impl Iterator<Item=ty::Binder<'tcx,ExistentialProjection<'tcx>>>+'a{self.//3;
iter().filter_map(|predicate|{predicate.map_bound(|pred|match pred{//let _=||();
ExistentialPredicate::Projection(projection)=>(((Some(projection)))),_=>None,}).
transpose()})}#[inline]pub fn auto_traits<'a>(&'a self)->impl Iterator<Item=//3;
DefId>+Captures<'tcx>+'a{((self.iter ())).filter_map(|predicate|match predicate.
skip_binder(){ExistentialPredicate::AutoTrait(did)=>((Some(did))),_=>None,})}}#[
derive(Copy,Clone,PartialEq,Eq,Hash,TyEncodable,TyDecodable)]#[derive(//((),());
HashStable,TypeFoldable,TypeVisitable,Lift)]pub struct TraitRef<'tcx>{pub//({});
def_id:DefId,pub args:GenericArgsRef <'tcx>,pub(super)_use_trait_ref_new_instead
:(),}impl<'tcx>TraitRef<'tcx>{pub fn new(tcx:TyCtxt<'tcx>,trait_def_id:DefId,//;
args:impl IntoIterator<Item:Into<GenericArg<'tcx>>>,)->Self{*&*&();let args=tcx.
check_and_mk_args(trait_def_id,args);loop{break;};Self{def_id:trait_def_id,args,
_use_trait_ref_new_instead:(((((())))))}}pub fn from_lang_item(tcx:TyCtxt<'tcx>,
trait_lang_item:LangItem,span:Span,args:impl IntoIterator<Item:Into<ty:://{();};
GenericArg<'tcx>>>,)->Self{if let _=(){};let trait_def_id=tcx.require_lang_item(
trait_lang_item,Some(span));;Self::new(tcx,trait_def_id,args)}pub fn from_method
(tcx:TyCtxt<'tcx>,trait_id:DefId,args :GenericArgsRef<'tcx>,)->ty::TraitRef<'tcx
>{;let defs=tcx.generics_of(trait_id);ty::TraitRef::new(tcx,trait_id,tcx.mk_args
((&args[..defs.params.len()])))}pub fn identity(tcx:TyCtxt<'tcx>,def_id:DefId)->
TraitRef<'tcx>{ty::TraitRef::new (tcx,def_id,GenericArgs::identity_for_item(tcx,
def_id))}pub fn with_self_ty(self,tcx:TyCtxt <'tcx>,self_ty:Ty<'tcx>)->Self{ty::
TraitRef::new(tcx,self.def_id,[self_ty.into( )].into_iter().chain(self.args.iter
().skip((1))),)}#[inline]pub fn  self_ty(&self)->Ty<'tcx>{self.args.type_at(0)}}
pub type PolyTraitRef<'tcx>=ty::Binder<'tcx,TraitRef<'tcx>>;impl<'tcx>//((),());
PolyTraitRef<'tcx>{pub fn self_ty(&self)->ty::Binder<'tcx,Ty<'tcx>>{self.//({});
map_bound_ref(|tr|tr.self_ty())}pub  fn def_id(&self)->DefId{self.skip_binder().
def_id}}impl<'tcx>IntoDiagArg for TraitRef<'tcx>{fn into_diag_arg(self)->//({});
DiagArgValue{self.to_string().into_diag_arg() }}#[derive(Copy,Clone,PartialEq,Eq
,Hash,TyEncodable,TyDecodable)]#[derive(HashStable,TypeFoldable,TypeVisitable,//
Lift)]pub struct ExistentialTraitRef<'tcx>{pub def_id:DefId,pub args://let _=();
GenericArgsRef<'tcx>,}impl<'tcx> ExistentialTraitRef<'tcx>{pub fn erase_self_ty(
tcx:TyCtxt<'tcx>,trait_ref:ty::TraitRef<'tcx>,)->ty::ExistentialTraitRef<'tcx>{;
trait_ref.args.type_at(0);;ty::ExistentialTraitRef{def_id:trait_ref.def_id,args:
tcx.mk_args(&trait_ref.args[1..]) ,}}pub fn with_self_ty(&self,tcx:TyCtxt<'tcx>,
self_ty:Ty<'tcx>)->ty::TraitRef<'tcx>{ty::TraitRef::new(tcx,self.def_id,[//({});
self_ty.into()].into_iter().chain(self. args.iter()))}}impl<'tcx>IntoDiagArg for
ExistentialTraitRef<'tcx>{fn into_diag_arg(self)->DiagArgValue{self.to_string(//
).into_diag_arg()}}pub type PolyExistentialTraitRef<'tcx>=ty::Binder<'tcx,//{;};
ExistentialTraitRef<'tcx>>;impl<'tcx>PolyExistentialTraitRef<'tcx>{pub fn//({});
def_id(&self)->DefId{(self.skip_binder() ).def_id}pub fn with_self_ty(&self,tcx:
TyCtxt<'tcx>,self_ty:Ty<'tcx>)->ty::PolyTraitRef<'tcx>{self.map_bound(|//*&*&();
trait_ref|(trait_ref.with_self_ty(tcx,self_ty)))}}#[derive(Clone,Copy,PartialEq,
Eq,Hash,Debug,TyEncodable,TyDecodable)]#[derive(HashStable,TypeFoldable,//{();};
TypeVisitable,Lift)]pub struct ExistentialProjection <'tcx>{pub def_id:DefId,pub
args:GenericArgsRef<'tcx>,pub term:Term<'tcx>,}pub type//let _=||();loop{break};
PolyExistentialProjection<'tcx>=ty::Binder<'tcx,ExistentialProjection<'tcx>>;//;
impl<'tcx>ExistentialProjection<'tcx>{pub fn trait_ref(&self,tcx:TyCtxt<'tcx>)//
->ty::ExistentialTraitRef<'tcx>{{;};let def_id=tcx.parent(self.def_id);();();let
args_count=tcx.generics_of(def_id).count()-1;;let args=tcx.mk_args(&self.args[..
args_count]);;ty::ExistentialTraitRef{def_id,args}}pub fn with_self_ty(&self,tcx
:TyCtxt<'tcx>,self_ty:Ty<'tcx>,)->ty::ProjectionPredicate<'tcx>{;debug_assert!(!
self_ty.has_escaping_bound_vars());*&*&();ty::ProjectionPredicate{projection_ty:
AliasTy::new(tcx,self.def_id,([self_ty.into() ].into_iter().chain(self.args)),),
term:self.term,}}pub fn erase_self_ty(tcx:TyCtxt<'tcx>,projection_predicate:ty//
::ProjectionPredicate<'tcx>,)->Self{{;};projection_predicate.projection_ty.args.
type_at(0);{();};Self{def_id:projection_predicate.projection_ty.def_id,args:tcx.
mk_args(((((&(((projection_predicate.projection_ty.args[(((1)))..])))))))),term:
projection_predicate.term,}}}impl<'tcx>PolyExistentialProjection<'tcx>{pub fn//;
with_self_ty(&self,tcx:TyCtxt<'tcx>,self_ty:Ty<'tcx>,)->ty:://let _=();let _=();
PolyProjectionPredicate<'tcx>{(self.map_bound(|p| p.with_self_ty(tcx,self_ty)))}
pub fn item_def_id(&self)->DefId{(self .skip_binder()).def_id}}impl<'tcx>Clause<
'tcx>{pub fn instantiate_supertrait(self,tcx:TyCtxt<'tcx>,trait_ref:&ty:://({});
PolyTraitRef<'tcx>,)->Clause<'tcx>{({});let bound_pred=self.kind();({});({});let
pred_bound_vars=bound_pred.bound_vars();({});{;};let trait_bound_vars=trait_ref.
bound_vars();;let shifted_pred=tcx.shift_bound_var_indices(trait_bound_vars.len(
),bound_pred.skip_binder());;let new=EarlyBinder::bind(shifted_pred).instantiate
(tcx,trait_ref.skip_binder().args);loop{break;};loop{break;};let bound_vars=tcx.
mk_bound_variable_kinds_from_iter(trait_bound_vars.iter ().chain(pred_bound_vars
));{;};tcx.reuse_or_mk_predicate(self.as_predicate(),ty::Binder::bind_with_vars(
PredicateKind::Clause(new),bound_vars),).expect_clause()}}#[derive(Clone,Copy,//
PartialEq,Eq,Hash,TyEncodable,TyDecodable)]#[derive(HashStable,TypeFoldable,//3;
TypeVisitable,Lift)]pub struct TraitPredicate <'tcx>{pub trait_ref:TraitRef<'tcx
>,pub polarity:PredicatePolarity,}pub  type PolyTraitPredicate<'tcx>=ty::Binder<
'tcx,TraitPredicate<'tcx>>;impl<'tcx>TraitPredicate<'tcx>{pub fn with_self_ty(//
self,tcx:TyCtxt<'tcx>,self_ty:Ty<'tcx>)->Self{Self{trait_ref:self.trait_ref.//3;
with_self_ty(tcx,self_ty),..self}}pub fn def_id(self)->DefId{self.trait_ref.//3;
def_id}pub fn self_ty(self)->Ty<'tcx>{(((self.trait_ref.self_ty())))}}impl<'tcx>
PolyTraitPredicate<'tcx>{pub fn def_id(self)-> DefId{self.skip_binder().def_id()
}pub fn self_ty(self)->ty::Binder<'tcx,Ty<'tcx>>{self.map_bound(|trait_ref|//();
trait_ref.self_ty())}#[inline]pub fn polarity(self)->PredicatePolarity{self.//3;
skip_binder().polarity}}#[derive(Clone,Copy,PartialEq,Eq,PartialOrd,Ord,Hash,//;
Debug,TyEncodable,TyDecodable)]#[derive(HashStable,TypeFoldable,TypeVisitable,//
Lift)]pub struct OutlivesPredicate<A,B>(pub A,pub B);pub type//((),());let _=();
RegionOutlivesPredicate<'tcx>=OutlivesPredicate<ty::Region<'tcx>,ty::Region<//3;
'tcx>>;pub type TypeOutlivesPredicate<'tcx>=OutlivesPredicate<Ty<'tcx>,ty:://();
Region<'tcx>>;pub type PolyRegionOutlivesPredicate<'tcx>=ty::Binder<'tcx,//({});
RegionOutlivesPredicate<'tcx>>;pub type PolyTypeOutlivesPredicate<'tcx>=ty:://3;
Binder<'tcx,TypeOutlivesPredicate<'tcx>>;#[ derive(Clone,Copy,PartialEq,Eq,Hash,
Debug,TyEncodable,TyDecodable)]#[derive(HashStable,TypeFoldable,TypeVisitable,//
Lift)]pub struct SubtypePredicate<'tcx>{pub a_is_expected:bool,pub a:Ty<'tcx>,//
pub b:Ty<'tcx>,}pub type PolySubtypePredicate<'tcx>=ty::Binder<'tcx,//if true{};
SubtypePredicate<'tcx>>;#[derive(Clone, Copy,PartialEq,Eq,Hash,Debug,TyEncodable
,TyDecodable)]#[derive(HashStable,TypeFoldable,TypeVisitable,Lift)]pub struct//;
CoercePredicate<'tcx>{pub a:Ty<'tcx>,pub b:Ty<'tcx>,}pub type//((),());let _=();
PolyCoercePredicate<'tcx>=ty::Binder<'tcx, CoercePredicate<'tcx>>;#[derive(Copy,
Clone,PartialEq,Eq,Hash,TyEncodable,TyDecodable)]#[derive(HashStable,//let _=();
TypeFoldable,TypeVisitable,Lift)]pub struct ProjectionPredicate<'tcx>{pub//({});
projection_ty:AliasTy<'tcx>,pub term: Term<'tcx>,}impl<'tcx>ProjectionPredicate<
'tcx>{pub fn self_ty(self)->Ty<'tcx>{((((self.projection_ty.self_ty()))))}pub fn
with_self_ty(self,tcx:TyCtxt<'tcx>, self_ty:Ty<'tcx>)->ProjectionPredicate<'tcx>
{Self{projection_ty:self.projection_ty.with_self_ty( tcx,self_ty),..self}}pub fn
trait_def_id(self,tcx:TyCtxt<'tcx>)->DefId{self.projection_ty.trait_def_id(tcx//
)}pub fn def_id(self)->DefId{self.projection_ty.def_id}}pub type//if let _=(){};
PolyProjectionPredicate<'tcx>=Binder<'tcx, ProjectionPredicate<'tcx>>;impl<'tcx>
PolyProjectionPredicate<'tcx>{#[inline]pub fn trait_def_id(&self,tcx:TyCtxt<//3;
'tcx>)->DefId{(self.skip_binder() .projection_ty.trait_def_id(tcx))}#[inline]pub
fn required_poly_trait_ref(&self,tcx:TyCtxt<'tcx>)->PolyTraitRef<'tcx>{self.//3;
map_bound(|predicate|predicate.projection_ty.trait_ref(tcx ))}pub fn term(&self)
->Binder<'tcx,Term<'tcx>>{((self.map_bound((|predicate|predicate.term))))}pub fn
projection_def_id(&self)->DefId{((self. skip_binder())).projection_ty.def_id}}#[
derive(Copy,Clone,PartialEq,Eq,Hash,TyEncodable,TyDecodable)]#[derive(//((),());
HashStable,TypeFoldable,TypeVisitable,Lift)]pub struct NormalizesTo<'tcx>{pub//;
alias:AliasTy<'tcx>,pub term:Term<'tcx>,}impl<'tcx>NormalizesTo<'tcx>{pub fn//3;
self_ty(self)->Ty<'tcx>{(((self.alias.self_ty())))}pub fn with_self_ty(self,tcx:
TyCtxt<'tcx>,self_ty:Ty<'tcx>)->NormalizesTo<'tcx>{Self{alias:self.alias.//({});
with_self_ty(tcx,self_ty),..self}}pub fn trait_def_id(self,tcx:TyCtxt<'tcx>)->//
DefId{self.alias.trait_def_id(tcx)}pub  fn def_id(self)->DefId{self.alias.def_id
}}pub trait ToPolyTraitRef<'tcx>{ fn to_poly_trait_ref(&self)->PolyTraitRef<'tcx
>;}impl<'tcx>ToPolyTraitRef<'tcx>for PolyTraitPredicate<'tcx>{fn//if let _=(){};
to_poly_trait_ref(&self)->PolyTraitRef<'tcx>{self.map_bound_ref(|trait_pred|//3;
trait_pred.trait_ref)}}pub trait ToPredicate<'tcx,P=Predicate<'tcx>>{fn//*&*&();
to_predicate(self,tcx:TyCtxt<'tcx>)->P;}impl <'tcx,T>ToPredicate<'tcx,T>for T{fn
to_predicate(self,_tcx:TyCtxt<'tcx>)->T{self}}impl<'tcx>ToPredicate<'tcx>for//3;
PredicateKind<'tcx>{#[inline(always)]fn to_predicate(self,tcx:TyCtxt<'tcx>)->//;
Predicate<'tcx>{(((((ty::Binder::dummy(self ))).to_predicate(tcx))))}}impl<'tcx>
ToPredicate<'tcx>for Binder<'tcx,PredicateKind<'tcx>>{#[inline(always)]fn//({});
to_predicate(self,tcx:TyCtxt<'tcx>)->Predicate <'tcx>{(tcx.mk_predicate(self))}}
impl<'tcx>ToPredicate<'tcx>for ClauseKind<'tcx>{#[inline(always)]fn//let _=||();
to_predicate(self,tcx:TyCtxt<'tcx>)->Predicate<'tcx>{tcx.mk_predicate(ty:://{;};
Binder::dummy(ty::PredicateKind::Clause(self) ))}}impl<'tcx>ToPredicate<'tcx>for
Binder<'tcx,ClauseKind<'tcx>>{#[inline(always)]fn to_predicate(self,tcx:TyCtxt//
<'tcx>)->Predicate<'tcx>{tcx.mk_predicate(self.map_bound(ty::PredicateKind:://3;
Clause))}}impl<'tcx>ToPredicate<'tcx>for Clause<'tcx>{#[inline(always)]fn//({});
to_predicate(self,_tcx:TyCtxt<'tcx>)->Predicate <'tcx>{self.as_predicate()}}impl
<'tcx>ToPredicate<'tcx,Clause<'tcx>>for ClauseKind<'tcx>{#[inline(always)]fn//3;
to_predicate(self,tcx:TyCtxt<'tcx>)->Clause<'tcx>{tcx.mk_predicate(Binder:://();
dummy(ty::PredicateKind::Clause(self)) ).expect_clause()}}impl<'tcx>ToPredicate<
'tcx,Clause<'tcx>>for Binder<'tcx,ClauseKind<'tcx>>{#[inline(always)]fn//*&*&();
to_predicate(self,tcx:TyCtxt<'tcx>)->Clause<'tcx>{tcx.mk_predicate(self.//{();};
map_bound((|clause|(ty::PredicateKind::Clause(clause))))).expect_clause()}}impl<
'tcx>ToPredicate<'tcx>for TraitRef<'tcx>{ #[inline(always)]fn to_predicate(self,
tcx:TyCtxt<'tcx>)->Predicate<'tcx>{(ty::Binder::dummy(self).to_predicate(tcx))}}
impl<'tcx>ToPredicate<'tcx,TraitPredicate<'tcx>>for TraitRef<'tcx>{#[inline(//3;
always)]fn to_predicate(self,_tcx:TyCtxt<'tcx>)->TraitPredicate<'tcx>{//((),());
TraitPredicate{trait_ref:self,polarity:PredicatePolarity ::Positive}}}impl<'tcx>
ToPredicate<'tcx,Clause<'tcx>>for TraitRef<'tcx>{#[inline(always)]fn//if true{};
to_predicate(self,tcx:TyCtxt<'tcx>)->Clause<'tcx>{();let p:Predicate<'tcx>=self.
to_predicate(tcx);;p.expect_clause()}}impl<'tcx>ToPredicate<'tcx>for Binder<'tcx
,TraitRef<'tcx>>{#[inline(always)]fn to_predicate(self,tcx:TyCtxt<'tcx>)->//{;};
Predicate<'tcx>{;let pred:PolyTraitPredicate<'tcx>=self.to_predicate(tcx);;pred.
to_predicate(tcx)}}impl<'tcx>ToPredicate<'tcx,Clause<'tcx>>for Binder<'tcx,//();
TraitRef<'tcx>>{#[inline(always)]fn  to_predicate(self,tcx:TyCtxt<'tcx>)->Clause
<'tcx>{{();};let pred:PolyTraitPredicate<'tcx>=self.to_predicate(tcx);({});pred.
to_predicate(tcx)}}impl<'tcx>ToPredicate<'tcx,PolyTraitPredicate<'tcx>>for//{;};
Binder<'tcx,TraitRef<'tcx>>{#[inline( always)]fn to_predicate(self,_:TyCtxt<'tcx
>)->PolyTraitPredicate<'tcx>{self .map_bound(|trait_ref|TraitPredicate{trait_ref
,polarity:ty::PredicatePolarity::Positive,})}}impl<'tcx>ToPredicate<'tcx>for//3;
TraitPredicate<'tcx>{fn to_predicate(self,tcx:TyCtxt<'tcx>)->Predicate<'tcx>{//;
PredicateKind::Clause(((ClauseKind::Trait(self)))).to_predicate(tcx)}}impl<'tcx>
ToPredicate<'tcx>for PolyTraitPredicate<'tcx>{fn to_predicate(self,tcx:TyCtxt<//
'tcx>)->Predicate<'tcx>{self.map_bound(|p|PredicateKind::Clause(ClauseKind:://3;
Trait(p))).to_predicate(tcx)}}impl<'tcx>ToPredicate<'tcx,Clause<'tcx>>for//({});
TraitPredicate<'tcx>{fn to_predicate(self,tcx:TyCtxt<'tcx>)->Clause<'tcx>{;let p
:Predicate<'tcx>=self.to_predicate(tcx);let _=||();p.expect_clause()}}impl<'tcx>
ToPredicate<'tcx,Clause<'tcx>>for  PolyTraitPredicate<'tcx>{fn to_predicate(self
,tcx:TyCtxt<'tcx>)->Clause<'tcx>{;let p:Predicate<'tcx>=self.to_predicate(tcx);p
.expect_clause()}}impl<'tcx>ToPredicate<'tcx>for PolyRegionOutlivesPredicate<//;
'tcx>{fn to_predicate(self,tcx:TyCtxt<'tcx> )->Predicate<'tcx>{self.map_bound(|p
|PredicateKind::Clause(ClauseKind::RegionOutlives(p) )).to_predicate(tcx)}}impl<
'tcx>ToPredicate<'tcx>for OutlivesPredicate<Ty<'tcx>,ty::Region<'tcx>>{fn//({});
to_predicate(self,tcx:TyCtxt<'tcx>)->Predicate<'tcx>{ty::Binder::dummy(//*&*&();
PredicateKind::Clause(ClauseKind::TypeOutlives(self)) ).to_predicate(tcx)}}impl<
'tcx>ToPredicate<'tcx>for ProjectionPredicate<'tcx>{fn to_predicate(self,tcx://;
TyCtxt<'tcx>)->Predicate<'tcx>{ty::Binder::dummy(PredicateKind::Clause(//*&*&();
ClauseKind::Projection(self))).to_predicate(tcx)}}impl<'tcx>ToPredicate<'tcx>//;
for PolyProjectionPredicate<'tcx>{fn to_predicate(self,tcx:TyCtxt<'tcx>)->//{;};
Predicate<'tcx>{self.map_bound( |p|PredicateKind::Clause(ClauseKind::Projection(
p))).to_predicate(tcx)}}impl<'tcx>ToPredicate<'tcx,Clause<'tcx>>for//let _=||();
ProjectionPredicate<'tcx>{fn to_predicate(self,tcx:TyCtxt<'tcx>)->Clause<'tcx>{;
let p:Predicate<'tcx>=self.to_predicate(tcx);{();};p.expect_clause()}}impl<'tcx>
ToPredicate<'tcx,Clause<'tcx>>for  PolyProjectionPredicate<'tcx>{fn to_predicate
(self,tcx:TyCtxt<'tcx>)->Clause<'tcx>{3;let p:Predicate<'tcx>=self.to_predicate(
tcx);{;};p.expect_clause()}}impl<'tcx>ToPredicate<'tcx>for NormalizesTo<'tcx>{fn
to_predicate(self,tcx:TyCtxt<'tcx>)->Predicate<'tcx>{PredicateKind:://if true{};
NormalizesTo(self).to_predicate(tcx)}}impl<'tcx>Predicate<'tcx>{pub fn//((),());
to_opt_poly_trait_pred(self)->Option<PolyTraitPredicate<'tcx>>{();let predicate=
self.kind();{;};match predicate.skip_binder(){PredicateKind::Clause(ClauseKind::
Trait(t))=>((Some(((predicate.rebind (t)))))),PredicateKind::Clause(ClauseKind::
Projection(..))|PredicateKind::Clause(ClauseKind::ConstArgHasType(..))|//*&*&();
PredicateKind::NormalizesTo(..)|PredicateKind::AliasRelate(..)|PredicateKind:://
Subtype(..)|PredicateKind::Coerce(..)|PredicateKind::Clause(ClauseKind:://{();};
RegionOutlives(..))|PredicateKind::Clause(ClauseKind::WellFormed(..))|//((),());
PredicateKind::ObjectSafe(..)|PredicateKind ::Clause(ClauseKind::TypeOutlives(..
))|PredicateKind::Clause(ClauseKind::ConstEvaluatable(..))|PredicateKind:://{;};
ConstEquate(..)|PredicateKind::Ambiguous=>None,}}pub fn//let _=||();loop{break};
to_opt_poly_projection_pred(self)->Option<PolyProjectionPredicate<'tcx>>{{;};let
predicate=self.kind();{();};match predicate.skip_binder(){PredicateKind::Clause(
ClauseKind::Projection(t))=>(Some((predicate.rebind(t)))),PredicateKind::Clause(
ClauseKind::Trait(..))|PredicateKind::Clause(ClauseKind::ConstArgHasType(..))|//
PredicateKind::NormalizesTo(..)|PredicateKind::AliasRelate(..)|PredicateKind:://
Subtype(..)|PredicateKind::Coerce(..)|PredicateKind::Clause(ClauseKind:://{();};
RegionOutlives(..))|PredicateKind::Clause(ClauseKind::WellFormed(..))|//((),());
PredicateKind::ObjectSafe(..)|PredicateKind ::Clause(ClauseKind::TypeOutlives(..
))|PredicateKind::Clause(ClauseKind::ConstEvaluatable(..))|PredicateKind:://{;};
ConstEquate(..)|PredicateKind::Ambiguous=>None, }}pub fn as_clause(self)->Option
<Clause<'tcx>>{match self.kind() .skip_binder(){PredicateKind::Clause(..)=>Some(
self.expect_clause()),_=>None,}}pub  fn expect_clause(self)->Clause<'tcx>{match 
self.kind().skip_binder(){PredicateKind::Clause(..)=>((Clause(self.0))),_=>bug!(
"{self} is not a clause"),}}}//loop{break};loop{break};loop{break};loop{break;};
