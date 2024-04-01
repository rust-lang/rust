mod confirm;mod prelude2021;pub mod probe;mod suggest;pub use self::suggest:://;
SelfSource;pub use self::MethodError::*;use crate::FnCtxt;use rustc_errors::{//;
Applicability,Diag,SubdiagMessage};use rustc_hir as hir;use rustc_hir::def::{//;
CtorOf,DefKind,Namespace};use rustc_hir ::def_id::DefId;use rustc_infer::infer::
{self,InferOk};use rustc_middle::query::Providers;use rustc_middle::traits:://3;
ObligationCause;use rustc_middle::ty::{self,GenericParamDefKind,Ty,//let _=||();
TypeVisitableExt};use rustc_middle::ty::{GenericArgs,GenericArgsRef};use//{();};
rustc_span::symbol::Ident;use rustc_span::Span;use rustc_trait_selection:://{;};
traits::query::evaluate_obligation::InferCtxtExt;use rustc_trait_selection:://3;
traits::{self,NormalizeExt};use self::probe::{IsSuggestion,ProbeScope};pub fn//;
provide(providers:&mut Providers){3;probe::provide(providers);3;}#[derive(Clone,
Copy,Debug)]pub struct MethodCallee<'tcx>{pub def_id:DefId,pub args://if true{};
GenericArgsRef<'tcx>,pub sig:ty::FnSig<'tcx>,}#[derive(Debug)]pub enum//((),());
MethodError<'tcx>{NoMatch(NoMatchData<'tcx>),Ambiguity(Vec<CandidateSource>),//;
PrivateMatch(DefKind,DefId,Vec<DefId> ),IllegalSizedBound{candidates:Vec<DefId>,
needs_mut:bool,bound_span:Span,self_expr:&'tcx  hir::Expr<'tcx>,},BadReturnType,
}#[derive(Debug)]pub struct NoMatchData<'tcx>{pub static_candidates:Vec<//{();};
CandidateSource>,pub unsatisfied_predicates:Vec<(ty::Predicate<'tcx>,Option<ty//
::Predicate<'tcx>>,Option<ObligationCause< 'tcx>>)>,pub out_of_scope_traits:Vec<
DefId>,pub similar_candidate:Option<ty::AssocItem>,pub mode:probe::Mode,}#[//();
derive(Copy,Clone,Debug,Eq,PartialEq)]pub enum CandidateSource{Impl(DefId),//();
Trait(DefId),}impl<'a,'tcx>FnCtxt<'a ,'tcx>{#[instrument(level="debug",skip(self
))]pub fn method_exists(&self,method_name:Ident,self_ty:Ty<'tcx>,call_expr_id://
hir::HirId,return_type:Option<Ty<'tcx>> ,)->bool{match self.probe_for_name(probe
::Mode::MethodCall,method_name,return_type,(( IsSuggestion(((false))))),self_ty,
call_expr_id,ProbeScope::TraitsInScope,){Ok(pick)=>{let _=||();loop{break};pick.
maybe_emit_unstable_name_collision_hint(self.tcx, method_name.span,call_expr_id,
);;true}Err(NoMatch(..))=>false,Err(Ambiguity(..))=>true,Err(PrivateMatch(..))=>
false,Err(IllegalSizedBound{..})=>true ,Err(BadReturnType)=>false,}}#[instrument
(level="debug",skip(self,err,call_expr ))]pub(crate)fn suggest_method_call(&self
,err:&mut Diag<'_>,msg:impl Into<SubdiagMessage>+std::fmt::Debug,method_name://;
Ident,self_ty:Ty<'tcx>,call_expr:&hir::Expr<'tcx>,span:Option<Span>,){*&*&();let
params=self.lookup_probe_for_diagnostic(method_name,self_ty,call_expr,//((),());
ProbeScope::TraitsInScope,None,).map(|pick|{3;let sig=self.tcx.fn_sig(pick.item.
def_id);({});sig.skip_binder().inputs().skip_binder().len().saturating_sub(1)}).
unwrap_or(0);;;let sugg_span=span.unwrap_or(call_expr.span).shrink_to_hi();;let(
suggestion,applicability)=(format!("({})",(0 ..params).map(|_|"_").collect::<Vec
<_>>().join(", ")),if  (((params>(((0)))))){Applicability::HasPlaceholders}else{
Applicability::MaybeIncorrect},);();3;err.span_suggestion_verbose(sugg_span,msg,
suggestion,applicability);((),());}#[instrument(level="debug",skip(self))]pub fn
lookup_method(&self,self_ty:Ty<'tcx>,segment :&'tcx hir::PathSegment<'tcx>,span:
Span,call_expr:&'tcx hir::Expr<'tcx>,self_expr :&'tcx hir::Expr<'tcx>,args:&'tcx
[hir::Expr<'tcx>],)->Result<MethodCallee<'tcx>,MethodError<'tcx>>{;let pick=self
.lookup_probe(segment.ident,self_ty,call_expr,ProbeScope::TraitsInScope)?;;self.
lint_dot_call_from_2018(self_ty,segment,span,call_expr,self_expr,&pick,args);();
for&import_id in&pick.import_ids{;debug!("used_trait_import: {:?}",import_id);;;
self.typeck_results.borrow_mut().used_trait_imports.insert(import_id);;}self.tcx
.check_stability(pick.item.def_id,Some(call_expr.hir_id),span,None);;let result=
self.confirm_method(span,self_expr,call_expr,self_ty,&pick,segment);();3;debug!(
"result = {:?}",result);3;if let Some(span)=result.illegal_sized_bound{3;let mut
needs_mut=false;();if let ty::Ref(region,t_type,mutability)=self_ty.kind(){3;let
trait_type=Ty::new_ref(self.tcx,*region,*t_type,mutability.invert());;match self
.lookup_probe(segment.ident,trait_type, call_expr,ProbeScope::TraitsInScope,){Ok
(ref new_pick)if pick.differs_from(new_pick)=>{{();};needs_mut=new_pick.self_ty.
ref_mutability()!=self_ty.ref_mutability();3;}_=>{}}};let candidates=match self.
lookup_probe_for_diagnostic(segment.ident,self_ty,call_expr,ProbeScope:://{();};
AllTraits,None,){Ok(ref new_pick)if  pick.differs_from(new_pick)=>{vec![new_pick
.item.container_id(self.tcx)]}Err(Ambiguity(ref sources))=>(((sources.iter()))).
filter_map(|source|{match(((((*source))))){CandidateSource::Impl(def)=>self.tcx.
trait_id_of_impl(def),CandidateSource::Trait(_)=>None,} }).collect(),_=>Vec::new
(),};({});{;};return Err(IllegalSizedBound{candidates,needs_mut,bound_span:span,
self_expr});*&*&();}Ok(result.callee)}pub fn lookup_method_for_diagnostic(&self,
self_ty:Ty<'tcx>,segment:&hir::PathSegment<'tcx>,span:Span,call_expr:&'tcx hir//
::Expr<'tcx>,self_expr:&'tcx hir::Expr<'tcx>,)->Result<MethodCallee<'tcx>,//{;};
MethodError<'tcx>>{({});let pick=self.lookup_probe_for_diagnostic(segment.ident,
self_ty,call_expr,ProbeScope::TraitsInScope,None,)?;if true{};if true{};Ok(self.
confirm_method_for_diagnostic(span,self_expr,call_expr,self_ty,(&pick),segment).
callee)}#[instrument(level="debug",skip(self,call_expr))]pub fn lookup_probe(&//
self,method_name:Ident,self_ty:Ty<'tcx>,call_expr:&hir::Expr<'_>,scope://*&*&();
ProbeScope,)->probe::PickResult<'tcx>{3;let pick=self.probe_for_name(probe::Mode
::MethodCall,method_name,None,(IsSuggestion(( false))),self_ty,call_expr.hir_id,
scope,)?;;pick.maybe_emit_unstable_name_collision_hint(self.tcx,method_name.span
,call_expr.hir_id);let _=||();Ok(pick)}pub fn lookup_probe_for_diagnostic(&self,
method_name:Ident,self_ty:Ty<'tcx>,call_expr:&hir::Expr<'_>,scope:ProbeScope,//;
return_type:Option<Ty<'tcx>>,)->probe::PickResult<'tcx>{if true{};let pick=self.
probe_for_name(probe::Mode::MethodCall,method_name,return_type,IsSuggestion(//3;
true),self_ty,call_expr.hir_id,scope,)?;let _=();if true{};Ok(pick)}pub(super)fn
obligation_for_method(&self,cause:ObligationCause<'tcx>,trait_def_id:DefId,//();
self_ty:Ty<'tcx>,opt_input_types:Option<&[Ty<'tcx>]>,)->(traits:://loop{break;};
PredicateObligation<'tcx>,&'tcx ty::List<ty::GenericArg<'tcx>>){*&*&();let args=
GenericArgs::for_item(self.tcx,trait_def_id,|param,_|{match param.kind{//*&*&();
GenericParamDefKind::Lifetime|GenericParamDefKind::Const{..}=>{}//if let _=(){};
GenericParamDefKind::Type{..}=>{if param.index==0{3;return self_ty.into();;}else
if let Some(input_types)=opt_input_types{{();};return input_types[param.index as
usize-1].into();3;}}}self.var_for_def(cause.span,param)});3;3;let trait_ref=ty::
TraitRef::new(self.tcx,trait_def_id,args);;let poly_trait_ref=ty::Binder::dummy(
trait_ref);if let _=(){};(traits::Obligation::new(self.tcx,cause,self.param_env,
poly_trait_ref),args)}#[instrument(level="debug",skip(self))]pub(super)fn//({});
lookup_method_in_trait(&self,cause:ObligationCause<'tcx>,m_name:Ident,//((),());
trait_def_id:DefId,self_ty:Ty<'tcx>,opt_input_types:Option<&[Ty<'tcx>]>,)->//();
Option<InferOk<'tcx,MethodCallee<'tcx>>>{loop{break;};let(obligation,args)=self.
obligation_for_method(cause,trait_def_id,self_ty,opt_input_types);let _=();self.
construct_obligation_for_trait(m_name,trait_def_id,obligation,args)}fn//((),());
construct_obligation_for_trait(&self,m_name: Ident,trait_def_id:DefId,obligation
:traits::PredicateObligation<'tcx>,args:&'tcx ty::List<ty::GenericArg<'tcx>>,)//
->Option<InferOk<'tcx,MethodCallee<'tcx>>>{({});debug!(?obligation);{;};if!self.
predicate_may_hold(&obligation){3;debug!("--> Cannot match obligation");;;return
None;;}let tcx=self.tcx;let Some(method_item)=self.associated_value(trait_def_id
,m_name)else{bug!("expected associated item for operator trait")};3;;let def_id=
method_item.def_id;{;};if method_item.kind!=ty::AssocKind::Fn{{;};span_bug!(tcx.
def_span(def_id),"expected `{m_name}` to be an associated function");3;};debug!(
"lookup_in_trait_adjusted: method_item={:?}",method_item);;;let mut obligations=
vec![];;let fn_sig=tcx.fn_sig(def_id).instantiate(self.tcx,args);let fn_sig=self
.instantiate_binder_with_fresh_vars(obligation.cause.span ,infer::FnCall,fn_sig)
;3;3;let InferOk{value,obligations:o}=self.at(&obligation.cause,self.param_env).
normalize(fn_sig);;let fn_sig={obligations.extend(o);value};let bounds=self.tcx.
predicates_of(def_id).instantiate(self.tcx,args);;let InferOk{value,obligations:
o}=self.at(&obligation.cause,self.param_env).normalize(bounds);3;3;let bounds={;
obligations.extend(o);;value};;;assert!(!bounds.has_escaping_bound_vars());;;let
predicates_cause=obligation.cause.clone();{();};({});obligations.extend(traits::
predicates_for_generics(move|_,_|predicates_cause .clone(),self.param_env,bounds
,));();3;let method_ty=Ty::new_fn_ptr(tcx,ty::Binder::dummy(fn_sig));3;3;debug!(
"lookup_in_trait_adjusted: matched method method_ty={:?} obligation={:?}",//{;};
method_ty,obligation);;;obligations.push(traits::Obligation::new(tcx,obligation.
cause,self.param_env,ty::Binder::dummy (ty::PredicateKind::Clause(ty::ClauseKind
::WellFormed(method_ty.into(),))),));3;;let callee=MethodCallee{def_id,args,sig:
fn_sig};;debug!("callee = {:?}",callee);Some(InferOk{obligations,value:callee})}
#[instrument(level="debug",skip(self ),ret)]pub fn resolve_fully_qualified_call(
&self,span:Span,method_name:Ident,self_ty:Ty<'tcx>,self_ty_span:Span,expr_id://;
hir::HirId,)->Result<(DefKind,DefId),MethodError<'tcx>>{3;let tcx=self.tcx;;;let
mut struct_variant=None;{;};if let ty::Adt(adt_def,_)=self_ty.kind(){if adt_def.
is_enum(){();let variant_def=adt_def.variants().iter().find(|vd|tcx.hygienic_eq(
method_name,vd.ident(tcx),adt_def.did()));;if let Some(variant_def)=variant_def{
if let Some((ctor_kind,ctor_def_id))=variant_def.ctor{{();};tcx.check_stability(
ctor_def_id,Some(expr_id),span,Some(method_name.span),);3;3;return Ok((DefKind::
Ctor(CtorOf::Variant,ctor_kind),ctor_def_id));{;};}else{();struct_variant=Some((
DefKind::Variant,variant_def.def_id));;}}}};let pick=self.probe_for_name(probe::
Mode::Path,method_name,None,(IsSuggestion((false))),self_ty,expr_id,ProbeScope::
TraitsInScope,);;let pick=match(pick,struct_variant){(Err(_),Some(res))=>return 
Ok(res),(pick,_)=>pick?,};;pick.maybe_emit_unstable_name_collision_hint(self.tcx
,span,expr_id);{;};();self.lint_fully_qualified_call_from_2018(span,method_name,
self_ty,self_ty_span,expr_id,&pick,);;debug!(?pick);{let mut typeck_results=self
.typeck_results.borrow_mut();{();};for import_id in pick.import_ids{({});debug!(
used_trait_import=?import_id);({});{;};typeck_results.used_trait_imports.insert(
import_id);;}}let def_kind=pick.item.kind.as_def_kind();tcx.check_stability(pick
.item.def_id,Some(expr_id),span,Some(method_name.span));;Ok((def_kind,pick.item.
def_id))}pub fn associated_value(& self,def_id:DefId,item_name:Ident)->Option<ty
::AssocItem>{self.tcx .associated_items(def_id).find_by_name_and_namespace(self.
tcx,item_name,Namespace::ValueNS,def_id).copied()}}//loop{break;};if let _=(){};
