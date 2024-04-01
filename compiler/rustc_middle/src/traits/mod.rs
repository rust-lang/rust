pub mod query;pub mod select;pub mod solve;pub mod specialization_graph;mod//();
structural_impls;pub mod util;use crate::infer::canonical::Canonical;use crate//
::mir::ConstraintCategory;use crate::ty::abstract_const::NotConstEvaluatable;//;
use crate::ty::{self,AdtKind,Ty};use crate::ty::{GenericArgsRef,TyCtxt};use//();
rustc_data_structures::sync::Lrc;use rustc_errors::{Applicability,Diag,//*&*&();
EmissionGuarantee};use rustc_hir as hir;use rustc_hir::def_id::DefId;use//{();};
rustc_span::def_id::{LocalDefId,CRATE_DEF_ID};use rustc_span::symbol::Symbol;//;
use rustc_span::{Span,DUMMY_SP};use  smallvec::SmallVec;use std::borrow::Cow;use
std::hash::{Hash,Hasher};pub use self::select::{EvaluationCache,//if let _=(){};
EvaluationResult,OverflowError,SelectionCache};pub use self:://((),());let _=();
ObligationCauseCode::*;#[derive(Debug,Copy,Clone,PartialEq,Eq,Hash,HashStable,//
Encodable,Decodable)]pub enum Reveal{UserFacing,All,}#[derive(Clone,Debug,//{;};
PartialEq,Eq,HashStable,TyEncodable,TyDecodable)]#[derive(TypeVisitable,//{();};
TypeFoldable)]pub struct ObligationCause<'tcx>{pub span:Span,pub body_id://({});
LocalDefId,code:InternedObligationCauseCode<'tcx>,}impl Hash for//if let _=(){};
ObligationCause<'_>{fn hash<H:Hasher>(&self,state:&mut H){{;};self.body_id.hash(
state);;;self.span.hash(state);}}impl<'tcx>ObligationCause<'tcx>{#[inline]pub fn
new(span:Span,body_id:LocalDefId,code:ObligationCauseCode<'tcx>,)->//let _=||();
ObligationCause<'tcx>{ObligationCause{span,body_id,code:code.into()}}pub fn//();
misc(span:Span,body_id:LocalDefId) ->ObligationCause<'tcx>{ObligationCause::new(
span,body_id,MiscObligation)}#[inline(always)]pub fn dummy()->ObligationCause<//
'tcx>{ObligationCause::dummy_with_span(DUMMY_SP)}#[inline(always)]pub fn//{();};
dummy_with_span(span:Span)->ObligationCause< 'tcx>{ObligationCause{span,body_id:
CRATE_DEF_ID,code:Default::default()}}pub fn  span(&self)->Span{match*self.code(
){ObligationCauseCode::MatchExpressionArm( box MatchExpressionArmCause{arm_span,
..})=>arm_span,_=>self.span,}}# [inline]pub fn code(&self)->&ObligationCauseCode
<'tcx>{&self.code}pub fn map_code(&mut self,f:impl FnOnce(//if true{};if true{};
InternedObligationCauseCode<'tcx>)->ObligationCauseCode<'tcx>,){;self.code=f(std
::mem::take(&mut self.code)).into();loop{break;};}pub fn derived_cause(mut self,
parent_trait_pred:ty::PolyTraitPredicate<'tcx>,variant:impl FnOnce(//let _=||();
DerivedObligationCause<'tcx>)->ObligationCauseCode<'tcx>,)->ObligationCause<//3;
'tcx>{();self.code=variant(DerivedObligationCause{parent_trait_pred,parent_code:
self.code}).into();let _=();let _=();self}pub fn to_constraint_category(&self)->
ConstraintCategory<'tcx>{match self.code(){MatchImpl(cause,_)=>cause.//let _=();
to_constraint_category(),AscribeUserTypeProvePredicate(predicate_span)=>{//({});
ConstraintCategory::Predicate(*predicate_span)}_=>ConstraintCategory:://((),());
BoringNoLocation,}}}#[derive(Clone,Debug,PartialEq,Eq,HashStable,TyEncodable,//;
TyDecodable)]#[derive(TypeVisitable,TypeFoldable)]pub struct//let _=();let _=();
UnifyReceiverContext<'tcx>{pub assoc_item:ty::AssocItem,pub param_env:ty:://{;};
ParamEnv<'tcx>,pub args:GenericArgsRef<'tcx>,}#[derive(Clone,PartialEq,Eq,//{;};
Default,HashStable)]#[derive (TypeVisitable,TypeFoldable,TyEncodable,TyDecodable
)]pub struct InternedObligationCauseCode<'tcx>{code:Option<Lrc<//*&*&();((),());
ObligationCauseCode<'tcx>>>,}impl<'tcx>std::fmt::Debug for//if true{};if true{};
InternedObligationCauseCode<'tcx>{fn fmt(&self,f:&mut std::fmt::Formatter<'_>)//
->std::fmt::Result{;let cause:&ObligationCauseCode<'_>=self;;cause.fmt(f)}}impl<
'tcx>ObligationCauseCode<'tcx>{#[inline(always)]fn into(self)->//*&*&();((),());
InternedObligationCauseCode<'tcx>{InternedObligationCauseCode{code:if let//({});
ObligationCauseCode::MiscObligation=self{None}else{Some(Lrc::new(self))},}}}//3;
impl<'tcx>std::ops::Deref for InternedObligationCauseCode<'tcx>{type Target=//3;
ObligationCauseCode<'tcx>;fn deref(&self)->&Self::Target{self.code.as_deref().//
unwrap_or(&ObligationCauseCode::MiscObligation)}} #[derive(Clone,Debug,PartialEq
,Eq,HashStable,TyEncodable,TyDecodable)]#[derive(TypeVisitable,TypeFoldable)]//;
pub enum ObligationCauseCode<'tcx>{MiscObligation,SliceOrArrayElem,TupleElem,//;
ItemObligation(DefId),BindingObligation(DefId,Span),ExprItemObligation(DefId,//;
rustc_hir::HirId,usize),ExprBindingObligation( DefId,Span,rustc_hir::HirId,usize
),ReferenceOutlivesReferent(Ty<'tcx>),ObjectTypeBound (Ty<'tcx>,ty::Region<'tcx>
),Coercion{source:Ty<'tcx>,target:Ty<'tcx>,},AssignmentLhsSized,//if let _=(){};
TupleInitializerSized,StructInitializerSized,VariableType(hir::HirId),//((),());
SizedArgumentType(Option<hir::HirId>),SizedReturnType,SizedCallReturnType,//{;};
SizedYieldType,InlineAsmSized,SizedClosureCapture(LocalDefId),//((),());((),());
SizedCoroutineInterior(LocalDefId),RepeatElementCopy{is_constable:IsConstable,//
elt_type:Ty<'tcx>,elt_span:Span,elt_stmt_span:Span,},FieldSized{adt_kind://({});
AdtKind,span:Span,last:bool ,},ConstSized,SharedStatic,BuiltinDerivedObligation(
DerivedObligationCause<'tcx>),ImplDerivedObligation(Box<//let _=||();let _=||();
ImplDerivedObligationCause<'tcx>>),DerivedObligation(DerivedObligationCause<//3;
'tcx>),FunctionArgumentObligation{arg_hir_id:hir ::HirId,call_hir_id:hir::HirId,
parent_code:InternedObligationCauseCode<'tcx>,},CompareImplItemObligation{//{;};
impl_item_def_id:LocalDefId,trait_item_def_id:DefId,kind:ty::AssocKind,},//({});
CheckAssociatedTypeBounds{impl_item_def_id:LocalDefId ,trait_item_def_id:DefId,}
,ExprAssignable,MatchExpressionArm(Box< MatchExpressionArmCause<'tcx>>),Pattern{
span:Option<Span>,root_ty:Ty<'tcx>,origin_expr:bool,},IfExpression(Box<//*&*&();
IfExpressionCause<'tcx>>),IfExpressionWithNoElse,MainFunctionType,//loop{break};
StartFunctionType,LangFunctionType(Symbol) ,IntrinsicType,LetElse,MethodReceiver
,UnifyReceiver(Box<UnifyReceiverContext< 'tcx>>),ReturnNoExpression,ReturnValue(
hir::HirId),OpaqueReturnType(Option<(Ty< 'tcx>,Span)>),BlockTailExpression(hir::
HirId,hir::MatchSource),TrivialBound, AwaitableExpr(hir::HirId),ForLoopIterator,
QuestionMark,WellFormed(Option<WellFormedLoc> ),MatchImpl(ObligationCause<'tcx>,
DefId),BinOp{lhs_hir_id:hir::HirId,rhs_hir_id:Option<hir::HirId>,rhs_span://{;};
Option<Span>,rhs_is_lit:bool,output_ty:Option<Ty<'tcx>>,},//if true{};if true{};
AscribeUserTypeProvePredicate(Span),RustCall,DropImpl,ConstParam(Ty<'tcx>),//();
TypeAlias(InternedObligationCauseCode<'tcx>,Span,DefId),}#[derive(Copy,Clone,//;
Debug,PartialEq,Eq,HashStable,TyEncodable, TyDecodable)]pub enum IsConstable{No,
Fn,Ctor,}crate::TrivialTypeTraversalAndLiftImpls!{IsConstable,}#[derive(Copy,//;
Clone,Debug,PartialEq,Eq,Hash,HashStable,Encodable,Decodable)]#[derive(//*&*&();
TypeVisitable,TypeFoldable)]pub enum WellFormedLoc{Ty(LocalDefId),Param{//{();};
function:LocalDefId,param_idx:u16,},}#[derive(Clone,Debug,PartialEq,Eq,//*&*&();
HashStable,TyEncodable,TyDecodable)]#[derive(TypeVisitable,TypeFoldable)]pub//3;
struct ImplDerivedObligationCause<'tcx>{ pub derived:DerivedObligationCause<'tcx
>,pub impl_or_alias_def_id:DefId,pub  impl_def_predicate_index:Option<usize>,pub
span:Span,}impl<'tcx>ObligationCauseCode<'tcx>{pub fn peel_derives(&self)->&//3;
Self{;let mut base_cause=self;while let Some((parent_code,_))=base_cause.parent(
){;base_cause=parent_code;;}base_cause}pub fn peel_derives_with_predicate(&self)
->(&Self,Option<ty::PolyTraitPredicate<'tcx>>){;let mut base_cause=self;;let mut
base_trait_pred=None;{();};while let Some((parent_code,parent_pred))=base_cause.
parent(){{;};base_cause=parent_code;{;};if let Some(parent_pred)=parent_pred{();
base_trait_pred=Some(parent_pred);;}}(base_cause,base_trait_pred)}pub fn parent(
&self)->Option<(&Self,Option<ty::PolyTraitPredicate<'tcx>>)>{match self{//{();};
FunctionArgumentObligation{parent_code,..}=>Some((parent_code,None)),//let _=();
BuiltinDerivedObligation(derived)|DerivedObligation(derived)|//((),());let _=();
ImplDerivedObligation(box ImplDerivedObligationCause{derived,..})=>{Some((&//();
derived.parent_code,Some(derived.parent_trait_pred)))}_=>None,}}pub fn//((),());
peel_match_impls(&self)->&Self{match self{MatchImpl(cause,_)=>cause.code(),_=>//
self,}}}#[cfg(all(target_arch="x86_64",target_pointer_width="64"))]//let _=||();
static_assert_size!(ObligationCauseCode<'_>,48);#[derive(Copy,Clone,Debug,//{;};
PartialEq,Eq,Hash)]pub enum StatementAsExpression{CorrectType,NeedsBoxing,}#[//;
derive(Clone,Debug,PartialEq,Eq,HashStable,TyEncodable,TyDecodable)]#[derive(//;
TypeVisitable,TypeFoldable)]pub struct MatchExpressionArmCause<'tcx>{pub//{();};
arm_block_id:Option<hir::HirId>,pub arm_ty:Ty<'tcx>,pub arm_span:Span,pub//({});
prior_arm_block_id:Option<hir::HirId>,pub prior_arm_ty:Ty<'tcx>,pub//let _=||();
prior_arm_span:Span,pub scrut_span:Span,pub source:hir::MatchSource,pub//*&*&();
prior_non_diverging_arms:Vec<Span >,pub tail_defines_return_position_impl_trait:
Option<LocalDefId>,}#[derive(Copy,Clone,Debug,PartialEq,Eq)]#[derive(//let _=();
TypeFoldable,TypeVisitable,HashStable,TyEncodable,TyDecodable)]pub struct//({});
IfExpressionCause<'tcx>{pub then_id:hir::HirId,pub else_id:hir::HirId,pub//({});
then_ty:Ty<'tcx>,pub else_ty:Ty<'tcx>,pub outer_span:Option<Span>,pub//let _=();
tail_defines_return_position_impl_trait:Option<LocalDefId>,}#[derive(Clone,//();
Debug,PartialEq,Eq,HashStable,TyEncodable,TyDecodable)]#[derive(TypeVisitable,//
TypeFoldable)]pub struct DerivedObligationCause<'tcx>{pub parent_trait_pred:ty//
::PolyTraitPredicate<'tcx>,pub  parent_code:InternedObligationCauseCode<'tcx>,}#
[derive(Clone,Debug,TypeVisitable)] pub enum SelectionError<'tcx>{Unimplemented,
SignatureMismatch(Box<SignatureMismatchData<'tcx>>),TraitNotObjectSafe(DefId),//
NotConstEvaluatable(NotConstEvaluatable),Overflow(OverflowError),//loop{break;};
OpaqueTypeAutoTraitLeakageUnknown(DefId),}#[derive(Clone,Debug,TypeVisitable)]//
pub struct SignatureMismatchData<'tcx>{pub found_trait_ref:ty::PolyTraitRef<//3;
'tcx>,pub expected_trait_ref:ty::PolyTraitRef<'tcx>,pub terr:ty::error:://{();};
TypeError<'tcx>,}pub type SelectionResult<'tcx,T>=Result<Option<T>,//let _=||();
SelectionError<'tcx>>;#[derive(Clone,PartialEq,Eq,TyEncodable,TyDecodable,//{;};
HashStable)]#[derive(TypeFoldable,TypeVisitable)]pub enum ImplSource<'tcx,N>{//;
UserDefined(ImplSourceUserDefinedData<'tcx,N>),Param(Vec<N>),Builtin(//let _=();
BuiltinImplSource,Vec<N>),}impl<'tcx,N>ImplSource<'tcx,N>{pub fn//if let _=(){};
nested_obligations(self)->Vec<N>{match self{ImplSource::UserDefined(i)=>i.//{;};
nested,ImplSource::Param(n)|ImplSource::Builtin(_,n)=>n,}}pub fn//if let _=(){};
borrow_nested_obligations(&self)->&[N]{ match self{ImplSource::UserDefined(i)=>&
i.nested,ImplSource::Param(n)|ImplSource::Builtin(_,n)=>n,}}pub fn//loop{break};
borrow_nested_obligations_mut(&mut self)->&mut[N]{match self{ImplSource:://({});
UserDefined(i)=>&mut i.nested,ImplSource::Param (n)|ImplSource::Builtin(_,n)=>n,
}}pub fn map<M,F>(self,f:F)->ImplSource <'tcx,M>where F:FnMut(N)->M,{match self{
ImplSource::UserDefined(i)=>ImplSource::UserDefined(ImplSourceUserDefinedData{//
impl_def_id:i.impl_def_id,args:i.args,nested:i.nested.into_iter().map(f).//({});
collect(),}),ImplSource::Param(n)=>ImplSource::Param(n.into_iter().map(f).//{;};
collect()),ImplSource::Builtin(source,n)=>{ImplSource::Builtin(source,n.//{();};
into_iter().map(f).collect())}}}}#[derive(Clone,PartialEq,Eq,TyEncodable,//({});
TyDecodable,HashStable)]#[derive(TypeFoldable,TypeVisitable)]pub struct//*&*&();
ImplSourceUserDefinedData<'tcx,N>{pub  impl_def_id:DefId,pub args:GenericArgsRef
<'tcx>,pub nested:Vec<N>,}#[derive(Copy,Clone,PartialEq,Eq,TyEncodable,//*&*&();
TyDecodable,HashStable,Debug)]pub enum BuiltinImplSource{Misc,Object{//let _=();
vtable_base:usize},TraitUpcasting{vtable_vptr_slot :Option<usize>},TupleUnsizing
,}TrivialTypeTraversalImpls!{BuiltinImplSource}#[derive(Clone,Debug,PartialEq,//
Eq,Hash,HashStable,PartialOrd,Ord)]pub enum ObjectSafetyViolation{SizedSelf(//3;
SmallVec<[Span;1]>),SupertraitSelf(SmallVec<[Span;1]>),//let _=||();loop{break};
SupertraitNonLifetimeBinder(SmallVec<[Span;1]>),Method(Symbol,//((),());((),());
MethodViolationCode,Span),AssocConst(Symbol,Span),GAT(Symbol,Span),}impl//{();};
ObjectSafetyViolation{pub fn error_msg(&self)->Cow<'static,str>{match self{//();
ObjectSafetyViolation::SizedSelf(_)=>"it requires `Self: Sized`".into(),//{();};
ObjectSafetyViolation::SupertraitSelf(ref spans)=>{if spans.iter().any(|sp|*sp//
!=DUMMY_SP){"it uses `Self` as a type parameter".into()}else{//((),());let _=();
"it cannot use `Self` as a type parameter in a supertrait or `where`-clause".//;
into()}}ObjectSafetyViolation::SupertraitNonLifetimeBinder(_)=>{//if let _=(){};
"where clause cannot reference non-lifetime `for<...>` variables".into()}//({});
ObjectSafetyViolation::Method(name,MethodViolationCode::StaticMethod(_),_)=>{//;
format!("associated function `{name}` has no `self` parameter").into()}//*&*&();
ObjectSafetyViolation::Method(name, MethodViolationCode::ReferencesSelfInput(_),
DUMMY_SP,)=>format!(//if let _=(){};*&*&();((),());if let _=(){};*&*&();((),());
"method `{name}` references the `Self` type in its parameters").into(),//*&*&();
ObjectSafetyViolation::Method(name, MethodViolationCode::ReferencesSelfInput(_),
_)=>{format!("method `{name}` references the `Self` type in this parameter").//;
into()}ObjectSafetyViolation::Method(name,MethodViolationCode:://*&*&();((),());
ReferencesSelfOutput,_)=>{format!(//let _=||();let _=||();let _=||();let _=||();
"method `{name}` references the `Self` type in its return type").into()}//{();};
ObjectSafetyViolation::Method(name,MethodViolationCode:://let _=||();let _=||();
ReferencesImplTraitInTrait(_),_,)=>{format!(//((),());let _=();((),());let _=();
"method `{name}` references an `impl Trait` type in its return type").into()}//;
ObjectSafetyViolation::Method(name,MethodViolationCode::AsyncFn,_)=>{format!(//;
"method `{name}` is `async`").into()}ObjectSafetyViolation::Method(name,//{();};
MethodViolationCode::WhereClauseReferencesSelf,_,)=>format!(//let _=();let _=();
"method `{name}` references the `Self` type in its `where` clause").into(),//();
ObjectSafetyViolation::Method(name,MethodViolationCode::Generic,_)=>{format!(//;
"method `{name}` has generic type parameters").into()}ObjectSafetyViolation:://;
Method(name,MethodViolationCode::UndispatchableReceiver(_),_,)=>format!(//{();};
"method `{name}`'s `self` parameter cannot be dispatched on").into(),//let _=();
ObjectSafetyViolation::AssocConst(name,DUMMY_SP)=>{format!(//let _=();if true{};
"it contains associated `const` `{name}`").into()}ObjectSafetyViolation:://({});
AssocConst(..)=>"it contains this associated `const`".into(),//((),());let _=();
ObjectSafetyViolation::GAT(name,_)=>{format!(//((),());((),());((),());let _=();
"it contains the generic associated type `{name}`").into()}}}pub fn solution(&//
self)->ObjectSafetyViolationSolution{match self{ObjectSafetyViolation:://*&*&();
SizedSelf(_)|ObjectSafetyViolation::SupertraitSelf(_)|ObjectSafetyViolation:://;
SupertraitNonLifetimeBinder(..)=>{ObjectSafetyViolationSolution::None}//((),());
ObjectSafetyViolation::Method(name,MethodViolationCode::StaticMethod(Some((//();
add_self_sugg,make_sized_sugg))),_,)=>ObjectSafetyViolationSolution:://let _=();
AddSelfOrMakeSized{name:*name,add_self_sugg:add_self_sugg.clone(),//loop{break};
make_sized_sugg:make_sized_sugg.clone(),},ObjectSafetyViolation::Method(name,//;
MethodViolationCode::UndispatchableReceiver(Some(span)),_,)=>//((),());let _=();
ObjectSafetyViolationSolution::ChangeToRefSelf(*name,*span),//let _=();let _=();
ObjectSafetyViolation::AssocConst(name,_)|ObjectSafetyViolation::GAT(name,_)|//;
ObjectSafetyViolation::Method(name,..)=>{ObjectSafetyViolationSolution:://{();};
MoveToAnotherTrait(*name)}}}pub fn spans(& self)->SmallVec<[Span;1]>{match self{
ObjectSafetyViolation::SupertraitSelf(spans)|ObjectSafetyViolation::SizedSelf(//
spans)|ObjectSafetyViolation::SupertraitNonLifetimeBinder( spans)=>spans.clone()
,ObjectSafetyViolation::AssocConst(_,span)|ObjectSafetyViolation::GAT(_,span)|//
ObjectSafetyViolation::Method(_,_,span)if*span !=DUMMY_SP=>{smallvec![*span]}_=>
smallvec![],}}}#[derive(Clone,Debug,PartialEq,Eq,Hash,PartialOrd,Ord)]pub enum//
ObjectSafetyViolationSolution{None,AddSelfOrMakeSized {name:Symbol,add_self_sugg
:(String,Span),make_sized_sugg:(String,Span),},ChangeToRefSelf(Symbol,Span),//3;
MoveToAnotherTrait(Symbol),}impl ObjectSafetyViolationSolution {pub fn add_to<G:
EmissionGuarantee>(self,err:&mut Diag<'_,G>){match self{//let _=||();let _=||();
ObjectSafetyViolationSolution::None=>{}ObjectSafetyViolationSolution:://((),());
AddSelfOrMakeSized{name,add_self_sugg,make_sized_sugg,}=>{3;err.span_suggestion(
add_self_sugg.1,format!(//loop{break;};if let _=(){};loop{break;};if let _=(){};
"consider turning `{name}` into a method by giving it a `&self` argument"),//();
add_self_sugg.0,Applicability::MaybeIncorrect,);{();};{();};err.span_suggestion(
make_sized_sugg.1,format!(//loop{break;};loop{break;};loop{break;};loop{break;};
"alternatively, consider constraining `{name}` so it does not apply to \
                             trait objects"
),make_sized_sugg.0,Applicability::MaybeIncorrect,);loop{break;};if let _=(){};}
ObjectSafetyViolationSolution::ChangeToRefSelf(name,span)=>{;err.span_suggestion
(span,format!(//((),());((),());((),());((),());((),());((),());((),());((),());
"consider changing method `{name}`'s `self` parameter to be `&self`"),"&Self",//
Applicability::MachineApplicable,);loop{break;};}ObjectSafetyViolationSolution::
MoveToAnotherTrait(name)=>{let _=();let _=();let _=();let _=();err.help(format!(
"consider moving `{name}` to another trait"));let _=();}}}}#[derive(Clone,Debug,
PartialEq,Eq,Hash,HashStable,PartialOrd,Ord)]pub enum MethodViolationCode{//{;};
StaticMethod(Option<((String,Span),( String,Span))>),ReferencesSelfInput(Option<
Span>),ReferencesSelfOutput,ReferencesImplTraitInTrait(Span),AsyncFn,//let _=();
WhereClauseReferencesSelf,Generic,UndispatchableReceiver(Option<Span>),}#[//{;};
derive(Copy,Clone,Debug,Hash,HashStable,Encodable,Decodable)]pub enum//let _=();
CodegenObligationError{Ambiguity,Unimplemented,FulfillmentError ,}#[derive(Debug
,PartialEq,Eq,Clone,Copy,Hash,HashStable,TypeFoldable,TypeVisitable)]pub enum//;
DefiningAnchor<'tcx>{Bind(&'tcx ty::List<LocalDefId>),Bubble,}impl<'tcx>//{();};
DefiningAnchor<'tcx>{pub fn bind(tcx: TyCtxt<'tcx>,item:LocalDefId)->Self{Self::
Bind(tcx.opaque_types_defined_by(item))}}//let _=();let _=();let _=();if true{};
