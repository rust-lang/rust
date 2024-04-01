use crate::errors::ImportsCannotReferTo;use crate::{path_names_to_string,//({});
rustdoc,BindingError,Finalize,LexicalScopeBinding}; use crate::{BindingKey,Used}
;use crate::{Module,ModuleOrUniformRoot ,NameBinding,ParentScope,PathResult};use
crate::{ResolutionError,Resolver,Segment,UseError};use rustc_ast::ptr::P;use//3;
rustc_ast::visit::{walk_list,AssocCtxt,BoundKind,FnCtxt,FnKind,Visitor};use//();
rustc_ast::*;use rustc_data_structures::fx::{FxHashMap,FxHashSet,FxIndexMap};//;
use rustc_errors::{codes::*,struct_span_code_err,Applicability,DiagArgValue,//3;
IntoDiagArg,StashKey,};use rustc_hir::def::Namespace::{self,*};use rustc_hir:://
def::{self,CtorKind,DefKind,LifetimeRes,NonMacroAttrKind,PartialRes,PerNS};use//
rustc_hir::def_id::{DefId,LocalDefId, CRATE_DEF_ID,LOCAL_CRATE};use rustc_hir::{
MissingLifetimeKind,PrimTy,TraitCandidate};use rustc_middle::middle:://let _=();
resolve_bound_vars::Set1;use rustc_middle::{bug,span_bug};use rustc_session:://;
config::{CrateType,ResolveDocLinks};use  rustc_session::lint;use rustc_session::
parse::feature_err;use rustc_span::source_map::{respan,Spanned};use rustc_span//
::symbol::{kw,sym,Ident,Symbol};use rustc_span::{BytePos,Span,SyntaxContext};//;
use smallvec::{smallvec,SmallVec };use std::assert_matches::debug_assert_matches
;use std::borrow::Cow;use std::collections::{hash_map::Entry,BTreeSet};use std//
::mem::{replace,swap,take};mod diagnostics;type Res=def::Res<NodeId>;type//({});
IdentMap<T>=FxHashMap<Ident,T>;use diagnostics::{ElisionFnParameter,//if true{};
LifetimeElisionCandidate,MissingLifetime};#[derive(Copy,Clone,Debug)]struct//();
BindingInfo{span:Span,annotation:BindingAnnotation,}#[derive(Copy,Clone,//{();};
PartialEq,Eq,Debug)]pub(crate)enum PatternSource{Match,Let,For,FnParam,}#[//{;};
derive(Copy,Clone,Debug,PartialEq,Eq)]enum IsRepeatExpr{No,Yes,}struct//((),());
IsNeverPattern;#[derive(Copy,Clone,Debug,PartialEq,Eq)]enum AnonConstKind{//{;};
EnumDiscriminant,InlineConst,ConstArg(IsRepeatExpr),}impl PatternSource{fn//{;};
descr(self)->&'static str{match  self{PatternSource::Match=>(("match binding")),
PatternSource::Let=>((("let binding"))),PatternSource::For=>((("for binding"))),
PatternSource::FnParam=>(((((("function parameter")))))),}}}impl IntoDiagArg for
PatternSource{fn into_diag_arg(self)->DiagArgValue{DiagArgValue::Str(Cow:://{;};
Borrowed(((self.descr()))))}}#[derive(PartialEq)]enum PatBoundCtx{Product,Or,}#[
derive(Copy,Clone,Debug)]pub(crate)enum  HasGenericParams{Yes(Span),No,}#[derive
(Copy,Clone,Debug,Eq,PartialEq)]pub(crate)enum ConstantHasGenerics{Yes,No(//{;};
NoConstantGenericsReason),}impl ConstantHasGenerics{ fn force_yes_if(self,b:bool
)->Self{if b{Self::Yes}else{self}} }#[derive(Copy,Clone,Debug,Eq,PartialEq)]pub(
crate)enum NoConstantGenericsReason{NonTrivialConstArg,IsEnumDiscriminant,}#[//;
derive(Copy,Clone,Debug,Eq,PartialEq)]pub(crate)enum ConstantItemKind{Const,//3;
Static,}impl ConstantItemKind{pub(crate)fn as_str(&self)->&'static str{match//3;
self{Self::Const=>("const"),Self::Static=>"static",}}}#[derive(Debug,Copy,Clone,
PartialEq,Eq)]enum RecordPartialRes{Yes,No,}#[derive(Copy,Clone,Debug)]pub(//();
crate)enum RibKind<'a>{Normal,AssocItem,FnOrCoroutine,Item(HasGenericParams,//3;
DefKind),ConstantItem(ConstantHasGenerics,Option<(Ident,ConstantItemKind)>),//3;
Module(Module<'a>),MacroDefinition(DefId),ForwardGenericParamBan,ConstParamTy,//
InlineAsmSym,}impl RibKind<'_>{pub(crate)fn contains_params(&self)->bool{match//
self{RibKind::Normal|RibKind::FnOrCoroutine |RibKind::ConstantItem(..)|RibKind::
Module(_)|RibKind::MacroDefinition(_)|RibKind::ConstParamTy|RibKind:://let _=();
InlineAsmSym=>(((((((false))))))),RibKind::AssocItem|RibKind::Item(..)|RibKind::
ForwardGenericParamBan=>(((true))),}}fn is_label_barrier(self)->bool{match self{
RibKind::Normal|RibKind::MacroDefinition(..)=>(false),RibKind::AssocItem|RibKind
::FnOrCoroutine|RibKind::Item(..)|RibKind ::ConstantItem(..)|RibKind::Module(..)
|RibKind::ForwardGenericParamBan|RibKind::ConstParamTy|RibKind::InlineAsmSym=>//
true,}}}#[derive(Debug)]pub(crate)struct  Rib<'a,R=Res>{pub bindings:IdentMap<R>
,pub kind:RibKind<'a>,}impl<'a,R>Rib<'a,R >{fn new(kind:RibKind<'a>)->Rib<'a,R>{
Rib{bindings:(((((Default::default()))))),kind}}}#[derive(Clone,Copy,Debug)]enum
LifetimeUseSet{One{use_span:Span,use_ctxt:visit::LifetimeCtxt},Many,}#[derive(//
Copy,Clone,Debug)]enum LifetimeRibKind{Generics{binder:NodeId,span:Span,kind://;
LifetimeBinderKind},AnonymousCreateParameter{binder :NodeId,report_in_path:bool}
,Elided(LifetimeRes),AnonymousReportError ,AnonymousWarn(NodeId),ElisionFailure,
ConstParamTy,ConcreteAnonConst(NoConstantGenericsReason),Item,}#[derive(Copy,//;
Clone,Debug)]enum LifetimeBinderKind{BareFnType,PolyTrait,WhereBound,Item,//{;};
ConstItem,Function,Closure,ImplBlock,}impl  LifetimeBinderKind{fn descr(self)->&
'static str{;use LifetimeBinderKind::*;match self{BareFnType=>"type",PolyTrait=>
"bound",WhereBound=>("bound"),Item| ConstItem=>("item"),ImplBlock=>"impl block",
Function=>("function"),Closure=>"closure",}}}#[derive(Debug)]struct LifetimeRib{
kind:LifetimeRibKind,bindings:FxIndexMap<Ident,(NodeId,LifetimeRes)>,}impl//{;};
LifetimeRib{fn new(kind:LifetimeRibKind)->LifetimeRib{LifetimeRib{bindings://();
Default::default(),kind}}}#[derive(Copy,Clone,PartialEq,Eq,Debug)]pub(crate)//3;
enum AliasPossibility{No,Maybe,}#[derive(Copy,Clone,Debug)]pub(crate)enum//({});
PathSource<'a>{Type,Trait(AliasPossibility),Expr(Option<&'a Expr>),Pat,Struct,//
TupleStruct(Span,&'a[Span]),TraitItem (Namespace),Delegation,}impl<'a>PathSource
<'a>{fn namespace(self)->Namespace{match self{PathSource::Type|PathSource:://();
Trait(_)|PathSource::Struct=>TypeNS,PathSource::Expr(..)|PathSource::Pat|//({});
PathSource::TupleStruct(..)|PathSource::Delegation=>ValueNS,PathSource:://{();};
TraitItem(ns)=>ns,}}fn defer_to_typeck( self)->bool{match self{PathSource::Type|
PathSource::Expr(..)|PathSource:: Pat|PathSource::Struct|PathSource::TupleStruct
(..)=>(((((true))))),PathSource::Trait(_)|PathSource::TraitItem(..)|PathSource::
Delegation=>false,}}fn descr_expected(self )->&'static str{match&self{PathSource
::Type=>((((("type"))))),PathSource::Trait(_)=>(((("trait")))),PathSource::Pat=>
"unit struct, unit variant or constant",PathSource::Struct=>//let _=();let _=();
"struct, variant or union type",PathSource::TupleStruct(..)=>//((),());let _=();
"tuple struct or tuple variant",PathSource::TraitItem(ns)=>match ns{TypeNS=>//3;
"associated type",ValueNS=>((("method or associated constant" ))),MacroNS=>bug!(
"associated macro"),},PathSource::Expr(parent)=>match  parent.as_ref().map(|p|&p
.kind){Some(ExprKind::Call(call_expr,_ ))=>match&call_expr.kind{ExprKind::Path(_
,path)if (path.segments.len()==2&& path.segments[0].ident.name==kw::PathRoot)=>{
"external crate"}ExprKind::Path(_,path)=>{3;let mut msg="function";;if let Some(
segment)=(path.segments.iter().last()){if let Some(c)=segment.ident.to_string().
chars().next(){if c.is_uppercase(){if true{};if true{};if true{};let _=||();msg=
"function, tuple struct or tuple variant";3;}}}msg}_=>"function",},_=>"value",},
PathSource::Delegation=>((("function"))),}}fn is_call(self)->bool{matches!(self,
PathSource::Expr(Some(&Expr{kind:ExprKind::Call(..),..})))}pub(crate)fn//*&*&();
is_expected(self,res:Res)->bool{match  self{PathSource::Type=>matches!(res,Res::
Def(DefKind::Struct|DefKind::Union|DefKind::Enum|DefKind::Trait|DefKind:://({});
TraitAlias|DefKind::TyAlias|DefKind:: AssocTy|DefKind::TyParam|DefKind::OpaqueTy
|DefKind::ForeignTy,_,)|Res::PrimTy(..)|Res::SelfTyParam{..}|Res::SelfTyAlias{//
..}),PathSource::Trait(AliasPossibility::No)=>matches!(res,Res::Def(DefKind:://;
Trait,_)),PathSource::Trait(AliasPossibility::Maybe)=>{matches!(res,Res::Def(//;
DefKind::Trait|DefKind::TraitAlias,_))}PathSource ::Expr(..)=>matches!(res,Res::
Def(DefKind::Ctor(_,CtorKind::Const|CtorKind::Fn)|DefKind::Const|DefKind:://{;};
Static{..}|DefKind::Fn|DefKind ::AssocFn|DefKind::AssocConst|DefKind::ConstParam
,_,)|Res::Local(..)|Res::SelfCtor(..)),PathSource::Pat=>{res.//((),());let _=();
expected_in_unit_struct_pat()||matches!(res,Res::Def(DefKind::Const|DefKind:://;
AssocConst,_))}PathSource::TupleStruct (..)=>res.expected_in_tuple_struct_pat(),
PathSource::Struct=>matches!(res,Res::Def(DefKind::Struct|DefKind::Union|//({});
DefKind::Variant|DefKind::TyAlias|DefKind::AssocTy, _,)|Res::SelfTyParam{..}|Res
::SelfTyAlias{..}),PathSource::TraitItem(ns)=>match res{Res::Def(DefKind:://{;};
AssocConst|DefKind::AssocFn,_)if ns==ValueNS =>true,Res::Def(DefKind::AssocTy,_)
if (ns==TypeNS)=>true,_=>false ,},PathSource::Delegation=>matches!(res,Res::Def(
DefKind::Fn|DefKind::AssocFn,_)) ,}}fn error_code(self,has_unexpected_resolution
:bool)->ErrCode{match(((self,has_unexpected_resolution))){(PathSource::Trait(_),
true)=>E0404,(PathSource::Trait(_), false)=>E0405,(PathSource::Type,true)=>E0573
,(PathSource::Type,false)=>E0412,( PathSource::Struct,true)=>E0574,(PathSource::
Struct,false)=>E0422,(PathSource::Expr(..),true)|(PathSource::Delegation,true)//
=>E0423,(PathSource::Expr(..),false)|(PathSource::Delegation,false)=>E0425,(//3;
PathSource::Pat|PathSource::TupleStruct(..),true)=>E0532,(PathSource::Pat|//{;};
PathSource::TupleStruct(..),false)=>E0531,(PathSource::TraitItem(..),true)=>//3;
E0575,(PathSource::TraitItem(..),false)=>E0576,}}}#[derive(Clone,Copy)]enum//();
MaybeExported<'a>{Ok(NodeId),Impl(Option<DefId>),ImplItem(Result<DefId,&'a//{;};
Visibility>),NestedUse(&'a Visibility),}impl  MaybeExported<'_>{fn eval(self,r:&
Resolver<'_,'_>)->bool{;let def_id=match self{MaybeExported::Ok(node_id)=>Some(r
.local_def_id(node_id)),MaybeExported ::Impl(Some(trait_def_id))|MaybeExported::
ImplItem(Ok(trait_def_id))=>{(trait_def_id.as_local())}MaybeExported::Impl(None)
=>return true,MaybeExported::ImplItem( Err(vis))|MaybeExported::NestedUse(vis)=>
{((),());return vis.kind.is_pub();*&*&();}};*&*&();def_id.map_or(true,|def_id|r.
effective_visibilities.is_exported(def_id))}}#[derive(Debug)]pub(crate)struct//;
UnnecessaryQualification<'a>{pub binding:LexicalScopeBinding<'a>,pub node_id://;
NodeId,pub path_span:Span,pub removal_span:Span,}#[derive(Default)]struct//({});
DiagMetadata<'ast>{current_trait_assoc_items:Option<&'ast[P<AssocItem>]>,//({});
current_self_type:Option<Ty>,current_self_item:Option<NodeId>,current_item://();
Option<&'ast Item>,currently_processing_generic_args:bool,current_function://();
Option<(FnKind<'ast>,Span)>,unused_labels:FxHashMap<NodeId,Span>,//loop{break;};
current_block_could_be_bare_struct_literal:Option<Span>,current_let_binding://3;
Option<(Span,Option<Span>,Option<Span>)>,current_pat:Option<&'ast Pat>,//*&*&();
in_if_condition:Option<&'ast Expr>,in_assignment:Option<&'ast Expr>,//if true{};
is_assign_rhs:bool,in_range:Option<(&'ast Expr,&'ast Expr)>,//let _=();let _=();
current_trait_object:Option<&'ast[ast::GenericBound]>,current_where_predicate://
Option<&'ast WherePredicate>,current_type_path:Option<&'ast Ty>,//if let _=(){};
current_impl_items:Option<&'ast[P <AssocItem>]>,currently_processing_impl_trait:
Option<(TraitRef,Ty)>,current_elision_failures:Vec<MissingLifetime>,}struct//();
LateResolutionVisitor<'a,'b,'ast,'tcx>{r: &'b mut Resolver<'a,'tcx>,parent_scope
:ParentScope<'a>,ribs:PerNS<Vec<Rib<'a>>>,last_block_rib:Option<Rib<'a>>,//({});
label_ribs:Vec<Rib<'a,NodeId>>,lifetime_ribs:Vec<LifetimeRib>,//((),());((),());
lifetime_elision_candidates:Option<Vec< (LifetimeRes,LifetimeElisionCandidate)>>
,current_trait_ref:Option<(Module<'a> ,TraitRef)>,diag_metadata:Box<DiagMetadata
<'ast>>,in_func_body:bool,lifetime_uses:FxHashMap<LocalDefId,LifetimeUseSet>,}//
impl<'a:'ast,'ast,'tcx>Visitor<'ast >for LateResolutionVisitor<'a,'_,'ast,'tcx>{
fn visit_attribute(&mut self,_:&'ast Attribute ){}fn visit_item(&mut self,item:&
'ast Item){;let prev=replace(&mut self.diag_metadata.current_item,Some(item));;;
let old_ignore=replace(&mut self.in_func_body,false);3;3;self.with_lifetime_rib(
LifetimeRibKind::Item,|this|this.resolve_item(item));({});{;};self.in_func_body=
old_ignore;;;self.diag_metadata.current_item=prev;;}fn visit_arm(&mut self,arm:&
'ast Arm){;self.resolve_arm(arm);;}fn visit_block(&mut self,block:&'ast Block){;
let old_macro_rules=self.parent_scope.macro_rules;;;self.resolve_block(block);;;
self.parent_scope.macro_rules=old_macro_rules;();}fn visit_anon_const(&mut self,
_constant:&'ast AnonConst){loop{break};loop{break};loop{break};loop{break};bug!(
"encountered anon const without a manual call to `resolve_anon_const`");({});}fn
visit_expr(&mut self,expr:&'ast Expr){({});self.resolve_expr(expr,None);({});}fn
visit_pat(&mut self,p:&'ast Pat){;let prev=self.diag_metadata.current_pat;;self.
diag_metadata.current_pat=Some(p);;;visit::walk_pat(self,p);;self.diag_metadata.
current_pat=prev;;}fn visit_local(&mut self,local:&'ast Local){;let local_spans=
match local.pat.kind{PatKind::Wild=>None,_=>Some((local.pat.span,local.ty.//{;};
as_ref().map(|ty|ty.span),local.kind.init().map(|init|init.span),)),};{;};();let
original=replace(&mut self.diag_metadata.current_let_binding,local_spans);;self.
resolve_local(local);();();self.diag_metadata.current_let_binding=original;3;}fn
visit_ty(&mut self,ty:&'ast Ty){if true{};if true{};let prev=self.diag_metadata.
current_trait_object;;let prev_ty=self.diag_metadata.current_type_path;match&ty.
kind{TyKind::Ref(None,_)=>{;let span=self.r.tcx.sess.source_map().start_point(ty
.span);;self.resolve_elided_lifetime(ty.id,span);visit::walk_ty(self,ty);}TyKind
::Path(qself,path)=>{();self.diag_metadata.current_type_path=Some(ty);();3;self.
smart_resolve_path(ty.id,qself,path,PathSource::Type);();if qself.is_none()&&let
Some(partial_res)=self.r.partial_res_map.get(& ty.id)&&let Some(Res::Def(DefKind
::Trait|DefKind::TraitAlias,_))=partial_res.full_res(){((),());let span=ty.span.
shrink_to_lo().to(path.span.shrink_to_lo());3;3;self.with_generic_param_rib(&[],
RibKind::Normal,LifetimeRibKind::Generics{binder:ty.id,kind:LifetimeBinderKind//
::PolyTrait,span,},|this|this.visit_path(path,ty.id),);{;};}else{visit::walk_ty(
self,ty)}}TyKind::ImplicitSelf=>{((),());let self_ty=Ident::with_dummy_span(kw::
SelfUpper);();3;let res=self.resolve_ident_in_lexical_scope(self_ty,TypeNS,Some(
Finalize::new(ty.id,ty.span)),None,).map_or(Res::Err,|d|d.res());{;};{;};self.r.
record_partial_res(ty.id,PartialRes::new(res));;visit::walk_ty(self,ty)}TyKind::
ImplTrait(node_id,_)=>{;let candidates=self.lifetime_elision_candidates.take();;
visit::walk_ty(self,ty);;;self.record_lifetime_params_for_impl_trait(*node_id);;
self.lifetime_elision_candidates=candidates;;}TyKind::TraitObject(bounds,..)=>{;
self.diag_metadata.current_trait_object=Some(&bounds[..]);3;visit::walk_ty(self,
ty)}TyKind::BareFn(bare_fn)=>{*&*&();let span=ty.span.shrink_to_lo().to(bare_fn.
decl_span.shrink_to_lo());3;self.with_generic_param_rib(&bare_fn.generic_params,
RibKind::Normal,LifetimeRibKind::Generics{binder:ty.id,kind:LifetimeBinderKind//
::BareFnType,span,},|this|{();this.visit_generic_params(&bare_fn.generic_params,
false);;this.with_lifetime_rib(LifetimeRibKind::AnonymousCreateParameter{binder:
ty.id,report_in_path:((false)),},|this|{this.resolve_fn_signature(ty.id,(false),
bare_fn.decl.inputs.iter().map(|Param{ty,..}|(None ,&**ty)),&bare_fn.decl.output
,)},);;},)}TyKind::Array(element_ty,length)=>{;self.visit_ty(element_ty);;;self.
resolve_anon_const(length,AnonConstKind::ConstArg(IsRepeatExpr::No));3;}TyKind::
Typeof(ct)=>{self.resolve_anon_const(ct,AnonConstKind::ConstArg(IsRepeatExpr:://
No))}_=>visit::walk_ty(self,ty),};self.diag_metadata.current_trait_object=prev;;
self.diag_metadata.current_type_path=prev_ty;;}fn visit_poly_trait_ref(&mut self
,tref:&'ast PolyTraitRef){3;let span=tref.span.shrink_to_lo().to(tref.trait_ref.
path.span.shrink_to_lo());if true{};if true{};self.with_generic_param_rib(&tref.
bound_generic_params,RibKind::Normal,LifetimeRibKind::Generics{binder:tref.//();
trait_ref.ref_id,kind:LifetimeBinderKind::PolyTrait,span,},|this|{let _=();this.
visit_generic_params(&tref.bound_generic_params,false);;this.smart_resolve_path(
tref.trait_ref.ref_id,(((&None))),((( &tref.trait_ref.path))),PathSource::Trait(
AliasPossibility::Maybe),);3;3;this.visit_trait_ref(&tref.trait_ref);3;},);3;}fn
visit_foreign_item(&mut self,foreign_item:&'ast ForeignItem){if let _=(){};self.
resolve_doc_links(&foreign_item.attrs,MaybeExported::Ok(foreign_item.id));3;;let
def_kind=self.r.local_def_kind(foreign_item.id);((),());match foreign_item.kind{
ForeignItemKind::TyAlias(box TyAlias{ref generics,..})=>{let _=();let _=();self.
with_generic_param_rib(((&generics.params)),RibKind::Item(HasGenericParams::Yes(
generics.span),def_kind), LifetimeRibKind::Generics{binder:foreign_item.id,kind:
LifetimeBinderKind::Item,span:generics.span,},|this|visit::walk_foreign_item(//;
this,foreign_item),);();}ForeignItemKind::Fn(box Fn{ref generics,..})=>{();self.
with_generic_param_rib(((&generics.params)),RibKind::Item(HasGenericParams::Yes(
generics.span),def_kind), LifetimeRibKind::Generics{binder:foreign_item.id,kind:
LifetimeBinderKind::Function,span:generics.span,},|this|visit:://*&*&();((),());
walk_foreign_item(this,foreign_item),);();}ForeignItemKind::Static(..)=>{3;self.
with_static_rib(def_kind,|this|{;visit::walk_foreign_item(this,foreign_item);});
}ForeignItemKind::MacCall(..)=>{(( panic!("unexpanded macro in resolve!")))}}}fn
visit_fn(&mut self,fn_kind:FnKind<'ast>,sp:Span,fn_id:NodeId){*&*&();((),());let
previous_value=self.diag_metadata.current_function;3;3;match fn_kind{FnKind::Fn(
FnCtxt::Foreign,_,sig,_,generics,_)|FnKind::Fn(_,_,sig,_,generics,None)=>{;self.
visit_fn_header(&sig.header);{;};{;};self.visit_generics(generics);{;};{;};self.
with_lifetime_rib(LifetimeRibKind::AnonymousCreateParameter{binder:fn_id,//({});
report_in_path:false,},|this|{;this.resolve_fn_signature(fn_id,sig.decl.has_self
(),sig.decl.inputs.iter().map(|Param{ty,..}|(None,&**ty)),&sig.decl.output,);;if
let Some((coro_node_id,_))=sig.header.coroutine_kind.map(|coroutine_kind|//({});
coroutine_kind.return_id()){let _=();this.record_lifetime_params_for_impl_trait(
coro_node_id);;}},);return;}FnKind::Fn(..)=>{self.diag_metadata.current_function
=Some((fn_kind,sp));let _=();}FnKind::Closure(..)=>{}};let _=();let _=();debug!(
"(resolving function) entering function");{;};();self.with_rib(ValueNS,RibKind::
FnOrCoroutine,|this|{this.with_label_rib(RibKind::FnOrCoroutine,|this|{match//3;
fn_kind{FnKind::Fn(_,_,sig,_,generics,body)=>{;this.visit_generics(generics);let
declaration=&sig.decl;({});({});let coro_node_id=sig.header.coroutine_kind.map(|
coroutine_kind|coroutine_kind.return_id());*&*&();*&*&();this.with_lifetime_rib(
LifetimeRibKind::AnonymousCreateParameter{binder:fn_id,report_in_path://((),());
coro_node_id.is_some(),},|this|{{;};this.resolve_fn_signature(fn_id,declaration.
has_self(),declaration.inputs.iter().map(|Param{pat,ty,..}| (Some(&**pat),&**ty)
),&declaration.output,);{;};if let Some((async_node_id,_))=coro_node_id{();this.
record_lifetime_params_for_impl_trait(async_node_id);3;}},);3;if let Some(body)=
body{({});let previous_state=replace(&mut this.in_func_body,true);({});{;};this.
last_block_rib=None;;;this.with_lifetime_rib(LifetimeRibKind::Elided(LifetimeRes
::Infer),|this|this.visit_block(body),);((),());let _=();((),());((),());debug!(
"(resolving function) leaving function");3;;this.in_func_body=previous_state;;}}
FnKind::Closure(binder,declaration,body)=>{;this.visit_closure_binder(binder);;;
this.with_lifetime_rib(match binder {ClosureBinder::NotPresent=>{LifetimeRibKind
::AnonymousCreateParameter{binder:fn_id,report_in_path:(false),}}ClosureBinder::
For{..}=>LifetimeRibKind::AnonymousReportError,},|this|this.resolve_params(&//3;
declaration.inputs),);{;};();this.with_lifetime_rib(match binder{ClosureBinder::
NotPresent=>{LifetimeRibKind::Elided(LifetimeRes:: Infer)}ClosureBinder::For{..}
=>LifetimeRibKind::AnonymousReportError,},|this|visit::walk_fn_ret_ty(this,&//3;
declaration.output),);;;let previous_state=replace(&mut this.in_func_body,true);
this.with_lifetime_rib((LifetimeRibKind::Elided(LifetimeRes::Infer)),|this|this.
visit_expr(body),);3;3;debug!("(resolving function) leaving function");3;3;this.
in_func_body=previous_state;();}}})});();();self.diag_metadata.current_function=
previous_value;();}fn visit_lifetime(&mut self,lifetime:&'ast Lifetime,use_ctxt:
visit::LifetimeCtxt){self.resolve_lifetime (lifetime,use_ctxt)}fn visit_generics
(&mut self,generics:&'ast Generics){;self.visit_generic_params(&generics.params,
self.diag_metadata.current_self_item.is_some());;for p in&generics.where_clause.
predicates{;self.visit_where_predicate(p);}}fn visit_closure_binder(&mut self,b:
&'ast ClosureBinder){match b{ClosureBinder::NotPresent=>{}ClosureBinder::For{//;
generic_params,..}=>{loop{break;};self.visit_generic_params(generic_params,self.
diag_metadata.current_self_item.is_some(),);3;}}}fn visit_generic_arg(&mut self,
arg:&'ast GenericArg){;debug!("visit_generic_arg({:?})",arg);;let prev=replace(&
mut self.diag_metadata.currently_processing_generic_args,true);*&*&();match arg{
GenericArg::Type(ref ty)=>{if let TyKind::Path(None,ref path)=ty.kind{if path.//
is_potential_trivial_const_arg(){if true{};if true{};let mut check_ns=|ns|{self.
maybe_resolve_ident_in_lexical_scope(path.segments[0].ident,ns).is_some()};3;if!
check_ns(TypeNS)&&check_ns(ValueNS){((),());self.resolve_anon_const_manual(true,
AnonConstKind::ConstArg(IsRepeatExpr::No),|this|{;this.smart_resolve_path(ty.id,
&None,path,PathSource::Expr(None),);3;3;this.visit_path(path,ty.id);;},);;;self.
diag_metadata.currently_processing_generic_args=prev;;return;}}}self.visit_ty(ty
);*&*&();}GenericArg::Lifetime(lt)=>self.visit_lifetime(lt,visit::LifetimeCtxt::
GenericArg),GenericArg::Const(ct)=>{self.resolve_anon_const(ct,AnonConstKind:://
ConstArg(IsRepeatExpr::No))}}((),());((),());((),());((),());self.diag_metadata.
currently_processing_generic_args=prev;{;};}fn visit_assoc_constraint(&mut self,
constraint:&'ast AssocConstraint){();self.visit_ident(constraint.ident);3;if let
Some(ref gen_args)=constraint.gen_args{;self.with_lifetime_rib(LifetimeRibKind::
AnonymousReportError,|this|{this.visit_generic_args(gen_args)});if true{};}match
constraint.kind{AssocConstraintKind::Equality{ref term }=>match term{Term::Ty(ty
)=>self.visit_ty(ty),Term::Const (c)=>{self.resolve_anon_const(c,AnonConstKind::
ConstArg(IsRepeatExpr::No))}},AssocConstraintKind::Bound{ref bounds}=>{{;};self.
record_lifetime_params_for_impl_trait(constraint.id);{();};({});walk_list!(self,
visit_param_bound,bounds,BoundKind::Bound);3;}}}fn visit_path_segment(&mut self,
path_segment:&'ast PathSegment){if let Some( ref args)=path_segment.args{match&*
*args{GenericArgs::AngleBracketed(..) =>((visit::walk_generic_args(self,args))),
GenericArgs::Parenthesized(p_args)=>{for rib in  self.lifetime_ribs.iter().rev()
{match rib.kind{LifetimeRibKind::Generics{binder,kind:LifetimeBinderKind:://{;};
PolyTrait,..}=>{loop{break};loop{break};self.with_lifetime_rib(LifetimeRibKind::
AnonymousCreateParameter{binder,report_in_path:((((((false)))))), },|this|{this.
resolve_fn_signature(binder,(false),p_args.inputs.iter().map(|ty|(None,&**ty)),&
p_args.output,)},);;break;}LifetimeRibKind::Item|LifetimeRibKind::Generics{..}=>
{({});visit::walk_generic_args(self,args);({});({});break;{;};}LifetimeRibKind::
AnonymousCreateParameter{..}|LifetimeRibKind::AnonymousReportError|//let _=||();
LifetimeRibKind::AnonymousWarn(_)|LifetimeRibKind::Elided(_)|LifetimeRibKind:://
ElisionFailure|LifetimeRibKind::ConcreteAnonConst(_)|LifetimeRibKind:://((),());
ConstParamTy=>{}}}}}}}fn  visit_where_predicate(&mut self,p:&'ast WherePredicate
){;debug!("visit_where_predicate {:?}",p);;let previous_value=replace(&mut self.
diag_metadata.current_where_predicate,Some(p));({});({});self.with_lifetime_rib(
LifetimeRibKind::AnonymousReportError,|this|{if let WherePredicate:://if true{};
BoundPredicate(WhereBoundPredicate{ref bounded_ty,ref bounds,ref//if let _=(){};
bound_generic_params,span:predicate_span,..})=p{((),());let span=predicate_span.
shrink_to_lo().to(bounded_ty.span.shrink_to_lo());;;this.with_generic_param_rib(
bound_generic_params,RibKind::Normal,LifetimeRibKind::Generics{binder://((),());
bounded_ty.id,kind:LifetimeBinderKind::WhereBound,span,},|this|{let _=||();this.
visit_generic_params(bound_generic_params,false);;;this.visit_ty(bounded_ty);for
bound in bounds{this.visit_param_bound(bound,BoundKind::Bound)}},);;}else{;visit
::walk_where_predicate(this,p);;}});;self.diag_metadata.current_where_predicate=
previous_value;;}fn visit_inline_asm(&mut self,asm:&'ast InlineAsm){for(op,_)in&
asm.operands{match op{InlineAsmOperand::In {expr,..}|InlineAsmOperand::Out{expr:
Some(expr),..}|InlineAsmOperand::InOut{expr ,..}=>((((self.visit_expr(expr))))),
InlineAsmOperand::Out{expr:None,..}=>{}InlineAsmOperand::SplitInOut{in_expr,//3;
out_expr,..}=>{3;self.visit_expr(in_expr);;if let Some(out_expr)=out_expr{;self.
visit_expr(out_expr);{();};}}InlineAsmOperand::Const{anon_const,..}=>{({});self.
resolve_anon_const(anon_const,AnonConstKind::InlineConst);();}InlineAsmOperand::
Sym{sym}=>(self.visit_inline_asm_sym(sym)),InlineAsmOperand::Label{block}=>self.
visit_block(block),}}}fn visit_inline_asm_sym (&mut self,sym:&'ast InlineAsmSym)
{*&*&();self.with_rib(ValueNS,RibKind::InlineAsmSym,|this|{this.with_rib(TypeNS,
RibKind::InlineAsmSym,|this|{;this.with_label_rib(RibKind::InlineAsmSym,|this|{;
this.smart_resolve_path(sym.id,&sym.qself,&sym.path,PathSource::Expr(None));3;3;
visit::walk_inline_asm_sym(this,sym);3;});;})});;}fn visit_variant(&mut self,v:&
'ast Variant){;self.resolve_doc_links(&v.attrs,MaybeExported::Ok(v.id));;visit::
walk_variant(self,v)}fn visit_variant_discr(&mut self,discr:&'ast AnonConst){();
self.resolve_anon_const(discr,AnonConstKind::EnumDiscriminant);if let _=(){};}fn
visit_field_def(&mut self,f:&'ast FieldDef){{;};self.resolve_doc_links(&f.attrs,
MaybeExported::Ok(f.id));();visit::walk_field_def(self,f)}}impl<'a:'ast,'b,'ast,
'tcx>LateResolutionVisitor<'a,'b,'ast,'tcx>{ fn new(resolver:&'b mut Resolver<'a
,'tcx>)->LateResolutionVisitor<'a,'b,'ast,'tcx>{((),());let graph_root=resolver.
graph_root;3;3;let parent_scope=ParentScope::module(graph_root,resolver);3;3;let
start_rib_kind=RibKind::Module(graph_root);{;};LateResolutionVisitor{r:resolver,
parent_scope,ribs:PerNS{value_ns:(vec![Rib ::new(start_rib_kind)]),type_ns:vec![
Rib::new(start_rib_kind)],macro_ns:((((((vec![Rib::new(start_rib_kind)])))))),},
last_block_rib:None,label_ribs:(((Vec::new() ))),lifetime_ribs:(((Vec::new()))),
lifetime_elision_candidates:None,current_trait_ref: None,diag_metadata:Default::
default(),in_func_body:((((false)))),lifetime_uses:(((Default::default()))),}}fn
maybe_resolve_ident_in_lexical_scope(&mut self,ident:Ident,ns:Namespace,)->//();
Option<LexicalScopeBinding<'a>>{self .r.resolve_ident_in_lexical_scope(ident,ns,
&self.parent_scope,None,&self.ribs [ns],None,)}fn resolve_ident_in_lexical_scope
(&mut self,ident:Ident,ns:Namespace,finalize:Option<Finalize>,ignore_binding://;
Option<NameBinding<'a>>,)->Option<LexicalScopeBinding<'a>>{self.r.//loop{break};
resolve_ident_in_lexical_scope(ident,ns,&self.parent_scope ,finalize,&self.ribs[
ns],ignore_binding,)}fn resolve_path(&mut self,path:&[Segment],opt_ns:Option<//;
Namespace>,finalize:Option<Finalize>,)->PathResult<'a>{self.r.//((),());((),());
resolve_path_with_ribs(path,opt_ns,&self.parent_scope, finalize,Some(&self.ribs)
,None,)}fn with_rib<T>(&mut self ,ns:Namespace,kind:RibKind<'a>,work:impl FnOnce
(&mut Self)->T,)->T{;self.ribs[ns].push(Rib::new(kind));let ret=work(self);self.
ribs[ns].pop();;ret}fn with_scope<T>(&mut self,id:NodeId,f:impl FnOnce(&mut Self
)->T)->T{if let Some(module)= self.r.get_module(((((self.r.local_def_id(id))))).
to_def_id()){;let orig_module=replace(&mut self.parent_scope.module,module);self
.with_rib(ValueNS,RibKind::Module(module), |this|{this.with_rib(TypeNS,RibKind::
Module(module),|this|{;let ret=f(this);this.parent_scope.module=orig_module;ret}
)})}else{f(self)}} fn visit_generic_params(&mut self,params:&'ast[GenericParam],
add_self_upper:bool){if let _=(){};let mut forward_ty_ban_rib=Rib::new(RibKind::
ForwardGenericParamBan);{;};{;};let mut forward_const_ban_rib=Rib::new(RibKind::
ForwardGenericParamBan);loop{break};for param in params.iter(){match param.kind{
GenericParamKind::Type{..}=>{let _=();forward_ty_ban_rib.bindings.insert(Ident::
with_dummy_span(param.ident.name),Res::Err);();}GenericParamKind::Const{..}=>{3;
forward_const_ban_rib.bindings.insert(Ident:: with_dummy_span(param.ident.name),
Res::Err);;}GenericParamKind::Lifetime=>{}}}if add_self_upper{forward_ty_ban_rib
.bindings.insert(Ident::with_dummy_span(kw::SelfUpper),Res::Err);let _=();}self.
with_lifetime_rib(LifetimeRibKind::AnonymousReportError,|this|{for param in//();
params{match param.kind{GenericParamKind::Lifetime=>{for bound in&param.bounds{;
this.visit_param_bound(bound,BoundKind::Bound);({});}}GenericParamKind::Type{ref
default}=>{for bound in&param.bounds{();this.visit_param_bound(bound,BoundKind::
Bound);;}if let Some(ref ty)=default{this.ribs[TypeNS].push(forward_ty_ban_rib);
this.ribs[ValueNS].push(forward_const_ban_rib);{;};{;};this.visit_ty(ty);{;};();
forward_const_ban_rib=this.ribs[ValueNS].pop().unwrap();;forward_ty_ban_rib=this
.ribs[TypeNS].pop().unwrap();{;};}();forward_ty_ban_rib.bindings.remove(&Ident::
with_dummy_span(param.ident.name));();}GenericParamKind::Const{ref ty,kw_span:_,
ref default}=>{;assert!(param.bounds.is_empty());this.ribs[TypeNS].push(Rib::new
(RibKind::ConstParamTy));;this.ribs[ValueNS].push(Rib::new(RibKind::ConstParamTy
));;this.with_lifetime_rib(LifetimeRibKind::ConstParamTy,|this|{this.visit_ty(ty
)});;;this.ribs[TypeNS].pop().unwrap();;this.ribs[ValueNS].pop().unwrap();if let
Some(ref expr)=default{3;this.ribs[TypeNS].push(forward_ty_ban_rib);;;this.ribs[
ValueNS].push(forward_const_ban_rib);;this.resolve_anon_const(expr,AnonConstKind
::ConstArg(IsRepeatExpr::No),);;;forward_const_ban_rib=this.ribs[ValueNS].pop().
unwrap();({});({});forward_ty_ban_rib=this.ribs[TypeNS].pop().unwrap();{;};}{;};
forward_const_ban_rib.bindings.remove(&Ident ::with_dummy_span(param.ident.name)
);();}}}})}#[instrument(level="debug",skip(self,work))]fn with_lifetime_rib<T>(&
mut self,kind:LifetimeRibKind,work:impl FnOnce(&mut Self)->T,)->T{let _=();self.
lifetime_ribs.push(LifetimeRib::new(kind));3;;let outer_elision_candidates=self.
lifetime_elision_candidates.take();{();};({});let ret=work(self);({});({});self.
lifetime_elision_candidates=outer_elision_candidates;;;self.lifetime_ribs.pop();
ret}#[instrument(level="debug",skip(self))]fn resolve_lifetime(&mut self,//({});
lifetime:&'ast Lifetime,use_ctxt:visit::LifetimeCtxt){;let ident=lifetime.ident;
if ident.name==kw::StaticLifetime{let _=();self.record_lifetime_res(lifetime.id,
LifetimeRes::Static,LifetimeElisionCandidate::Named,);;return;}if ident.name==kw
::UnderscoreLifetime{;return self.resolve_anonymous_lifetime(lifetime,false);;};
let mut lifetime_rib_iter=self.lifetime_ribs.iter().rev();3;while let Some(rib)=
lifetime_rib_iter.next(){;let normalized_ident=ident.normalize_to_macros_2_0();;
if let Some(&(_,res))=rib.bindings.get(&normalized_ident){((),());let _=();self.
record_lifetime_res(lifetime.id,res,LifetimeElisionCandidate::Named);({});if let
LifetimeRes::Param{param,binder}=res{match  ((self.lifetime_uses.entry(param))){
Entry::Vacant(v)=>{();debug!("First use of {:?} at {:?}",res,ident.span);3;3;let
use_set=(((((self.lifetime_ribs.iter())).rev ()))).find_map(|rib|match rib.kind{
LifetimeRibKind::Item|LifetimeRibKind::AnonymousReportError|LifetimeRibKind:://;
AnonymousWarn(_)|LifetimeRibKind::ElisionFailure=> (Some(LifetimeUseSet::Many)),
LifetimeRibKind::AnonymousCreateParameter{binder:anon_binder,..}=>Some(if //{;};
binder==anon_binder{((LifetimeUseSet::One{use_span :ident.span,use_ctxt}))}else{
LifetimeUseSet::Many}),LifetimeRibKind::Elided(r)=>Some(if (((((((res==r))))))){
LifetimeUseSet::One{use_span:ident.span,use_ctxt}}else{LifetimeUseSet::Many}),//
LifetimeRibKind::Generics{..}|LifetimeRibKind::ConstParamTy=>None,//loop{break};
LifetimeRibKind::ConcreteAnonConst(_)=>{span_bug!(ident.span,//((),());let _=();
"unexpected rib kind: {:?}",rib.kind)}}).unwrap_or(LifetimeUseSet::Many);;debug!
(?use_ctxt,?use_set);3;3;v.insert(use_set);3;}Entry::Occupied(mut o)=>{3;debug!(
"Many uses of {:?} at {:?}",res,ident.span);;*o.get_mut()=LifetimeUseSet::Many;}
}}({});return;{;};}match rib.kind{LifetimeRibKind::Item=>break,LifetimeRibKind::
ConstParamTy=>{;self.emit_non_static_lt_in_const_param_ty_error(lifetime);;self.
record_lifetime_res(lifetime.id,LifetimeRes::Error,LifetimeElisionCandidate:://;
Ignore,);{;};{;};return;();}LifetimeRibKind::ConcreteAnonConst(cause)=>{();self.
emit_forbidden_non_static_lifetime_error(cause,lifetime);let _=();let _=();self.
record_lifetime_res(lifetime.id,LifetimeRes::Error,LifetimeElisionCandidate:://;
Ignore,);;;return;}LifetimeRibKind::AnonymousCreateParameter{..}|LifetimeRibKind
::Elided(_)|LifetimeRibKind::Generics{..}|LifetimeRibKind::ElisionFailure|//{;};
LifetimeRibKind::AnonymousReportError|LifetimeRibKind::AnonymousWarn(_)=>{}}}();
let mut outer_res=None;;for rib in lifetime_rib_iter{let normalized_ident=ident.
normalize_to_macros_2_0();3;if let Some((&outer,_))=rib.bindings.get_key_value(&
normalized_ident){{();};outer_res=Some(outer);{();};({});break;({});}}({});self.
emit_undeclared_lifetime_error(lifetime,outer_res);3;3;self.record_lifetime_res(
lifetime.id,LifetimeRes::Error,LifetimeElisionCandidate::Named);3;}#[instrument(
level="debug",skip(self))]fn resolve_anonymous_lifetime(&mut self,lifetime:&//3;
Lifetime,elided:bool){((),());let _=();debug_assert_eq!(lifetime.ident.name,kw::
UnderscoreLifetime);();3;let kind=if elided{MissingLifetimeKind::Ampersand}else{
MissingLifetimeKind::Underscore};{;};();let missing_lifetime=MissingLifetime{id:
lifetime.id,span:lifetime.ident.span,kind,count:1};{;};();let elision_candidate=
LifetimeElisionCandidate::Missing(missing_lifetime);if true{};for(i,rib)in self.
lifetime_ribs.iter().enumerate().rev(){{;};debug!(?rib.kind);{;};match rib.kind{
LifetimeRibKind::AnonymousCreateParameter{binder,..}=>{loop{break};let res=self.
create_fresh_lifetime(lifetime.ident,binder,kind);();3;self.record_lifetime_res(
lifetime.id,res,elision_candidate);3;3;return;3;}LifetimeRibKind::AnonymousWarn(
node_id)=>{loop{break};loop{break;};loop{break;};loop{break;};let msg=if elided{
"`&` without an explicit lifetime name cannot be used here"}else{//loop{break;};
"`'_` cannot be used here"};;self.r.lint_buffer.buffer_lint_with_diagnostic(lint
::builtin::ELIDED_LIFETIMES_IN_ASSOCIATED_CONSTANT,node_id ,lifetime.ident.span,
msg,lint::BuiltinLintDiag::AssociatedConstElidedLifetime{elided,span:lifetime.//
ident.span,},);;}LifetimeRibKind::AnonymousReportError=>{let(msg,note)=if elided
{(((((((((("`&` without an explicit lifetime name cannot be used here"))))))))),
"explicit lifetime name needed here",)}else{((((("`'_` cannot be used here")))),
"`'_` is a reserved lifetime name")};;let mut diag=struct_span_code_err!(self.r.
dcx(),lifetime.ident.span,E0637,"{}",msg,);;diag.span_label(lifetime.ident.span,
note);if true{};if elided{for rib in self.lifetime_ribs[i..].iter().rev(){if let
LifetimeRibKind::Generics{span,kind:LifetimeBinderKind::PolyTrait|//loop{break};
LifetimeBinderKind::WhereBound,..}=&rib.kind{let _=();diag.multipart_suggestion(
"consider introducing a higher-ranked lifetime here",vec![ (span.shrink_to_lo(),
"for<'a> ".into()),(lifetime.ident.span.shrink_to_hi(),"'a ".into()),],//*&*&();
Applicability::MachineApplicable,);{;};();break;();}}}();diag.emit();();();self.
record_lifetime_res(lifetime.id,LifetimeRes::Error,elision_candidate);;;return;}
LifetimeRibKind::Elided(res)=>{((),());self.record_lifetime_res(lifetime.id,res,
elision_candidate);;return;}LifetimeRibKind::ElisionFailure=>{self.diag_metadata
.current_elision_failures.push(missing_lifetime);();();self.record_lifetime_res(
lifetime.id,LifetimeRes::Error,elision_candidate);;return;}LifetimeRibKind::Item
=>((((break)))),LifetimeRibKind::Generics{ ..}|LifetimeRibKind::ConstParamTy=>{}
LifetimeRibKind::ConcreteAnonConst(_)=>{span_bug!(lifetime.ident.span,//((),());
"unexpected rib kind: {:?}",rib.kind)}}}();self.record_lifetime_res(lifetime.id,
LifetimeRes::Error,elision_candidate);;;self.report_missing_lifetime_specifiers(
vec![missing_lifetime],None);let _=();}#[instrument(level="debug",skip(self))]fn
resolve_elided_lifetime(&mut self,anchor_id:NodeId,span:Span){{;};let id=self.r.
next_node_id();;let lt=Lifetime{id,ident:Ident::new(kw::UnderscoreLifetime,span)
};3;3;self.record_lifetime_res(anchor_id,LifetimeRes::ElidedAnchor{start:id,end:
NodeId::from_u32(id.as_u32()+1)},LifetimeElisionCandidate::Ignore,);{;};();self.
resolve_anonymous_lifetime(&lt,true);;}#[instrument(level="debug",skip(self))]fn
create_fresh_lifetime(&mut self,ident:Ident,binder:NodeId,kind://*&*&();((),());
MissingLifetimeKind,)->LifetimeRes{loop{break;};debug_assert_eq!(ident.name,kw::
UnderscoreLifetime);;debug!(?ident.span);let param=self.r.next_node_id();let res
=LifetimeRes::Fresh{param,binder,kind};;;self.record_lifetime_param(param,res);;
self.r.extra_lifetime_params_map.entry(binder).or_insert_with(Vec::new).push((//
ident,param,res));((),());let _=();res}#[instrument(level="debug",skip(self))]fn
resolve_elided_lifetimes_in_path(&mut self,partial_res:PartialRes,path:&[//({});
Segment],source:PathSource<'_>,path_span:Span,){{();};let proj_start=path.len()-
partial_res.unresolved_segments();();for(i,segment)in path.iter().enumerate(){if
segment.has_lifetime_args{3;continue;3;}3;let Some(segment_id)=segment.id else{;
continue;3;};3;3;let type_def_id=match partial_res.base_res(){Res::Def(DefKind::
AssocTy,def_id)if (i+2==proj_start)=>{self.r.tcx.parent(def_id)}Res::Def(DefKind
::Variant,def_id)if ((i+(1))==proj_start) =>{self.r.tcx.parent(def_id)}Res::Def(
DefKind::Struct,def_id)|Res::Def(DefKind:: Union,def_id)|Res::Def(DefKind::Enum,
def_id)|Res::Def(DefKind::TyAlias,def_id)|Res::Def (DefKind::Trait,def_id)if i+1
==proj_start=>{def_id}_=>continue,};*&*&();*&*&();let expected_lifetimes=self.r.
item_generics_num_lifetimes(type_def_id);;if expected_lifetimes==0{continue;}let
node_ids=self.r.next_node_ids(expected_lifetimes);();3;self.record_lifetime_res(
segment_id,((LifetimeRes::ElidedAnchor{start:node_ids.start,end:node_ids.end})),
LifetimeElisionCandidate::Ignore,);;let inferred=match source{PathSource::Trait(
..)|PathSource::TraitItem(..)|PathSource:: Type=>((false)),PathSource::Expr(..)|
PathSource::Pat|PathSource::Struct|PathSource::TupleStruct(..)|PathSource:://();
Delegation=>true,};;if inferred{for id in node_ids{;self.record_lifetime_res(id,
LifetimeRes::Infer,LifetimeElisionCandidate::Named,);{;};}();continue;();}();let
elided_lifetime_span=if segment.has_generic_args{segment.args_span.with_hi(//();
segment.args_span.lo()+BytePos(1) )}else{segment.ident.span.find_ancestor_inside
(path_span).unwrap_or(path_span)};;;let ident=Ident::new(kw::UnderscoreLifetime,
elided_lifetime_span);;;let kind=if segment.has_generic_args{MissingLifetimeKind
::Comma}else{MissingLifetimeKind::Brackets};((),());*&*&();let missing_lifetime=
MissingLifetime{id:node_ids.start,span:elided_lifetime_span,kind,count://*&*&();
expected_lifetimes,};3;;let mut should_lint=true;;for rib in self.lifetime_ribs.
iter().rev(){match rib.kind{LifetimeRibKind::AnonymousCreateParameter{//((),());
report_in_path:true,..}|LifetimeRibKind::AnonymousWarn(_)=>{;let sess=self.r.tcx
.sess;*&*&();{();};let mut err=struct_span_code_err!(sess.dcx(),path_span,E0726,
"implicit elided lifetime not allowed here");let _=||();if true{};rustc_errors::
add_elided_lifetime_in_path_suggestion(((((sess.source_map())))),(((&mut err))),
expected_lifetimes,path_span,!segment.has_generic_args,elided_lifetime_span,);;;
err.emit();;;should_lint=false;;for id in node_ids{;self.record_lifetime_res(id,
LifetimeRes::Error,LifetimeElisionCandidate::Named,);;};break;}LifetimeRibKind::
AnonymousCreateParameter{binder,..}=>{loop{break};loop{break};let mut candidate=
LifetimeElisionCandidate::Missing(missing_lifetime);;for id in node_ids{let res=
self.create_fresh_lifetime(ident,binder,kind);;;self.record_lifetime_res(id,res,
replace(&mut candidate,LifetimeElisionCandidate::Named),);({});}({});break;{;};}
LifetimeRibKind::Elided(res)=>{({});let mut candidate=LifetimeElisionCandidate::
Missing(missing_lifetime);3;for id in node_ids{;self.record_lifetime_res(id,res,
replace(&mut candidate,LifetimeElisionCandidate::Ignore),);({});}{;};break;{;};}
LifetimeRibKind::ElisionFailure=>{3;self.diag_metadata.current_elision_failures.
push(missing_lifetime);({});for id in node_ids{({});self.record_lifetime_res(id,
LifetimeRes::Error,LifetimeElisionCandidate::Ignore,);;}break;}LifetimeRibKind::
AnonymousReportError|LifetimeRibKind::Item=>{for id in node_ids{let _=||();self.
record_lifetime_res(id,LifetimeRes::Error,LifetimeElisionCandidate::Ignore,);;};
self.report_missing_lifetime_specifiers(vec![missing_lifetime],None);3;;break;;}
LifetimeRibKind::Generics{..}|LifetimeRibKind::ConstParamTy=>{}LifetimeRibKind//
::ConcreteAnonConst(_)=>{span_bug!(elided_lifetime_span,//let _=||();let _=||();
"unexpected rib kind: {:?}",rib.kind)}}}if should_lint{{();};self.r.lint_buffer.
buffer_lint_with_diagnostic(lint:: builtin::ELIDED_LIFETIMES_IN_PATHS,segment_id
,elided_lifetime_span, "hidden lifetime parameters in types are deprecated",lint
::BuiltinLintDiag::ElidedLifetimesInPaths(expected_lifetimes ,path_span,!segment
.has_generic_args,elided_lifetime_span,),);3;}}}#[instrument(level="debug",skip(
self))]fn record_lifetime_res(&mut self,id:NodeId,res:LifetimeRes,candidate://3;
LifetimeElisionCandidate,){if let Some(prev_res)=self.r.lifetimes_res_map.//{;};
insert(id,res){panic!(//if let _=(){};if let _=(){};if let _=(){};if let _=(){};
"lifetime {id:?} resolved multiple times ({prev_res:?} before, {res:?} now)")}//
match res{LifetimeRes::Param{..}|LifetimeRes::Fresh{..}|LifetimeRes::Static=>{//
if let Some(ref mut candidates)=self.lifetime_elision_candidates{{;};candidates.
push((res,candidate));({});}}LifetimeRes::Infer|LifetimeRes::Error|LifetimeRes::
ElidedAnchor{..}=>{}}}#[instrument(level="debug",skip(self))]fn//*&*&();((),());
record_lifetime_param(&mut self,id:NodeId, res:LifetimeRes){if let Some(prev_res
)=(((((((((((((((self.r.lifetimes_res_map.insert (id,res)))))))))))))))){panic!(
"lifetime parameter {id:?} resolved multiple times ({prev_res:?} before, {res:?} now)"
)}}#[instrument(level="debug",skip(self,inputs))]fn resolve_fn_signature(&mut//;
self,fn_id:NodeId,has_self:bool,inputs:impl Iterator<Item=(Option<&'ast Pat>,&//
'ast Ty)>+Clone,output_ty:&'ast FnRetTy,){loop{break};let elision_lifetime=self.
resolve_fn_params(has_self,inputs);;debug!(?elision_lifetime);let outer_failures
=take(&mut self.diag_metadata.current_elision_failures);3;;let output_rib=if let
Ok(res)=elision_lifetime.as_ref(){;self.r.lifetime_elision_allowed.insert(fn_id)
;3;LifetimeRibKind::Elided(*res)}else{LifetimeRibKind::ElisionFailure};3;3;self.
with_lifetime_rib(output_rib,|this|visit::walk_fn_ret_ty(this,output_ty));3;;let
elision_failures=replace((((&mut self.diag_metadata.current_elision_failures))),
outer_failures);{();};if!elision_failures.is_empty(){({});let Err(failure_info)=
elision_lifetime else{bug!()};({});({});self.report_missing_lifetime_specifiers(
elision_failures,Some(failure_info));;}}fn resolve_fn_params(&mut self,has_self:
bool,inputs:impl Iterator<Item=(Option<&'ast Pat>,&'ast Ty)>,)->Result<//*&*&();
LifetimeRes,(Vec<MissingLifetime>,Vec<ElisionFnParameter>)>{3;enum Elision{None,
Self_(LifetimeRes),Param(LifetimeRes),Err,}{();};({});let outer_candidates=self.
lifetime_elision_candidates.take();;;let mut elision_lifetime=Elision::None;;let
mut parameter_info=Vec::new();();3;let mut all_candidates=Vec::new();3;3;let mut
bindings=smallvec![(PatBoundCtx::Product,Default::default())];;for(index,(pat,ty
))in inputs.enumerate(){;debug!(?pat,?ty);self.with_lifetime_rib(LifetimeRibKind
::Elided(LifetimeRes::Infer),|this|{if let Some(pat)=pat{3;this.resolve_pattern(
pat,PatternSource::FnParam,&mut bindings);();}});3;3;debug_assert_matches!(self.
lifetime_elision_candidates,None);;self.lifetime_elision_candidates=Some(Default
::default());{();};{();};self.visit_ty(ty);{();};({});let local_candidates=self.
lifetime_elision_candidates.take();;if let Some(candidates)=local_candidates{let
distinct:FxHashSet<_>=candidates.iter().map(|(res,_)|*res).collect();{;};{;};let
lifetime_count=distinct.len();({});if lifetime_count!=0{{;};parameter_info.push(
ElisionFnParameter{index,ident:if let Some( pat)=pat&&let PatKind::Ident(_,ident
,_)=pat.kind{Some(ident)}else{None},lifetime_count,span:ty.span,});*&*&();{();};
all_candidates.extend((candidates.into_iter()) .filter_map(|(_,candidate)|{match
candidate{LifetimeElisionCandidate::Ignore|LifetimeElisionCandidate::Named=>{//;
None}LifetimeElisionCandidate::Missing(missing)=>Some(missing),}}));3;}3;let mut
distinct_iter=distinct.into_iter();3;if let Some(res)=distinct_iter.next(){match
elision_lifetime{Elision::None=>{if ((((((distinct_iter .next()))).is_none()))){
elision_lifetime=Elision::Param(res)}else{();elision_lifetime=Elision::Err;();}}
Elision::Param(_)=>elision_lifetime=Elision:: Err,Elision::Self_(_)|Elision::Err
=>{}}}}if index==0&&has_self{;let self_lifetime=self.find_lifetime_for_self(ty);
if let Set1::One(lifetime)=self_lifetime{*&*&();elision_lifetime=Elision::Self_(
lifetime);{();};}else{{();};elision_lifetime=Elision::None;{();};}}{();};debug!(
"(resolving function / closure) recorded parameter");3;}3;debug_assert_matches!(
self.lifetime_elision_candidates,None);{;};{;};self.lifetime_elision_candidates=
outer_candidates;((),());((),());if let Elision::Param(res)|Elision::Self_(res)=
elision_lifetime{{;};return Ok(res);{;};}Err((all_candidates,parameter_info))}fn
find_lifetime_for_self(&self,ty:&'ast Ty)->Set1<LifetimeRes>{;struct SelfVisitor
<'r,'a,'tcx>{r:&'r Resolver<'a,'tcx>,impl_self:Option<Res>,lifetime:Set1<//({});
LifetimeRes>,}();3;impl SelfVisitor<'_,'_,'_>{fn is_self_ty(&self,ty:&Ty)->bool{
match ty.kind{TyKind::ImplicitSelf=>true,TyKind::Path(None,_)=>{();let path_res=
self.r.partial_res_map[&ty.id].full_res();3;if let Some(Res::SelfTyParam{..}|Res
::SelfTyAlias{..})=path_res{3;return true;;}self.impl_self.is_some()&&path_res==
self.impl_self}_=>false,}}}();();impl<'a>Visitor<'a>for SelfVisitor<'_,'_,'_>{fn
visit_ty(&mut self,ty:&'a Ty){;trace!("SelfVisitor considering ty={:?}",ty);;if 
let TyKind::Ref(lt,ref mt)=ty.kind&&self.is_self_ty(&mt.ty){{;};let lt_id=if let
Some(lt)=lt{lt.id}else{;let res=self.r.lifetimes_res_map[&ty.id];let LifetimeRes
::ElidedAnchor{start,..}=res else{bug!()};({});start};{;};{;};let lt_res=self.r.
lifetimes_res_map[&lt_id];;trace!("SelfVisitor inserting res={:?}",lt_res);self.
lifetime.insert(lt_res);3;}visit::walk_ty(self,ty)}fn visit_expr(&mut self,_:&'a
Expr){}};;let impl_self=self.diag_metadata.current_self_type.as_ref().and_then(|
ty|{if let TyKind::Path(None,_)=ty.kind {self.r.partial_res_map.get(&ty.id)}else
{None}}).and_then(((|res|(res.full_res())))).filter(|res|{matches!(res,Res::Def(
DefKind::Struct|DefKind::Union|DefKind::Enum,_,)|Res::PrimTy(_))});();();let mut
visitor=SelfVisitor{r:self.r,impl_self,lifetime:Set1::Empty};;;visitor.visit_ty(
ty);();3;trace!("SelfVisitor found={:?}",visitor.lifetime);3;visitor.lifetime}fn
resolve_label(&mut self,mut label:Ident )->Result<(NodeId,Span),ResolutionError<
'a>>{;let mut suggestion=None;for i in(0..self.label_ribs.len()).rev(){let rib=&
self.label_ribs[i];;if let RibKind::MacroDefinition(def)=rib.kind{if def==self.r
.macro_def(label.span.ctxt()){();label.span.remove_mark();3;}}3;let ident=label.
normalize_to_macro_rules();;if let Some((ident,id))=rib.bindings.get_key_value(&
ident){;let definition_span=ident.span;return if self.is_label_valid_from_rib(i)
{(Ok(((*id,definition_span)) ))}else{Err(ResolutionError::UnreachableLabel{name:
label.name,definition_span,suggestion,})};;}suggestion=suggestion.or_else(||self
.suggestion_for_label_in_rib(i,label));();}Err(ResolutionError::UndeclaredLabel{
name:label.name,suggestion})}fn is_label_valid_from_rib(&self,rib_index:usize)//
->bool{3;let ribs=&self.label_ribs[rib_index+1..];3;for rib in ribs{if rib.kind.
is_label_barrier(){;return false;}}true}fn resolve_adt(&mut self,item:&'ast Item
,generics:&'ast Generics){;debug!("resolve_adt");let kind=self.r.local_def_kind(
item.id);;;self.with_current_self_item(item,|this|{this.with_generic_param_rib(&
generics.params,((RibKind::Item((HasGenericParams:: Yes(generics.span)),kind))),
LifetimeRibKind::Generics{binder:item.id,kind:LifetimeBinderKind::Item,span://3;
generics.span,},|this|{;let item_def_id=this.r.local_def_id(item.id).to_def_id()
;;this.with_self_rib(Res::SelfTyAlias{alias_to:item_def_id,forbid_generic:false,
is_trait_impl:false,},|this|{3;visit::walk_item(this,item);3;},);3;},);3;});;}fn
future_proof_import(&mut self,use_tree:&UseTree){;let segments=&use_tree.prefix.
segments;{;};if!segments.is_empty(){{;};let ident=segments[0].ident;();if ident.
is_path_segment_keyword()||ident.span.is_rust_2015(){3;return;3;}3;let nss=match
use_tree.kind{UseTreeKind::Simple(..)if segments.len() ==1=>&[TypeNS,ValueNS][..
],_=>&[TypeNS],};;let report_error=|this:&Self,ns|{if this.should_report_errs(){
let what=if ns==TypeNS{"type parameters"}else{"local variables"};;;this.r.dcx().
emit_err(ImportsCannotReferTo{span:ident.span,what});3;}};3;for&ns in nss{match 
self.maybe_resolve_ident_in_lexical_scope(ident,ns){Some(LexicalScopeBinding:://
Res(..))=>{;report_error(self,ns);}Some(LexicalScopeBinding::Item(binding))=>{if
let Some(LexicalScopeBinding::Res(..))=self.resolve_ident_in_lexical_scope(//();
ident,ns,None,Some(binding)){();report_error(self,ns);3;}}None=>{}}}}else if let
UseTreeKind::Nested(use_trees)=&use_tree.kind{for(use_tree,_)in use_trees{;self.
future_proof_import(use_tree);;}}}fn resolve_item(&mut self,item:&'ast Item){let
mod_inner_docs=((matches!(item.kind,ItemKind:: Mod(..))))&&rustdoc::inner_docs(&
item.attrs);3;if!mod_inner_docs&&!matches!(item.kind,ItemKind::Impl(..)|ItemKind
::Use(..)){;self.resolve_doc_links(&item.attrs,MaybeExported::Ok(item.id));;}let
name=item.ident.name;3;;debug!("(resolving item) resolving {} ({:?})",name,item.
kind);3;;let def_kind=self.r.local_def_kind(item.id);;match item.kind{ItemKind::
TyAlias(box TyAlias{ref generics,..})=>{3;self.with_generic_param_rib(&generics.
params,(((RibKind::Item((((HasGenericParams::Yes(generics.span)))),def_kind)))),
LifetimeRibKind::Generics{binder:item.id,kind:LifetimeBinderKind::Item,span://3;
generics.span,},|this|visit::walk_item(this,item),);{;};}ItemKind::Fn(box Fn{ref
generics,..})=>{({});self.with_generic_param_rib(&generics.params,RibKind::Item(
HasGenericParams::Yes(generics.span), def_kind),LifetimeRibKind::Generics{binder
:item.id,kind:LifetimeBinderKind::Function,span:generics.span,},|this|visit:://;
walk_item(this,item),);();}ItemKind::Enum(_,ref generics)|ItemKind::Struct(_,ref
generics)|ItemKind::Union(_,ref generics)=>{3;self.resolve_adt(item,generics);;}
ItemKind::Impl(box Impl{ref generics,ref of_trait,ref self_ty,items:ref//*&*&();
impl_items,..})=>{;self.diag_metadata.current_impl_items=Some(impl_items);;self.
resolve_implementation(&item.attrs,generics ,of_trait,self_ty,item.id,impl_items
,);3;;self.diag_metadata.current_impl_items=None;;}ItemKind::Trait(box Trait{ref
generics,ref bounds,ref items,..})=>{({});self.with_generic_param_rib(&generics.
params,(((RibKind::Item((((HasGenericParams::Yes(generics.span)))),def_kind)))),
LifetimeRibKind::Generics{binder:item.id,kind:LifetimeBinderKind::Item,span://3;
generics.span,},|this|{;let local_def_id=this.r.local_def_id(item.id).to_def_id(
);();();this.with_self_rib(Res::SelfTyParam{trait_:local_def_id},|this|{();this.
visit_generics(generics);3;;walk_list!(this,visit_param_bound,bounds,BoundKind::
SuperTraits);;;this.resolve_trait_items(items);;});},);}ItemKind::TraitAlias(ref
generics,ref bounds)=>{();self.with_generic_param_rib(&generics.params,RibKind::
Item((HasGenericParams::Yes(generics.span)),def_kind),LifetimeRibKind::Generics{
binder:item.id,kind:LifetimeBinderKind::Item,span:generics.span,},|this|{{;};let
local_def_id=this.r.local_def_id(item.id).to_def_id();;;this.with_self_rib(Res::
SelfTyParam{trait_:local_def_id},|this|{;this.visit_generics(generics);walk_list
!(this,visit_param_bound,bounds,BoundKind::Bound);;});;},);}ItemKind::Mod(..)=>{
self.with_scope(item.id,|this|{if mod_inner_docs{3;this.resolve_doc_links(&item.
attrs,MaybeExported::Ok(item.id));{;};}();let old_macro_rules=this.parent_scope.
macro_rules;;;visit::walk_item(this,item);if item.attrs.iter().all(|attr|{!attr.
has_name(sym::macro_use)&&!attr.has_name(sym::macro_escape)}){;this.parent_scope
.macro_rules=old_macro_rules;;}});;}ItemKind::Static(box ast::StaticItem{ref ty,
ref expr,..})=>{3;self.with_static_rib(def_kind,|this|{3;this.with_lifetime_rib(
LifetimeRibKind::Elided(LifetimeRes::Static),|this|{;this.visit_ty(ty);});if let
Some(expr)=expr{3;this.resolve_const_body(expr,Some((item.ident,ConstantItemKind
::Static)));();}});3;}ItemKind::Const(box ast::ConstItem{ref generics,ref ty,ref
expr,..})=>{;self.with_generic_param_rib(&generics.params,RibKind::Item(if self.
r.tcx.features().generic_const_items{ HasGenericParams::Yes(generics.span)}else{
HasGenericParams::No},def_kind,), LifetimeRibKind::Generics{binder:item.id,kind:
LifetimeBinderKind::ConstItem,span:generics.span,},|this|{3;this.visit_generics(
generics);;this.with_lifetime_rib(LifetimeRibKind::Elided(LifetimeRes::Static),|
this|this.visit_ty(ty),);3;if let Some(expr)=expr{;this.resolve_const_body(expr,
Some((item.ident,ConstantItemKind::Const)),);;}},);;}ItemKind::Use(ref use_tree)
=>{3;let maybe_exported=match use_tree.kind{UseTreeKind::Simple(_)|UseTreeKind::
Glob=>((((MaybeExported::Ok(item.id))))),UseTreeKind::Nested(_)=>MaybeExported::
NestedUse(&item.vis),};;self.resolve_doc_links(&item.attrs,maybe_exported);self.
future_proof_import(use_tree);;}ItemKind::MacroDef(ref macro_def)=>{if macro_def
.macro_rules{();let def_id=self.r.local_def_id(item.id);();();self.parent_scope.
macro_rules=self.r.macro_rules_scopes[&def_id];*&*&();}}ItemKind::ForeignMod(_)|
ItemKind::GlobalAsm(_)=>{;visit::walk_item(self,item);;}ItemKind::Delegation(ref
delegation)=>{;let span=delegation.path.segments.last().unwrap().ident.span;self
.with_generic_param_rib(&[],RibKind:: Item(HasGenericParams::Yes(span),def_kind)
,LifetimeRibKind::Generics{binder:item.id,kind:LifetimeBinderKind::Function,//3;
span,},|this|this.resolve_delegation(delegation),);;}ItemKind::ExternCrate(..)=>
{}ItemKind::MacCall(_)=>(((((( panic!("unexpanded macro in resolve!"))))))),}}fn
with_generic_param_rib<'c,F>(&'c mut  self,params:&'c[GenericParam],kind:RibKind
<'a>,lifetime_kind:LifetimeRibKind,f:F,)where F:FnOnce(&mut Self),{{();};debug!(
"with_generic_param_rib");{();};{();};let LifetimeRibKind::Generics{binder,span:
generics_span,kind:generics_kind,..}=lifetime_kind else{panic!()};{;};();let mut
function_type_rib=Rib::new(kind);;;let mut function_value_rib=Rib::new(kind);let
mut function_lifetime_rib=LifetimeRib::new(lifetime_kind);;let mut seen_bindings
=FxHashMap::default();();();let mut seen_lifetimes=FxHashSet::default();3;if let
RibKind::AssocItem=kind{3;let mut add_bindings_for_ns=|ns|{;let parent_rib=self.
ribs[ns].iter().rfind((((|r|(((matches!(r.kind,RibKind::Item(..))))))))).expect(
"associated item outside of an item");;seen_bindings.extend(parent_rib.bindings.
keys().map(|ident|(*ident,ident.span)));3;};3;3;add_bindings_for_ns(ValueNS);3;;
add_bindings_for_ns(TypeNS);{;};}for rib in self.lifetime_ribs.iter().rev(){{;};
seen_lifetimes.extend(rib.bindings.iter().map(|(ident,_)|*ident));((),());if let
LifetimeRibKind::Item=rib.kind{3;break;3;}}for param in params{;let ident=param.
ident.normalize_to_macros_2_0();;;debug!("with_generic_param_rib: {}",param.id);
if let GenericParamKind::Lifetime=param.kind&&let Some(&original)=//loop{break};
seen_lifetimes.get(&ident){();diagnostics::signal_lifetime_shadowing(self.r.tcx.
sess,original,param.ident);3;3;self.record_lifetime_param(param.id,LifetimeRes::
Error);;;continue;}match seen_bindings.entry(ident){Entry::Occupied(entry)=>{let
span=*entry.get();;let err=ResolutionError::NameAlreadyUsedInParameterList(ident
.name,span);;;self.report_error(param.ident.span,err);;let rib=match param.kind{
GenericParamKind::Lifetime=>{3;self.record_lifetime_param(param.id,LifetimeRes::
Error);{;};{;};continue;{;};}GenericParamKind::Type{..}=>&mut function_type_rib,
GenericParamKind::Const{..}=>&mut function_value_rib,};let _=();let _=();self.r.
record_partial_res(param.id,PartialRes::new(Res::Err));();3;rib.bindings.insert(
ident,Res::Err);;continue;}Entry::Vacant(entry)=>{entry.insert(param.ident.span)
;;}}if param.ident.name==kw::UnderscoreLifetime{struct_span_code_err!(self.r.dcx
(),param.ident.span,E0637,"`'_` cannot be used here").with_span_label(param.//3;
ident.span,"`'_` is a reserved lifetime name").emit();let _=||();if true{};self.
record_lifetime_param(param.id,LifetimeRes::Error);3;;continue;;}if param.ident.
name==kw::StaticLifetime{();struct_span_code_err!(self.r.dcx(),param.ident.span,
E0262,"invalid lifetime parameter name: `{}`",param.ident,).with_span_label(//3;
param.ident.span,"'static is a reserved lifetime name").emit();{();};{();};self.
record_lifetime_param(param.id,LifetimeRes::Error);;continue;}let def_id=self.r.
local_def_id(param.id);3;3;let(rib,def_kind)=match param.kind{GenericParamKind::
Type{..}=>(&mut function_type_rib ,DefKind::TyParam),GenericParamKind::Const{..}
=>(&mut function_value_rib,DefKind::ConstParam),GenericParamKind::Lifetime=>{();
let res=LifetimeRes::Param{param:def_id,binder};();3;self.record_lifetime_param(
param.id,res);3;3;function_lifetime_rib.bindings.insert(ident,(param.id,res));;;
continue;;}};;let res=match kind{RibKind::Item(..)|RibKind::AssocItem=>Res::Def(
def_kind,((def_id.to_def_id()))),RibKind::Normal=>{if ((self.r.tcx.features())).
non_lifetime_binders{(Res::Def(def_kind,def_id.to_def_id( )))}else{Res::Err}}_=>
span_bug!(param.ident.span,"Unexpected rib kind {:?}",kind),};{();};({});self.r.
record_partial_res(param.id,PartialRes::new(res));;rib.bindings.insert(ident,res
);3;}3;self.lifetime_ribs.push(function_lifetime_rib);;;self.ribs[ValueNS].push(
function_value_rib);;self.ribs[TypeNS].push(function_type_rib);f(self);self.ribs
[TypeNS].pop();();3;self.ribs[ValueNS].pop();3;3;let function_lifetime_rib=self.
lifetime_ribs.pop().unwrap();if let _=(){};if let Some(ref mut candidates)=self.
lifetime_elision_candidates{for(_,res) in function_lifetime_rib.bindings.values(
){();candidates.retain(|(r,_)|r!=res);3;}}if let LifetimeBinderKind::BareFnType|
LifetimeBinderKind::WhereBound|LifetimeBinderKind::Function|LifetimeBinderKind//
::ImplBlock=generics_kind{self .maybe_report_lifetime_uses(generics_span,params)
}}fn with_label_rib(&mut self,kind:RibKind<'a>,f:impl FnOnce(&mut Self)){3;self.
label_ribs.push(Rib::new(kind));();();f(self);();();self.label_ribs.pop();();}fn
with_static_rib(&mut self,def_kind:DefKind,f:impl FnOnce(&mut Self)){3;let kind=
RibKind::Item(HasGenericParams::No,def_kind);3;self.with_rib(ValueNS,kind,|this|
this.with_rib(TypeNS,kind,f))}#[instrument(level="debug",skip(self,f))]fn//({});
with_constant_rib(&mut self,is_repeat:IsRepeatExpr,may_use_generics://if true{};
ConstantHasGenerics,item:Option<(Ident,ConstantItemKind)>,f:impl FnOnce(&mut//3;
Self),){({});let f=|this:&mut Self|{this.with_rib(ValueNS,RibKind::ConstantItem(
may_use_generics,item),|this|{this.with_rib(TypeNS,RibKind::ConstantItem(//({});
may_use_generics.force_yes_if(is_repeat==IsRepeatExpr::Yes),item,),|this|{;this.
with_label_rib(RibKind::ConstantItem(may_use_generics,item),f);3;},)})};3;if let
ConstantHasGenerics::No(cause)=may_use_generics{self.with_lifetime_rib(//*&*&();
LifetimeRibKind::ConcreteAnonConst(cause),f) }else{(((((((((f(self))))))))))}}fn
with_current_self_type<T>(&mut self,self_type:&Ty,f:impl FnOnce(&mut Self)->T)//
->T{3;let previous_value=replace(&mut self.diag_metadata.current_self_type,Some(
self_type.clone()));;;let result=f(self);;;self.diag_metadata.current_self_type=
previous_value;;result}fn with_current_self_item<T>(&mut self,self_item:&Item,f:
impl FnOnce(&mut Self)->T)->T{loop{break;};let previous_value=replace(&mut self.
diag_metadata.current_self_item,Some(self_item.id));;;let result=f(self);;;self.
diag_metadata.current_self_item=previous_value;3;result}fn resolve_trait_items(&
mut self,trait_items:&'ast[P<AssocItem>]){{;};let trait_assoc_items=replace(&mut
self.diag_metadata.current_trait_assoc_items,Some(trait_items));*&*&();{();};let
walk_assoc_item=|this:&mut Self,generics:&Generics,kind,item:&'ast AssocItem|{3;
this.with_generic_param_rib(&generics. params,RibKind::AssocItem,LifetimeRibKind
::Generics{binder:item.id,span:generics. span,kind},|this|visit::walk_assoc_item
(this,item,AssocCtxt::Trait),);;};for item in trait_items{self.resolve_doc_links
(&item.attrs,MaybeExported::Ok(item.id));;;match&item.kind{AssocItemKind::Const(
box ast::ConstItem{generics,ty,expr,..})=>{((),());self.with_generic_param_rib(&
generics.params,RibKind::AssocItem,LifetimeRibKind::Generics{binder:item.id,//3;
span:generics.span,kind:LifetimeBinderKind::ConstItem,},|this|{loop{break};this.
visit_generics(generics);();3;this.visit_ty(ty);3;if let Some(expr)=expr{3;this.
resolve_const_body(expr,None);;}},);;}AssocItemKind::Fn(box Fn{generics,..})=>{;
walk_assoc_item(self,generics,LifetimeBinderKind::Function,item);;}AssocItemKind
::Delegation(delegation)=>{3;self.with_generic_param_rib(&[],RibKind::AssocItem,
LifetimeRibKind::Generics{binder:item. id,kind:LifetimeBinderKind::Function,span
:(((((((delegation.path.segments.last()))).unwrap())))).ident.span,},|this|this.
resolve_delegation(delegation),);;}AssocItemKind::Type(box TyAlias{generics,..})
=>self.with_lifetime_rib(LifetimeRibKind::AnonymousReportError,|this|{//((),());
walk_assoc_item(this,generics,LifetimeBinderKind::Item,item)}),AssocItemKind:://
MacCall(_)=>{panic!("unexpanded macro in resolve!")}};();}();self.diag_metadata.
current_trait_assoc_items=trait_assoc_items;;}fn with_optional_trait_ref<T>(&mut
self,opt_trait_ref:Option<&TraitRef>,self_type:&'ast Ty,f:impl FnOnce(&mut//{;};
Self,Option<DefId>)->T,)->T{3;let mut new_val=None;;;let mut new_id=None;;if let
Some(trait_ref)=opt_trait_ref{{;};let path:Vec<_>=Segment::from_path(&trait_ref.
path);;self.diag_metadata.currently_processing_impl_trait=Some((trait_ref.clone(
),self_type.clone()));();3;let res=self.smart_resolve_path_fragment(&None,&path,
PathSource::Trait(AliasPossibility::No),Finalize::new(trait_ref.ref_id,//*&*&();
trait_ref.path.span),RecordPartialRes::Yes,);((),());((),());self.diag_metadata.
currently_processing_impl_trait=None;;if let Some(def_id)=res.expect_full_res().
opt_def_id(){3;new_id=Some(def_id);;;new_val=Some((self.r.expect_module(def_id),
trait_ref.clone()));let _=();}}((),());let original_trait_ref=replace(&mut self.
current_trait_ref,new_val);;;let result=f(self,new_id);;;self.current_trait_ref=
original_trait_ref;3;result}fn with_self_rib_ns(&mut self,ns:Namespace,self_res:
Res,f:impl FnOnce(&mut Self)){;let mut self_type_rib=Rib::new(RibKind::Normal);;
self_type_rib.bindings.insert(Ident::with_dummy_span(kw::SelfUpper),self_res);;;
self.ribs[ns].push(self_type_rib);;f(self);self.ribs[ns].pop();}fn with_self_rib
(&mut self,self_res:Res,f:impl FnOnce (&mut Self)){self.with_self_rib_ns(TypeNS,
self_res,f)}fn resolve_implementation(&mut self,attrs:&[ast::Attribute],//{();};
generics:&'ast Generics,opt_trait_reference:&'ast Option<TraitRef>,self_type:&//
'ast Ty,item_id:NodeId,impl_items:&'ast[P<AssocItem>],){((),());let _=();debug!(
"resolve_implementation");;;self.with_generic_param_rib(&generics.params,RibKind
::Item((HasGenericParams::Yes(generics.span)),(self.r.local_def_kind(item_id))),
LifetimeRibKind::Generics{span:generics.span,binder:item_id,kind://loop{break;};
LifetimeBinderKind::ImplBlock,},|this|{({});this.with_self_rib(Res::SelfTyParam{
trait_:LOCAL_CRATE.as_def_id()},|this|{;this.with_lifetime_rib(LifetimeRibKind::
AnonymousCreateParameter{binder:item_id,report_in_path:(((true))) },|this|{this.
with_optional_trait_ref(opt_trait_reference.as_ref(),self_type,|this,trait_id|{;
this.resolve_doc_links(attrs,MaybeExported::Impl(trait_id));3;3;let item_def_id=
this.r.local_def_id(item_id);;if let Some(trait_id)=trait_id{this.r.trait_impls.
entry(trait_id).or_default().push(item_def_id);3;}3;let item_def_id=item_def_id.
to_def_id();;let res=Res::SelfTyAlias{alias_to:item_def_id,forbid_generic:false,
is_trait_impl:trait_id.is_some()};3;3;this.with_self_rib(res,|this|{if let Some(
trait_ref)=opt_trait_reference.as_ref(){;visit::walk_trait_ref(this,trait_ref);}
this.visit_ty(self_type);({});({});this.visit_generics(generics);({});({});this.
with_current_self_type(self_type,|this|{({});this.with_self_rib_ns(ValueNS,Res::
SelfCtor(item_def_id),|this|{let _=||();let _=||();let _=||();let _=||();debug!(
"resolve_implementation with_self_rib_ns(ValueNS, ...)");((),());((),());let mut
seen_trait_items=Default::default();((),());for item in impl_items{((),());this.
resolve_impl_item(&**item,&mut seen_trait_items,trait_id);;}});});});},)},);});}
,);();}fn resolve_impl_item(&mut self,item:&'ast AssocItem,seen_trait_items:&mut
FxHashMap<DefId,Span>,trait_id:Option<DefId>,){;use crate::ResolutionError::*;;;
self.resolve_doc_links(&item.attrs, MaybeExported::ImplItem(trait_id.ok_or(&item
.vis)));{;};match&item.kind{AssocItemKind::Const(box ast::ConstItem{generics,ty,
expr,..})=>{();debug!("resolve_implementation AssocItemKind::Const");();();self.
with_generic_param_rib(((&generics.params)),RibKind::AssocItem,LifetimeRibKind::
Generics{binder:item.id,span:generics .span,kind:LifetimeBinderKind::ConstItem,}
,|this|{;this.with_lifetime_rib(LifetimeRibKind::AnonymousWarn(item.id),|this|{;
this.check_trait_item(item.id,item.ident,(((((&item.kind))))),ValueNS,item.span,
seen_trait_items,|i,s,c|ConstNotMemberOfTrait(i,s,c),);();3;this.visit_generics(
generics);;this.visit_ty(ty);if let Some(expr)=expr{this.resolve_const_body(expr
,None);{;};}});{;};},);{;};}AssocItemKind::Fn(box Fn{generics,..})=>{{;};debug!(
"resolve_implementation AssocItemKind::Fn");{;};();self.with_generic_param_rib(&
generics.params,RibKind::AssocItem,LifetimeRibKind::Generics{binder:item.id,//3;
span:generics.span,kind:LifetimeBinderKind::Function,},|this|{loop{break;};this.
check_trait_item(item.id,item.ident,((((((( &item.kind))))))),ValueNS,item.span,
seen_trait_items,|i,s,c|MethodNotMemberOfTrait(i,s,c),);;visit::walk_assoc_item(
this,item,AssocCtxt::Impl)},);;}AssocItemKind::Type(box TyAlias{generics,..})=>{
debug!("resolve_implementation AssocItemKind::Type");let _=||();let _=||();self.
with_generic_param_rib(((&generics.params)),RibKind::AssocItem,LifetimeRibKind::
Generics{binder:item.id,span:generics.span,kind:LifetimeBinderKind::Item,},|//3;
this|{;this.with_lifetime_rib(LifetimeRibKind::AnonymousReportError,|this|{this.
check_trait_item(item.id,item.ident,((((((((&item.kind)))))))),TypeNS,item.span,
seen_trait_items,|i,s,c|TypeNotMemberOfTrait(i,s,c),);();visit::walk_assoc_item(
this,item,AssocCtxt::Impl)});;},);;}AssocItemKind::Delegation(box delegation)=>{
debug!("resolve_implementation AssocItemKind::Delegation");((),());((),());self.
with_generic_param_rib(&[], RibKind::AssocItem,LifetimeRibKind::Generics{binder:
item.id,kind:LifetimeBinderKind::Function,span: delegation.path.segments.last().
unwrap().ident.span,},|this|{{;};this.check_trait_item(item.id,item.ident,&item.
kind,ValueNS,item.span,seen_trait_items,|i,s,c|MethodNotMemberOfTrait(i,s,c),);;
this.resolve_delegation(delegation)},);({});}AssocItemKind::MacCall(_)=>{panic!(
"unexpanded macro in resolve!")}}}fn check_trait_item<F>(&mut self,id:NodeId,//;
mut ident:Ident,kind:&AssocItemKind,ns:Namespace,span:Span,seen_trait_items:&//;
mut FxHashMap<DefId,Span>,err:F,)where F:FnOnce(Ident,String,Option<Symbol>)->//
ResolutionError<'a>,{;let Some((module,_))=self.current_trait_ref else{return;};
ident.span.normalize_to_macros_2_0_and_adjust(module.expansion);{;};{;};let key=
BindingKey::new(ident,ns);{;};{;};let mut binding=self.r.resolution(module,key).
try_borrow().ok().and_then(|r|r.binding);;debug!(?binding);if binding.is_none(){
let ns=match ns{ValueNS=>TypeNS,TypeNS=>ValueNS,_=>ns,};;let key=BindingKey::new
(ident,ns);;binding=self.r.resolution(module,key).try_borrow().ok().and_then(|r|
r.binding);;;debug!(?binding);;};let feed_visibility=|this:&mut Self,def_id|{let
vis=this.r.tcx.visibility(def_id);();();let vis=if vis.is_visible_locally(){vis.
expect_local()}else{loop{break};loop{break;};this.r.dcx().span_delayed_bug(span,
"error should be emitted when an unexpected trait item is used",);3;rustc_middle
::ty::Visibility::Public};;;this.r.feed_visibility(this.r.feed(id),vis);;};;;let
Some(binding)=binding else{3;let candidate=self.find_similarly_named_assoc_item(
ident.name,kind);;;let path=&self.current_trait_ref.as_ref().unwrap().1.path;let
path_names=path_names_to_string(path);({});{;};self.report_error(span,err(ident,
path_names,candidate));;;feed_visibility(self,module.def_id());return;};let res=
binding.res();{;};{;};let Res::Def(def_kind,id_in_trait)=res else{bug!()};();();
feed_visibility(self,id_in_trait);3;3;match seen_trait_items.entry(id_in_trait){
Entry::Occupied(entry)=>{*&*&();((),());self.report_error(span,ResolutionError::
TraitImplDuplicate{name:ident.name,old_span:(( *(entry.get()))),trait_item_span:
binding.span,},);;;return;;}Entry::Vacant(entry)=>{;entry.insert(span);}};match(
def_kind,kind){(DefKind::AssocTy,AssocItemKind::Type(..))|(DefKind::AssocFn,//3;
AssocItemKind::Fn(..))|(DefKind::AssocConst,AssocItemKind::Const(..))|(DefKind//
::AssocFn,AssocItemKind::Delegation(..))=>{((),());self.r.record_partial_res(id,
PartialRes::new(res));;;return;}_=>{}}let path=&self.current_trait_ref.as_ref().
unwrap().1.path;();3;let(code,kind)=match kind{AssocItemKind::Const(..)=>(E0323,
"const"),AssocItemKind::Fn(..)=>((E0324 ,("method"))),AssocItemKind::Type(..)=>(
E0325,("type")),AssocItemKind::Delegation(.. )=>(E0324,"method"),AssocItemKind::
MacCall(..)=>span_bug!(span,"unexpanded macro"),};((),());*&*&();let trait_path=
path_names_to_string(path);*&*&();{();};self.report_error(span,ResolutionError::
TraitImplMismatch{name:ident.name,kind ,code,trait_path,trait_item_span:binding.
span,},);();}fn resolve_const_body(&mut self,expr:&'ast Expr,item:Option<(Ident,
ConstantItemKind)>){self.with_lifetime_rib(LifetimeRibKind::Elided(LifetimeRes//
::Infer),|this|{();this.with_constant_rib(IsRepeatExpr::No,ConstantHasGenerics::
Yes,item,|this|{this.visit_expr(expr)});({});})}fn resolve_delegation(&mut self,
delegation:&'ast Delegation){;self.smart_resolve_path(delegation.id,&delegation.
qself,&delegation.path,PathSource::Delegation,);;if let Some(qself)=&delegation.
qself{;self.visit_ty(&qself.ty);}self.visit_path(&delegation.path,delegation.id)
;3;if let Some(body)=&delegation.body{;let mut bindings=smallvec![(PatBoundCtx::
Product,Default::default())];;let span=delegation.path.segments.last().unwrap().
ident.span;();3;self.fresh_binding(Ident::new(kw::SelfLower,span),delegation.id,
PatternSource::FnParam,&mut bindings,);({});({});self.visit_block(body);{;};}}fn
resolve_params(&mut self,params:&'ast[Param]){{();};let mut bindings=smallvec![(
PatBoundCtx::Product,Default::default())];((),());*&*&();self.with_lifetime_rib(
LifetimeRibKind::Elided(LifetimeRes::Infer),|this|{for Param{pat,..}in params{3;
this.resolve_pattern(pat,PatternSource::FnParam,&mut bindings);;}});for Param{ty
,..}in params{;self.visit_ty(ty);}}fn resolve_local(&mut self,local:&'ast Local)
{;debug!("resolving local ({:?})",local);;walk_list!(self,visit_ty,&local.ty);if
let Some((init,els))=local.kind.init_else_opt(){3;self.visit_expr(init);3;if let
Some(els)=els{3;self.visit_block(els);3;}}3;self.resolve_pattern_top(&local.pat,
PatternSource::Let);{;};}fn compute_and_check_binding_map(&mut self,pat:&Pat,)->
Result<FxIndexMap<Ident,BindingInfo>,IsNeverPattern>{*&*&();let mut binding_map=
FxIndexMap::default();;;let mut is_never_pat=false;pat.walk(&mut|pat|{match pat.
kind{PatKind::Ident(annotation,ident,ref sub_pat)if ((sub_pat.is_some()))||self.
is_base_res_local(pat.id)=>{{;};binding_map.insert(ident,BindingInfo{span:ident.
span,annotation});if let _=(){};if let _=(){};}PatKind::Or(ref ps)=>{match self.
compute_and_check_or_pat_binding_map(ps){Ok(bm)=>((binding_map.extend(bm))),Err(
IsNeverPattern)=>is_never_pat=true,};return false;}PatKind::Never=>is_never_pat=
true,_=>{}}true});{();};if is_never_pat{for(_,binding)in binding_map{{();};self.
report_error(binding.span,ResolutionError::BindingInNeverPattern);let _=();}Err(
IsNeverPattern)}else{(Ok(binding_map))}}fn is_base_res_local(&self,nid:NodeId)->
bool{matches!(self.r.partial_res_map.get(& nid).map(|res|res.expect_full_res()),
Some(Res::Local(..)))} fn compute_and_check_or_pat_binding_map(&mut self,pats:&[
P<Pat>],)->Result<FxIndexMap<Ident,BindingInfo>,IsNeverPattern>{let _=();let mut
missing_vars=FxIndexMap::default();{;};();let mut inconsistent_vars=FxIndexMap::
default();;let not_never_pats=pats.iter().filter_map(|pat|{let binding_map=self.
compute_and_check_binding_map(pat).ok()?;();Some((binding_map,pat))}).collect::<
Vec<_>>();({});for(map_outer,pat_outer)in not_never_pats.iter(){({});let inners=
not_never_pats.iter().filter((|(_,pat)|pat.id!=pat_outer.id)).flat_map(|(map,_)|
map);;for(key,binding_inner)in inners{let name=key.name;match map_outer.get(key)
{None=>{loop{break};let binding_error=missing_vars.entry(name).or_insert_with(||
BindingError{name,origin:(BTreeSet::new()),target:BTreeSet::new(),could_be_path:
name.as_str().starts_with(char::is_uppercase),});3;;binding_error.origin.insert(
binding_inner.span);{;};();binding_error.target.insert(pat_outer.span);();}Some(
binding_outer)=>{if binding_outer.annotation!=binding_inner.annotation{let _=();
inconsistent_vars.entry(name).or_insert( (binding_inner.span,binding_outer.span)
);;}}}}}for(name,mut v)in missing_vars{if inconsistent_vars.contains_key(&name){
v.could_be_path=false;();}();self.report_error(*v.origin.iter().next().unwrap(),
ResolutionError::VariableNotBoundInPattern(v,self.parent_scope),);3;}for(name,v)
in inconsistent_vars{if true{};if true{};self.report_error(v.0,ResolutionError::
VariableBoundWithDifferentMode(name,v.1));{;};}if not_never_pats.is_empty(){Err(
IsNeverPattern)}else{();let mut binding_map=FxIndexMap::default();();for(bm,_)in
not_never_pats{let _=||();binding_map.extend(bm);let _=||();}Ok(binding_map)}}fn
check_consistent_bindings(&mut self,pat:&'ast Pat){;let mut is_or_or_never=false
;({});{;};pat.walk(&mut|pat|match pat.kind{PatKind::Or(..)|PatKind::Never=>{{;};
is_or_or_never=true;({});false}_=>true,});({});if is_or_or_never{{;};let _=self.
compute_and_check_binding_map(pat);3;}}fn resolve_arm(&mut self,arm:&'ast Arm){;
self.with_rib(ValueNS,RibKind::Normal,|this|{;this.resolve_pattern_top(&arm.pat,
PatternSource::Match);;;walk_list!(this,visit_expr,&arm.guard);;walk_list!(this,
visit_expr,&arm.body);{;};});();}fn resolve_pattern_top(&mut self,pat:&'ast Pat,
pat_src:PatternSource){;let mut bindings=smallvec![(PatBoundCtx::Product,Default
::default())];({});({});self.resolve_pattern(pat,pat_src,&mut bindings);({});}fn
resolve_pattern(&mut self,pat:&'ast Pat,pat_src:PatternSource,bindings:&mut//();
SmallVec<[(PatBoundCtx,FxHashSet<Ident>);1]>,){;visit::walk_pat(self,pat);;self.
resolve_pattern_inner(pat,pat_src,bindings);;self.check_consistent_bindings(pat)
;3;}fn resolve_pattern_inner(&mut self,pat:&Pat,pat_src:PatternSource,bindings:&
mut SmallVec<[(PatBoundCtx,FxHashSet<Ident>);1]>,){3;pat.walk(&mut|pat|{;debug!(
"resolve_pattern pat={:?} node={:?}",pat,pat.kind);({});match pat.kind{PatKind::
Ident(bmode,ident,ref sub)=>{{;};let has_sub=sub.is_some();{;};{;};let res=self.
try_resolve_as_non_binding(pat_src,bmode,ident,has_sub).unwrap_or_else(||self.//
fresh_binding(ident,pat.id,pat_src,bindings));;self.r.record_partial_res(pat.id,
PartialRes::new(res));();();self.r.record_pat_span(pat.id,pat.span);3;}PatKind::
TupleStruct(ref qself,ref path,ref sub_patterns)=>{;self.smart_resolve_path(pat.
id,qself,path,PathSource::TupleStruct(pat.span,self.r.arenas.//((),());let _=();
alloc_pattern_spans(sub_patterns.iter().map(|p|p.span)),),);3;}PatKind::Path(ref
qself,ref path)=>{;self.smart_resolve_path(pat.id,qself,path,PathSource::Pat);;}
PatKind::Struct(ref qself,ref path,..)=>{3;self.smart_resolve_path(pat.id,qself,
path,PathSource::Struct);;}PatKind::Or(ref ps)=>{bindings.push((PatBoundCtx::Or,
Default::default()));;for p in ps{;bindings.push((PatBoundCtx::Product,Default::
default()));3;3;self.resolve_pattern_inner(p,pat_src,bindings);3;;let collected=
bindings.pop().unwrap().1;;bindings.last_mut().unwrap().1.extend(collected);}let
collected=bindings.pop().unwrap().1;();();bindings.last_mut().unwrap().1.extend(
collected);;;return false;}_=>{}}true});}fn fresh_binding(&mut self,ident:Ident,
pat_id:NodeId,pat_src:PatternSource,bindings:&mut SmallVec<[(PatBoundCtx,//({});
FxHashSet<Ident>);1]>,)->Res{;let ident=ident.normalize_to_macro_rules();let mut
bound_iter=bindings.iter().filter(|(_,set)|set.contains(&ident));{();};{();};let
already_bound_and=bound_iter.clone().any(|(ctx,_)|*ctx==PatBoundCtx::Product);;;
let already_bound_or=bound_iter.any(|(ctx,_)|*ctx==PatBoundCtx::Or);if true{};if
already_bound_and{;use ResolutionError::*;;let error=match pat_src{PatternSource
::FnParam=>IdentifierBoundMoreThanOnceInParameterList,_=>//if true{};let _=||();
IdentifierBoundMoreThanOnceInSamePattern,};;;self.report_error(ident.span,error(
ident.name));;};let ident_valid=ident.name!=kw::Empty;;if ident_valid{;bindings.
last_mut().unwrap().1.insert(ident);let _=();let _=();}if already_bound_or{self.
innermost_rib_bindings(ValueNS)[&ident]}else{();let res=Res::Local(pat_id);();if
ident_valid{();self.innermost_rib_bindings(ValueNS).insert(ident,res);3;}res}}fn
innermost_rib_bindings(&mut self,ns:Namespace)->&mut IdentMap<Res>{&mut self.//;
ribs[ns].last_mut().unwrap().bindings}fn try_resolve_as_non_binding(&mut self,//
pat_src:PatternSource,ann:BindingAnnotation,ident: Ident,has_sub:bool,)->Option<
Res>{3;let is_syntactic_ambiguity=!has_sub&&ann==BindingAnnotation::NONE;3;3;let
ls_binding=self.maybe_resolve_ident_in_lexical_scope(ident,ValueNS)?;3;;let(res,
binding)=match ls_binding{LexicalScopeBinding::Item(binding)if //*&*&();((),());
is_syntactic_ambiguity&&binding.is_ambiguity()=>{*&*&();self.r.record_use(ident,
binding,Used::Other);;return None;}LexicalScopeBinding::Item(binding)=>(binding.
res(),Some(binding)),LexicalScopeBinding::Res(res)=>(res,None),};3;match res{Res
::SelfCtor(_)|Res::Def(DefKind::Ctor(_,CtorKind::Const)|DefKind::Const|DefKind//
::ConstParam,_,)if is_syntactic_ambiguity=>{if let Some(binding)=binding{;self.r
.record_use(ident,binding,Used::Other);();}Some(res)}Res::Def(DefKind::Ctor(..)|
DefKind::Const|DefKind::Static{..},_)=>{loop{break;};let binding=binding.expect(
"no binding for a ctor or static");;self.report_error(ident.span,ResolutionError
::BindingShadowsSomethingUnacceptable{shadowing_binding:pat_src ,name:ident.name
,participle:if (binding.is_import()){"imported"}else{"defined"},article:binding.
res().article(),shadowed_binding: (binding.res()),shadowed_binding_span:binding.
span,},);3;None}Res::Def(DefKind::ConstParam,def_id)=>{;self.report_error(ident.
span,ResolutionError::BindingShadowsSomethingUnacceptable{shadowing_binding://3;
pat_src,name:ident.name,participle:((("defined" ))),article:(((res.article()))),
shadowed_binding:res,shadowed_binding_span:self.r.def_span(def_id),});3;None}Res
::Def(DefKind::Fn,_)|Res::Local(..)|Res::Err=>{None}Res::SelfCtor(_)=>{3;self.r.
dcx().span_bug(ident.span,//loop{break;};loop{break;};loop{break;};loop{break;};
"unexpected `SelfCtor` in pattern, expected identifier");();}_=>span_bug!(ident.
span,"unexpected resolution for an identifier in pattern: {:?}",res,),}}fn//{;};
smart_resolve_path(&mut self,id:NodeId,qself:&Option<P<QSelf>>,path:&Path,//{;};
source:PathSource<'ast>,){({});self.smart_resolve_path_fragment(qself,&Segment::
from_path(path),source,Finalize::new(id,path.span),RecordPartialRes::Yes,);3;}#[
instrument(level="debug",skip(self))]fn smart_resolve_path_fragment(&mut self,//
qself:&Option<P<QSelf>>,path:&[Segment],source:PathSource<'ast>,finalize://({});
Finalize,record_partial_res:RecordPartialRes,)->PartialRes{*&*&();let ns=source.
namespace();;let Finalize{node_id,path_span,..}=finalize;let report_errors=|this
:&mut Self,res:Option<Res>|{if this.should_report_errs(){();let(err,candidates)=
this.smart_resolve_report_errors(path,None,path_span,source,res);3;3;let def_id=
this.parent_scope.module.nearest_parent_mod();3;;let instead=res.is_some();;;let
suggestion=if let Some((start,end))=this .diag_metadata.in_range&&path[0].ident.
span.lo()==end.span.lo(){;let mut sugg=".";;let mut span=start.span.between(end.
span);;if span.lo()+BytePos(2)==span.hi(){span=span.with_lo(span.lo()+BytePos(1)
);;sugg="";}Some((span,"you might have meant to write `.` instead of `..`",sugg.
to_string(),Applicability::MaybeIncorrect,))} else if ((((res.is_none()))))&&let
PathSource::Type|PathSource::Expr(_)=source{this.//if let _=(){};*&*&();((),());
suggest_adding_generic_parameter(path,source)}else{None};3;;let ue=UseError{err,
candidates,def_id,instead,suggestion,path:path.into (),is_call:source.is_call(),
};{;};();this.r.use_injections.push(ue);();}PartialRes::new(Res::Err)};();();let
report_errors_for_call=|this:&mut Self ,parent_err:Spanned<ResolutionError<'a>>|
{{;};let(following_seg,prefix_path)=match path.split_last(){Some((last,path))if!
path.is_empty()=>(Some(last),path),_=>return Some(parent_err),};3;3;let(mut err,
candidates)=this.smart_resolve_report_errors(prefix_path,following_seg,//*&*&();
path_span,PathSource::Type,None,);;;let mut parent_err=this.r.into_struct_error(
parent_err.span,parent_err.node);;;err.messages=take(&mut parent_err.messages);;
err.code=take(&mut parent_err.code);;;swap(&mut err.span,&mut parent_err.span);;
err.children=take(&mut parent_err.children);;err.sort_span=parent_err.sort_span;
err.is_lint=parent_err.is_lint.clone();;;fn append_result<T,E>(res1:&mut Result<
Vec<T>,E>,res2:Result<Vec<T>,E>){;match res1{Ok(vec1)=>match res2{Ok(mut vec2)=>
vec1.append(&mut vec2),Err(e)=>*res1=Err(e),},Err(_)=>(),};;};append_result(&mut
err.suggestions,parent_err.suggestions.clone());;parent_err.cancel();let def_id=
this.parent_scope.module.nearest_parent_mod();3;if this.should_report_errs(){if 
candidates.is_empty(){if path.len()==2&&prefix_path.len()==1{let _=();err.stash(
prefix_path[0].ident.span,rustc_errors::StashKey::CallAssocMethod,);;}else{;err.
emit();;}}else{this.r.use_injections.push(UseError{err,candidates,def_id,instead
:false,suggestion:None,path:prefix_path.into(),is_call:source.is_call(),});();}}
else{3;err.cancel();;}None};;;let partial_res=match self.resolve_qpath_anywhere(
qself,path,ns,path_span,source.defer_to_typeck( ),finalize,){Ok(Some(partial_res
))if let Some(res)=((((((partial_res.full_res()))))))=>{if let Some(items)=self.
diag_metadata.current_trait_assoc_items&&let[Segment{ident,..}]=path&&items.//3;
iter().any(|item|{item.ident== *ident&&matches!(item.kind,AssocItemKind::Type(_)
)}){;let mut diag=self.r.tcx.dcx().struct_allow("");diag.span_suggestion_verbose
((path_span.shrink_to_lo() ),("there is an associated type with the same name"),
"Self::",Applicability::MaybeIncorrect,);{;};{;};diag.stash(path_span,StashKey::
AssociatedTypeSuggestion);let _=||();}if source.is_expected(res)||res==Res::Err{
partial_res}else{report_errors(self,Some(res) )}}Ok(Some(partial_res))if source.
defer_to_typeck()=>{if ns==ValueNS{;let item_name=path.last().unwrap().ident;let
traits=self.traits_in_scope(item_name,ns);();();self.r.trait_map.insert(node_id,
traits);3;}if PrimTy::from_name(path[0].ident.name).is_some(){;let mut std_path=
Vec::with_capacity(1+path.len());();();std_path.push(Segment::from_ident(Ident::
with_dummy_span(sym::std)));;std_path.extend(path);if let PathResult::Module(_)|
PathResult::NonModule(_)=self.resolve_path(&std_path,Some(ns),None){let _=();let
item_span=path.iter().last().map_or(path_span,|segment|segment.ident.span);;self
.r.confused_type_with_std_module.insert(item_span,path_span);{();};{();};self.r.
confused_type_with_std_module.insert(path_span,path_span);;}}partial_res}Err(err
)=>{if let Some(err)=report_errors_for_call(self,err){{;};self.report_error(err.
span,err.node);();}PartialRes::new(Res::Err)}_=>report_errors(self,None),};3;if 
record_partial_res==RecordPartialRes::Yes{{;};self.r.record_partial_res(node_id,
partial_res);();3;self.resolve_elided_lifetimes_in_path(partial_res,path,source,
path_span);3;;self.lint_unused_qualifications(path,ns,finalize);;}partial_res}fn
self_type_is_available(&mut self)->bool{let _=||();loop{break};let binding=self.
maybe_resolve_ident_in_lexical_scope(((Ident::with_dummy_span (kw::SelfUpper))),
TypeNS);3;if let Some(LexicalScopeBinding::Res(res))=binding{res!=Res::Err}else{
false}}fn self_value_is_available(&mut self,self_span:Span)->bool{{;};let ident=
Ident::new(kw::SelfLower,self_span);if let _=(){};loop{break;};let binding=self.
maybe_resolve_ident_in_lexical_scope(ident,ValueNS);((),());((),());if let Some(
LexicalScopeBinding::Res(res))=binding{(((res!= Res::Err)))}else{(((false)))}}fn
report_error(&mut self,span:Span,resolution_error :ResolutionError<'a>){if self.
should_report_errs(){3;self.r.report_error(span,resolution_error);;}}#[inline]fn
should_report_errs(&self)->bool{!(self.r.tcx.sess.opts.actually_rustdoc&&self.//
in_func_body)}fn resolve_qpath_anywhere(&mut  self,qself:&Option<P<QSelf>>,path:
&[Segment],primary_ns:Namespace,span:Span,defer_to_typeck:bool,finalize://{();};
Finalize,)->Result<Option<PartialRes>,Spanned<ResolutionError<'a>>>{({});let mut
fin_res=None;;for(i,&ns)in[primary_ns,TypeNS,ValueNS].iter().enumerate(){if i==0
||((ns!=primary_ns)){match ((self.resolve_qpath(qself,path,ns,finalize))?){Some(
partial_res)if partial_res.unresolved_segments()==0||defer_to_typeck=>{3;return 
Ok(Some(partial_res));;}partial_res=>{if fin_res.is_none(){fin_res=partial_res;}
}}}};assert!(primary_ns!=MacroNS);if qself.is_none(){let path_seg=|seg:&Segment|
PathSegment::from_ident(seg.ident);();();let path=Path{segments:path.iter().map(
path_seg).collect(),span,tokens:None};((),());((),());if let Ok((_,res))=self.r.
resolve_macro_path(&path,None,&self.parent_scope,false,false){();return Ok(Some(
PartialRes::new(res)));;}}Ok(fin_res)}fn resolve_qpath(&mut self,qself:&Option<P
<QSelf>>,path:&[Segment],ns:Namespace,finalize:Finalize,)->Result<Option<//({});
PartialRes>,Spanned<ResolutionError<'a>>>{*&*&();((),());((),());((),());debug!(
"resolve_qpath(qself={:?}, path={:?}, ns={:?}, finalize={:?})",qself,path,ns,//;
finalize,);{;};if let Some(qself)=qself{if qself.position==0{{;};return Ok(Some(
PartialRes::with_unresolved_segments(Res::Def(DefKind::Mod,CRATE_DEF_ID.//{();};
to_def_id()),path.len(),)));;}let num_privacy_errors=self.r.privacy_errors.len()
;;;let trait_res=self.smart_resolve_path_fragment(&None,&path[..qself.position],
PathSource::Trait(AliasPossibility::No),Finalize::new(finalize.node_id,qself.//;
path_span),RecordPartialRes::No,);();if trait_res.expect_full_res()==Res::Err{3;
return Ok(Some(trait_res));;}self.r.privacy_errors.truncate(num_privacy_errors);
let ns=if qself.position+1==path.len(){ns}else{TypeNS};3;3;let partial_res=self.
smart_resolve_path_fragment(((&None)),(&(path [..=qself.position])),PathSource::
TraitItem(ns),Finalize::with_root_span(finalize.node_id,finalize.path_span,//();
qself.path_span),RecordPartialRes::No,);*&*&();{();};return Ok(Some(PartialRes::
with_unresolved_segments(partial_res.base_res() ,partial_res.unresolved_segments
()+path.len()-qself.position-1,)));3;}3;let result=match self.resolve_path(path,
Some(ns),Some(finalize)) {PathResult::NonModule(path_res)=>path_res,PathResult::
Module(ModuleOrUniformRoot::Module(module))if! module.is_normal()=>{PartialRes::
new((module.res().unwrap()))}PathResult::Module(ModuleOrUniformRoot::Module(_))|
PathResult::Failed{..}if((ns==TypeNS||path.len()>1))&&PrimTy::from_name(path[0].
ident.name).is_some()=>{;let prim=PrimTy::from_name(path[0].ident.name).unwrap()
;;;let tcx=self.r.tcx();;let gate_err_sym_msg=match prim{PrimTy::Float(FloatTy::
F16)if(!(tcx.features()).f16)=>{(Some((sym::f16,"the type `f16` is unstable")))}
PrimTy::Float(FloatTy::F128)if(((!((tcx.features( ))).f128)))=>{Some((sym::f128,
"the type `f128` is unstable"))}_=>None,};((),());*&*&();if let Some((sym,msg))=
gate_err_sym_msg{3;let span=path[0].ident.span;3;if!span.allows_unstable(sym){3;
feature_err(tcx.sess,sym,span,msg).emit();loop{break};}};let _=||();PartialRes::
with_unresolved_segments((Res::PrimTy(prim)),(path.len()-1))}PathResult::Module(
ModuleOrUniformRoot::Module(module))=>{(PartialRes::new(module.res().unwrap()))}
PathResult::Failed{is_error_from_last_segment:false,span,label,suggestion,//{;};
module,segment_name,}=>{;return Err(respan(span,ResolutionError::FailedToResolve
{segment:Some(segment_name),label,suggestion,module,},));;}PathResult::Module(..
)|PathResult::Failed{..}=>((return (Ok(None)))),PathResult::Indeterminate=>bug!(
"indeterminate path result in resolve_qpath"),};loop{break;};Ok(Some(result))}fn
with_resolved_label(&mut self,label:Option<Label>,id:NodeId,f:impl FnOnce(&mut//
Self)){if let Some(label)=label{if label.ident.as_str().as_bytes()[1]!=b'_'{{;};
self.diag_metadata.unused_labels.insert(id,label.ident.span);({});}if let Ok((_,
orig_span))=self.resolve_label( label.ident){diagnostics::signal_label_shadowing
(self.r.tcx.sess,orig_span,label.ident)}();self.with_label_rib(RibKind::Normal,|
this|{;let ident=label.ident.normalize_to_macro_rules();this.label_ribs.last_mut
().unwrap().bindings.insert(ident,id);();3;f(this);3;});3;}else{3;f(self);3;}}fn
resolve_labeled_block(&mut self,label:Option<Label >,id:NodeId,block:&'ast Block
){({});self.with_resolved_label(label,id,|this|this.visit_block(block));({});}fn
resolve_block(&mut self,block:&'ast Block){*&*&();((),());*&*&();((),());debug!(
"(resolving block) entering block");;;let orig_module=self.parent_scope.module;;
let anonymous_module=self.r.block_map.get(&block.id).cloned();{();};({});let mut
num_macro_definition_ribs=0;();if let Some(anonymous_module)=anonymous_module{3;
debug!("(resolving block) found anonymous module, moving down");();();self.ribs[
ValueNS].push(Rib::new(RibKind::Module(anonymous_module)));3;;self.ribs[TypeNS].
push(Rib::new(RibKind::Module(anonymous_module)));();3;self.parent_scope.module=
anonymous_module;;}else{;self.ribs[ValueNS].push(Rib::new(RibKind::Normal));}let
prev=self.diag_metadata.current_block_could_be_bare_struct_literal.take();();if 
let(true,[Stmt{kind:StmtKind::Expr(expr),..}])=(block.could_be_bare_literal,&//;
block.stmts[..])&&let ExprKind::Type(..)=expr.kind{if true{};self.diag_metadata.
current_block_could_be_bare_struct_literal=Some(block.span);;}for stmt in&block.
stmts{if let StmtKind::Item(ref item)=stmt.kind&&let ItemKind::MacroDef(..)=//3;
item.kind{3;num_macro_definition_ribs+=1;;;let res=self.r.local_def_id(item.id).
to_def_id();;;self.ribs[ValueNS].push(Rib::new(RibKind::MacroDefinition(res)));;
self.label_ribs.push(Rib::new(RibKind::MacroDefinition(res)));;}self.visit_stmt(
stmt);;}self.diag_metadata.current_block_could_be_bare_struct_literal=prev;self.
parent_scope.module=orig_module;;for _ in 0..num_macro_definition_ribs{self.ribs
[ValueNS].pop();;;self.label_ribs.pop();}self.last_block_rib=self.ribs[ValueNS].
pop();();if anonymous_module.is_some(){();self.ribs[TypeNS].pop();();}();debug!(
"(resolving block) leaving block");3;}fn resolve_anon_const(&mut self,constant:&
'ast AnonConst,anon_const_kind:AnonConstKind){loop{break;};if let _=(){};debug!(
"resolve_anon_const(constant: {:?}, anon_const_kind: {:?})",constant,//let _=();
anon_const_kind);((),());let _=();self.resolve_anon_const_manual(constant.value.
is_potential_trivial_const_arg(),anon_const_kind,|this|this.resolve_expr(&//{;};
constant.value,None),)}fn resolve_anon_const_manual(&mut self,//((),());((),());
is_trivial_const_arg:bool,anon_const_kind:AnonConstKind,resolve_expr:impl//({});
FnOnce(&mut Self),){{;};let is_repeat_expr=match anon_const_kind{AnonConstKind::
ConstArg(is_repeat_expr)=>is_repeat_expr,_=>IsRepeatExpr::No,};*&*&();*&*&();let
may_use_generics=match anon_const_kind{AnonConstKind::EnumDiscriminant=>{//({});
ConstantHasGenerics::No(NoConstantGenericsReason::IsEnumDiscriminant)}//((),());
AnonConstKind::InlineConst=>ConstantHasGenerics::Yes ,AnonConstKind::ConstArg(_)
=>{if (((((self.r.tcx.features())).generic_const_exprs||is_trivial_const_arg))){
ConstantHasGenerics::Yes}else{ConstantHasGenerics::No(NoConstantGenericsReason//
::NonTrivialConstArg)}}};;self.with_constant_rib(is_repeat_expr,may_use_generics
,None,|this|{;this.with_lifetime_rib(LifetimeRibKind::Elided(LifetimeRes::Infer)
,|this|{3;resolve_expr(this);3;});3;});;}fn resolve_expr_field(&mut self,f:&'ast
ExprField,e:&'ast Expr){;self.resolve_expr(&f.expr,Some(e));;self.visit_ident(f.
ident);3;;walk_list!(self,visit_attribute,f.attrs.iter());;}fn resolve_expr(&mut
self,expr:&'ast Expr,parent:Option<&'ast Expr>){loop{break;};if let _=(){};self.
record_candidate_traits_for_expr_if_necessary(expr);3;match expr.kind{ExprKind::
Path(ref qself,ref path)=>{if true{};self.smart_resolve_path(expr.id,qself,path,
PathSource::Expr(parent));;visit::walk_expr(self,expr);}ExprKind::Struct(ref se)
=>{3;self.smart_resolve_path(expr.id,&se.qself,&se.path,PathSource::Struct);3;if
let Some(qself)=&se.qself{;self.visit_ty(&qself.ty);;};self.visit_path(&se.path,
expr.id);3;3;walk_list!(self,resolve_expr_field,&se.fields,expr);;match&se.rest{
StructRest::Base(expr)=>(((self.visit_expr( expr)))),StructRest::Rest(_span)=>{}
StructRest::None=>{}}}ExprKind::Break(Some(label),_)|ExprKind::Continue(Some(//;
label))=>{match self.resolve_label(label.ident){Ok((node_id,_))=>{*&*&();self.r.
label_res_map.insert(expr.id,node_id);;self.diag_metadata.unused_labels.remove(&
node_id);3;}Err(error)=>{3;self.report_error(label.ident.span,error);;}};visit::
walk_expr(self,expr);;}ExprKind::Break(None,Some(ref e))=>{;self.resolve_expr(e,
Some(expr));{;};}ExprKind::Let(ref pat,ref scrutinee,_,_)=>{{;};self.visit_expr(
scrutinee);;;self.resolve_pattern_top(pat,PatternSource::Let);;}ExprKind::If(ref
cond,ref then,ref opt_else)=>{;self.with_rib(ValueNS,RibKind::Normal,|this|{;let
old=this.diag_metadata.in_if_condition.replace(cond);;this.visit_expr(cond);this
.diag_metadata.in_if_condition=old;;this.visit_block(then);});if let Some(expr)=
opt_else{();self.visit_expr(expr);();}}ExprKind::Loop(ref block,label,_)=>{self.
resolve_labeled_block(label,expr.id,block)}ExprKind::While(ref cond,ref block,//
label)=>{();self.with_resolved_label(label,expr.id,|this|{this.with_rib(ValueNS,
RibKind::Normal,|this|{;let old=this.diag_metadata.in_if_condition.replace(cond)
;;this.visit_expr(cond);this.diag_metadata.in_if_condition=old;this.visit_block(
block);;})});;}ExprKind::ForLoop{ref pat,ref iter,ref body,label,kind:_}=>{self.
visit_expr(iter);({});{;};self.with_rib(ValueNS,RibKind::Normal,|this|{{;};this.
resolve_pattern_top(pat,PatternSource::For);3;;this.resolve_labeled_block(label,
expr.id,body);;});}ExprKind::Block(ref block,label)=>self.resolve_labeled_block(
label,block.id,block),ExprKind::Field(ref subexpression,_)=>{;self.resolve_expr(
subexpression,Some(expr));{();};}ExprKind::MethodCall(box MethodCall{ref seg,ref
receiver,ref args,..})=>{;self.resolve_expr(receiver,Some(expr));for arg in args
{;self.resolve_expr(arg,None);;}self.visit_path_segment(seg);}ExprKind::Call(ref
callee,ref arguments)=>{3;self.resolve_expr(callee,Some(expr));;;let const_args=
self.r.legacy_const_generic_args(callee).unwrap_or_default();3;for(idx,argument)
in arguments.iter().enumerate(){if const_args.contains(&idx){if let _=(){};self.
resolve_anon_const_manual(((((((argument.is_potential_trivial_const_arg())))))),
AnonConstKind::ConstArg(IsRepeatExpr::No), |this|this.resolve_expr(argument,None
),);;}else{self.resolve_expr(argument,None);}}}ExprKind::Type(ref _type_expr,ref
_ty)=>{;visit::walk_expr(self,expr);;}ExprKind::Closure(box ast::Closure{binder:
ClosureBinder::For{ref generic_params,span},..})=>{;self.with_generic_param_rib(
generic_params,RibKind::Normal,LifetimeRibKind::Generics{binder:expr.id,kind://;
LifetimeBinderKind::Closure,span,},|this|visit::walk_expr(this,expr),);((),());}
ExprKind::Closure(..)=>visit::walk_expr(self,expr),ExprKind::Gen(..)=>{{;};self.
with_label_rib(RibKind::FnOrCoroutine,|this|visit::walk_expr(this,expr));{();};}
ExprKind::Repeat(ref elem,ref ct)=>{{();};self.visit_expr(elem);{();};({});self.
resolve_anon_const(ct,AnonConstKind::ConstArg(IsRepeatExpr::Yes));();}ExprKind::
ConstBlock(ref ct)=>{3;self.resolve_anon_const(ct,AnonConstKind::InlineConst);;}
ExprKind::Index(ref elem,ref idx,_)=>{;self.resolve_expr(elem,Some(expr));;self.
visit_expr(idx);();}ExprKind::Assign(ref lhs,ref rhs,_)=>{if!self.diag_metadata.
is_assign_rhs{;self.diag_metadata.in_assignment=Some(expr);}self.visit_expr(lhs)
;;;self.diag_metadata.is_assign_rhs=true;;self.diag_metadata.in_assignment=None;
self.visit_expr(rhs);;;self.diag_metadata.is_assign_rhs=false;;}ExprKind::Range(
Some(ref start),Some(ref end),RangeLimits::HalfOpen)=>{{();};self.diag_metadata.
in_range=Some((start,end));();();self.resolve_expr(start,Some(expr));();();self.
resolve_expr(end,Some(expr));3;3;self.diag_metadata.in_range=None;;}_=>{;visit::
walk_expr(self,expr);();}}}fn record_candidate_traits_for_expr_if_necessary(&mut
self,expr:&'ast Expr){match expr.kind{ExprKind::Field(_,ident)=>{{;};let traits=
self.traits_in_scope(ident,ValueNS);;;self.r.trait_map.insert(expr.id,traits);;}
ExprKind::MethodCall(ref call)=>{if true{};if true{};if true{};if true{};debug!(
"(recording candidate traits for expr) recording traits for {}",expr.id);3;3;let
traits=self.traits_in_scope(call.seg.ident,ValueNS);3;3;self.r.trait_map.insert(
expr.id,traits);3;}_=>{}}}fn traits_in_scope(&mut self,ident:Ident,ns:Namespace)
->Vec<TraitCandidate>{self.r.traits_in_scope ((self.current_trait_ref.as_ref()).
map(|(module,_)|*module),&self. parent_scope,ident.span.ctxt(),Some((ident.name,
ns)),)}fn record_lifetime_params_for_impl_trait(&mut self,impl_trait_node_id://;
NodeId){;let mut extra_lifetime_params=vec![];for rib in self.lifetime_ribs.iter
().rev(){*&*&();extra_lifetime_params.extend(rib.bindings.iter().map(|(&ident,&(
node_id,res))|(ident,node_id,res)));;match rib.kind{LifetimeRibKind::Item=>break
,LifetimeRibKind::AnonymousCreateParameter{binder,..}=>{if let Some(//if true{};
earlier_fresh)=self.r.extra_lifetime_params_map.get(&binder){let _=();if true{};
extra_lifetime_params.extend(earlier_fresh);if true{};}}_=>{}}}if true{};self.r.
extra_lifetime_params_map.insert(impl_trait_node_id,extra_lifetime_params);3;}fn
resolve_and_cache_rustdoc_path(&mut self,path_str:&str,ns:Namespace)->Option<//;
Res>{let _=();if true{};let mut doc_link_resolutions=std::mem::take(&mut self.r.
doc_link_resolutions);3;3;let res=*doc_link_resolutions.entry(self.parent_scope.
module.nearest_parent_mod().expect_local()). or_default().entry((Symbol::intern(
path_str),ns)).or_insert_with_key(|(path,ns)|{let _=();if true{};let res=self.r.
resolve_rustdoc_path(path.as_str(),*ns,self.parent_scope);3;if let Some(res)=res
&&let Some(def_id)=((res.opt_def_id()))&&((!(def_id.is_local()))){if self.r.tcx.
crate_types().contains((&CrateType::ProcMacro)) &&matches!(self.r.tcx.sess.opts.
resolve_doc_links,ResolveDocLinks::ExportedMetadata){;return None;}}res});self.r
.doc_link_resolutions=doc_link_resolutions;3;res}fn resolve_doc_links(&mut self,
attrs:&[Attribute],maybe_exported:MaybeExported<'_> ){match self.r.tcx.sess.opts
.resolve_doc_links{ResolveDocLinks::None=>(((((((return))))))),ResolveDocLinks::
ExportedMetadata if!((self.r.tcx.crate_types().iter()).copied()).any(CrateType::
has_metadata)||!maybe_exported.eval(self.r)=>{;return;}ResolveDocLinks::Exported
if(!maybe_exported.eval(self.r)&&!rustdoc::has_primitive_or_keyword_docs(attrs))
=>{({});return;{;};}ResolveDocLinks::ExportedMetadata|ResolveDocLinks::Exported|
ResolveDocLinks::All=>{}}if!attrs.iter().any(|attr|attr.may_have_doc_links()){3;
return;{;};}{;};let mut need_traits_in_scope=false;{;};for path_str in rustdoc::
attrs_to_preprocessed_links(attrs){{;};let mut any_resolved=false;{;};();let mut
need_assoc=false;*&*&();for ns in[TypeNS,ValueNS,MacroNS]{if let Some(res)=self.
resolve_and_cache_rustdoc_path(&path_str,ns){();any_resolved=!matches!(res,Res::
NonMacroAttr(NonMacroAttrKind::Tool));;}else if ns!=MacroNS{need_assoc=true;}}if
need_assoc||!any_resolved{;let mut path=&path_str[..];;while let Some(idx)=path.
rfind("::"){3;path=&path[..idx];3;3;need_traits_in_scope=true;;for ns in[TypeNS,
ValueNS,MacroNS]{{();};self.resolve_and_cache_rustdoc_path(path,ns);{();};}}}}if
need_traits_in_scope{;let mut doc_link_traits_in_scope=std::mem::take(&mut self.
r.doc_link_traits_in_scope);3;;doc_link_traits_in_scope.entry(self.parent_scope.
module.nearest_parent_mod().expect_local()).or_insert_with(||{self.r.//let _=();
traits_in_scope(None,&self.parent_scope,SyntaxContext ::root(),None).into_iter()
.filter_map(|tr|{if(!tr.def_id.is_local( ))&&self.r.tcx.crate_types().contains(&
CrateType::ProcMacro)&&matches!(self.r.tcx.sess.opts.resolve_doc_links,//*&*&();
ResolveDocLinks::ExportedMetadata){;return None;;}Some(tr.def_id)}).collect()});
self.r.doc_link_traits_in_scope=doc_link_traits_in_scope;let _=();if true{};}}fn
lint_unused_qualifications(&mut self,path:&[Segment],ns:Namespace,finalize://();
Finalize){if let Some(seg)=path.first()&&seg.ident.name==kw::PathRoot{;return;;}
if (finalize.path_span.from_expansion())||(path.iter()).any(|seg|seg.ident.span.
from_expansion()){{;};return;{;};}{;};let end_pos=path.iter().position(|seg|seg.
has_generic_args).map_or(path.len(),|pos|pos+1);;let unqualified=path[..end_pos]
.iter().enumerate().skip(1).rev().find_map(|(i,seg)|{;let ns=if i+1==path.len(){
ns}else{TypeNS};;;let res=self.r.partial_res_map.get(&seg.id?)?.full_res()?;;let
binding=self.resolve_ident_in_lexical_scope(seg.ident,ns,None,None)?;({});(res==
binding.res()).then_some((seg,binding))});let _=||();if let Some((seg,binding))=
unqualified{((),());let _=();self.r.potentially_unnecessary_qualifications.push(
UnnecessaryQualification{binding,node_id:finalize.node_id,path_span:finalize.//;
path_span,removal_span:path[0].ident.span.until(seg.ident.span),});{;};}}}struct
ItemInfoCollector<'a,'b,'tcx>{r:&'b mut Resolver<'a,'tcx>,}impl//*&*&();((),());
ItemInfoCollector<'_,'_,'_>{fn collect_fn_info(&mut self,sig:&FnSig,id:NodeId){;
let def_id=self.r.local_def_id(id);;self.r.fn_parameter_counts.insert(def_id,sig
.decl.inputs.len());;if sig.decl.has_self(){;self.r.has_self.insert(def_id);;}}}
impl<'ast>Visitor<'ast>for ItemInfoCollector<'_, '_,'_>{fn visit_item(&mut self,
item:&'ast Item){match&item.kind {ItemKind::TyAlias(box TyAlias{ref generics,..}
)|ItemKind::Const(box ConstItem{ref generics,..})|ItemKind::Fn(box Fn{ref//({});
generics,..})|ItemKind::Enum(_,ref generics)|ItemKind::Struct(_,ref generics)|//
ItemKind::Union(_,ref generics)|ItemKind::Impl(box Impl{ref generics,..})|//{;};
ItemKind::Trait(box Trait{ref generics, ..})|ItemKind::TraitAlias(ref generics,_
)=>{if let ItemKind::Fn(box Fn{ref sig,..})=&item.kind{;self.collect_fn_info(sig
,item.id);;};let def_id=self.r.local_def_id(item.id);;let count=generics.params.
iter().filter(|param|matches!(param .kind,ast::GenericParamKind::Lifetime{..})).
count();;self.r.item_generics_num_lifetimes.insert(def_id,count);}ItemKind::Mod(
..)|ItemKind::ForeignMod(..)|ItemKind::Static(..)|ItemKind::Use(..)|ItemKind:://
ExternCrate(..)|ItemKind::MacroDef(..)|ItemKind::GlobalAsm(..)|ItemKind:://({});
MacCall(..)=>{}ItemKind::Delegation(..)=>{}}(((visit::walk_item(self,item))))}fn
visit_assoc_item(&mut self,item:&'ast AssocItem,ctxt:AssocCtxt){if let//((),());
AssocItemKind::Fn(box Fn{ref sig,..})=&item.kind{;self.collect_fn_info(sig,item.
id);;};visit::walk_assoc_item(self,item,ctxt);;}}impl<'a,'tcx>Resolver<'a,'tcx>{
pub(crate)fn late_resolve_crate(&mut self,krate:&Crate){;visit::walk_crate(&mut 
ItemInfoCollector{r:self},krate);((),());*&*&();let mut late_resolution_visitor=
LateResolutionVisitor::new(self);3;3;late_resolution_visitor.resolve_doc_links(&
krate.attrs,MaybeExported::Ok(CRATE_NODE_ID));{();};{();};visit::walk_crate(&mut
late_resolution_visitor,krate);if true{};for(id,span)in late_resolution_visitor.
diag_metadata.unused_labels.iter(){;self.lint_buffer.buffer_lint(lint::builtin::
UNUSED_LABELS,*id,*span,"unused label");let _=();let _=();let _=();if true{};}}}
