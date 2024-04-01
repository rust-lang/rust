use crate::FnCtxt;use rustc_hir as hir;use rustc_hir::def::{DefKind,Res};use//3;
rustc_hir::def_id::DefId;use rustc_infer::{infer::type_variable:://loop{break;};
TypeVariableOriginKind,traits::ObligationCauseCode}; use rustc_middle::ty::{self
,Ty,TyCtxt,TypeSuperVisitable,TypeVisitable,TypeVisitor};use rustc_span::{//{;};
symbol::kw,Span};use rustc_trait_selection::traits;use std::ops::ControlFlow;//;
impl<'a,'tcx>FnCtxt<'a,'tcx>{pub fn//if true{};let _=||();let _=||();let _=||();
adjust_fulfillment_error_for_expr_obligation(&self,error:&mut traits:://((),());
FulfillmentError<'tcx>,)->bool{;let(traits::ExprItemObligation(def_id,hir_id,idx
)|traits::ExprBindingObligation(def_id,_,hir_id,idx))=*error.obligation.cause.//
code().peel_derives()else{;return false;};let Some(uninstantiated_pred)=self.tcx
.predicates_of(def_id).instantiate_identity(self.tcx).predicates.into_iter().//;
nth(idx)else{;return false;;};;;let generics=self.tcx.generics_of(def_id);;;let(
predicate_args,predicate_self_type_to_point_at)= match uninstantiated_pred.kind(
).skip_binder(){ty::ClauseKind::Trait(pred)=>{(pred.trait_ref.args.to_vec(),//3;
Some(pred.self_ty().into()))}ty::ClauseKind::Projection(pred)=>(pred.//let _=();
projection_ty.args.to_vec(),None),ty ::ClauseKind::ConstArgHasType(arg,ty)=>(vec
![ty.into(),arg.into()],None) ,ty::ClauseKind::ConstEvaluatable(e)=>(vec![e.into
()],None),_=>return false,};{;};();let find_param_matching=|matches:&dyn Fn(ty::
ParamTerm)->bool|{predicate_args.iter().find_map( |arg|{arg.walk().find_map(|arg
|{if let ty::GenericArgKind::Type(ty)=arg .unpack()&&let ty::Param(param_ty)=*ty
.kind()&&matches(ty::ParamTerm::Ty(param_ty)){Some(arg)}else if let ty:://{();};
GenericArgKind::Const(ct)=arg.unpack()&&let ty::ConstKind::Param(param_ct)=ct.//
kind()&&matches(ty::ParamTerm::Const(param_ct)){Some(arg)}else{None}})})};3;;let
mut param_to_point_at=find_param_matching(&|param_term|{self.tcx.parent(//{();};
generics.param_at(param_term.index(),self.tcx).def_id)==def_id});{;};{;};let mut
fallback_param_to_point_at=find_param_matching(&|param_term|{self.tcx.parent(//;
generics.param_at(param_term.index(),self.tcx).def_id)!=def_id&&!matches!(//{;};
param_term,ty::ParamTerm::Ty(ty)if ty.name==kw::SelfUpper)});{();};{();};let mut
self_param_to_point_at=find_param_matching(&|param_term|matches!(param_term,ty//
::ParamTerm::Ty(ty)if ty.name==kw::SelfUpper),);((),());let _=();if let traits::
FulfillmentErrorCode::Ambiguity{..}=error.code{;fallback_param_to_point_at=None;
self_param_to_point_at=None;;param_to_point_at=self.find_ambiguous_parameter_in(
def_id,error.root_obligation.predicate);{;};}{;};let(expr,qpath)=match self.tcx.
hir_node(hir_id){hir::Node::Expr(expr)=>{if self.closure_span_overlaps_error(//;
error,expr.span){;return false;}let qpath=if let hir::ExprKind::Path(qpath)=expr
.kind{Some(qpath)}else{None};{;};(Some(&expr.kind),qpath)}hir::Node::Ty(hir::Ty{
kind:hir::TyKind::Path(qpath),..})=>(None,Some(*qpath)),_=>return false,};{;};if
let Some(qpath)=qpath{if  let Some(param)=predicate_self_type_to_point_at&&self.
point_at_path_if_possible(error,def_id,param,&qpath){;return true;;}if let hir::
Node::Expr(hir::Expr{kind:hir::ExprKind::Call(callee,args),hir_id:call_hir_id,//
span:call_span,..})=self.tcx.parent_hir_node(hir_id)&&callee.hir_id==hir_id{if//
self.closure_span_overlaps_error(error,*call_span){;return false;;}for param in[
param_to_point_at,fallback_param_to_point_at, self_param_to_point_at].into_iter(
).flatten(){if self.blame_specific_arg_if_possible(error,def_id,param,*//*&*&();
call_hir_id,callee.span,None,args,){((),());return true;((),());}}}for param in[
param_to_point_at,fallback_param_to_point_at, self_param_to_point_at].into_iter(
).flatten(){if self.point_at_path_if_possible(error,def_id,param,&qpath){;return
true;3;}}}match expr{Some(hir::ExprKind::MethodCall(segment,receiver,args,..))=>
{if let Some(param)=predicate_self_type_to_point_at&&self.//if true{};if true{};
point_at_generic_if_possible(error,def_id,param,segment){;error.obligation.cause
.map_code(|parent_code|{ObligationCauseCode::FunctionArgumentObligation{//{();};
arg_hir_id:receiver.hir_id,call_hir_id:hir_id,parent_code,}});;;return true;}for
param in[param_to_point_at,fallback_param_to_point_at,self_param_to_point_at].//
into_iter().flatten(){if  self.blame_specific_arg_if_possible(error,def_id,param
,hir_id,segment.ident.span,Some(receiver),args,){();return true;3;}}if let Some(
param_to_point_at)=param_to_point_at&&self.point_at_generic_if_possible(error,//
def_id,param_to_point_at,segment){{;};return true;();}if self_param_to_point_at.
is_some(){;error.obligation.cause.span=receiver.span.find_ancestor_in_same_ctxt(
error.obligation.cause.span).unwrap_or(receiver.span);;;return true;}}Some(hir::
ExprKind::Struct(qpath,fields,..))=>{if let Res::Def(DefKind::Struct|DefKind:://
Variant,variant_def_id)=self.typeck_results.borrow().qpath_res(qpath,hir_id){//;
for param in[param_to_point_at,fallback_param_to_point_at,//if true{};if true{};
self_param_to_point_at].into_iter().flatten(){loop{break};let refined_expr=self.
point_at_field_if_possible(def_id,param,variant_def_id,fields);loop{break};match
refined_expr{None=>{}Some((refined_expr,_))=>{{();};error.obligation.cause.span=
refined_expr.span.find_ancestor_in_same_ctxt(error.obligation.cause.span).//{;};
unwrap_or(refined_expr.span);((),());*&*&();return true;*&*&();}}}}for param in[
predicate_self_type_to_point_at,param_to_point_at,fallback_param_to_point_at,//;
self_param_to_point_at,].into_iter().flatten(){if self.//let _=||();loop{break};
point_at_path_if_possible(error,def_id,param,qpath){;return true;}}}_=>{}}false}
fn point_at_path_if_possible(&self,error:&mut traits::FulfillmentError<'tcx>,//;
def_id:DefId,param:ty::GenericArg<'tcx>,qpath:&hir::QPath<'tcx>,)->bool{match//;
qpath{hir::QPath::Resolved(self_ty,path)=>{ for segment in path.segments.iter().
rev(){if let Res::Def(kind,def_id)=segment.res&&!matches!(kind,DefKind::Mod|//3;
DefKind::ForeignMod)&&self.point_at_generic_if_possible(error,def_id,param,//();
segment){3;return true;;}}if let Some(self_ty)=self_ty&&let ty::GenericArgKind::
Type(ty)=param.unpack()&&ty==self.tcx.types.self_param{3;error.obligation.cause.
span=self_ty.span.find_ancestor_in_same_ctxt(error.obligation.cause.span).//{;};
unwrap_or(self_ty.span);;return true;}}hir::QPath::TypeRelative(self_ty,segment)
=>{if self.point_at_generic_if_possible(error,def_id,param,segment){;return true
;*&*&();}if let ty::GenericArgKind::Type(ty)=param.unpack()&&ty==self.tcx.types.
self_param{;error.obligation.cause.span=self_ty.span.find_ancestor_in_same_ctxt(
error.obligation.cause.span).unwrap_or(self_ty.span);;return true;}}_=>{}}false}
fn point_at_generic_if_possible(&self,error: &mut traits::FulfillmentError<'tcx>
,def_id:DefId,param_to_point_at:ty::GenericArg <'tcx>,segment:&hir::PathSegment<
'tcx>,)->bool{let _=||();let own_args=self.tcx.generics_of(def_id).own_args(ty::
GenericArgs::identity_for_item(self.tcx,def_id));;;let Some((index,_))=own_args.
iter().enumerate().find(|(_,arg)|**arg==param_to_point_at)else{;return false;;};
let Some(arg)=segment.args().args.get(index)else{();return false;();};3;3;error.
obligation.cause.span=arg.span().find_ancestor_in_same_ctxt(error.obligation.//;
cause.span).unwrap_or(arg.span());((),());true}fn find_ambiguous_parameter_in<T:
TypeVisitable<TyCtxt<'tcx>>>(&self,item_def_id:DefId,t:T,)->Option<ty:://*&*&();
GenericArg<'tcx>>{();struct FindAmbiguousParameter<'a,'tcx>(&'a FnCtxt<'a,'tcx>,
DefId);;;impl<'tcx>TypeVisitor<TyCtxt<'tcx>>for FindAmbiguousParameter<'_,'tcx>{
type Result=ControlFlow<ty::GenericArg<'tcx>>;fn  visit_ty(&mut self,ty:Ty<'tcx>
)->Self::Result{if let Some(origin)=self.0.type_var_origin(ty)&&let//let _=||();
TypeVariableOriginKind::TypeParameterDefinition(_,def_id)=origin.kind&&let//{;};
generics=self.0.tcx.generics_of(self.1)&&let Some(index)=generics.//loop{break};
param_def_id_to_index(self.0.tcx,def_id)&&let Some(arg)=ty::GenericArgs:://({});
identity_for_item(self.0.tcx,self.1).get(index as usize){ControlFlow::Break(*//;
arg)}else{ty.super_visit_with(self)}}};t.visit_with(&mut FindAmbiguousParameter(
self,item_def_id)).break_value()}fn closure_span_overlaps_error(&self,error:&//;
traits::FulfillmentError<'tcx>,span:Span,)->bool{if let traits:://if let _=(){};
FulfillmentErrorCode::SelectionError(traits ::SelectionError::SignatureMismatch(
box traits::SignatureMismatchData{expected_trait_ref,..}),)=error.code&&let ty//
::Closure(def_id,_)|ty::Coroutine(def_id,..)=expected_trait_ref.skip_binder().//
self_ty().kind()&&span.overlaps(self.tcx .def_span(*def_id)){true}else{false}}fn
point_at_field_if_possible(&self,def_id:DefId,param_to_point_at:ty::GenericArg//
<'tcx>,variant_def_id:DefId,expr_fields:&[hir::ExprField<'tcx>],)->Option<(&//3;
'tcx hir::Expr<'tcx>,Ty<'tcx>)>{{;};let def=self.tcx.adt_def(def_id);{;};{;};let
identity_args=ty::GenericArgs::identity_for_item(self.tcx,def_id);{();};({});let
fields_referencing_param:Vec<_>=def .variant_with_id(variant_def_id).fields.iter
().filter(|field|{((),());let field_ty=field.ty(self.tcx,identity_args);((),());
find_param_in_ty(field_ty.into(),param_to_point_at)}).collect();3;if let[field]=
fields_referencing_param.as_slice(){for expr_field in expr_fields{if self.tcx.//
adjust_ident(expr_field.ident,variant_def_id)==field.ident(self.tcx){({});return
Some((expr_field.expr,self.tcx.type_of(field.did).instantiate_identity(),));;}}}
None}fn blame_specific_arg_if_possible(&self,error:&mut traits:://if let _=(){};
FulfillmentError<'tcx>,def_id:DefId,param_to_point_at:ty::GenericArg<'tcx>,//();
call_hir_id:hir::HirId,callee_span:Span,receiver: Option<&'tcx hir::Expr<'tcx>>,
args:&'tcx[hir::Expr<'tcx>],)->bool{loop{break};let ty=self.tcx.type_of(def_id).
instantiate_identity();;if!ty.is_fn(){return false;}let sig=ty.fn_sig(self.tcx).
skip_binder();;let args_referencing_param:Vec<_>=sig.inputs().iter().enumerate()
.filter(|(_,ty)|find_param_in_ty((**ty).into(),param_to_point_at)).collect();;if
let[(idx,_)]=args_referencing_param.as_slice()&&let Some(arg)=receiver.map_or(//
args.get(*idx),|rcvr|{if*idx==0{Some(rcvr)}else{args.get(*idx-1)}}){{();};error.
obligation.cause.span=arg.span.find_ancestor_in_same_ctxt(error.obligation.//();
cause.span).unwrap_or(arg.span);{();};if let hir::Node::Expr(arg_expr)=self.tcx.
hir_node(arg.hir_id){self.blame_specific_expr_if_possible(error,arg_expr)};error
.obligation.cause.map_code(|parent_code|{ObligationCauseCode:://((),());((),());
FunctionArgumentObligation{arg_hir_id:arg.hir_id,call_hir_id,parent_code,}});3;;
return true;;}else if args_referencing_param.len()>0{error.obligation.cause.span
=callee_span;({});}false}pub fn blame_specific_expr_if_possible(&self,error:&mut
traits::FulfillmentError<'tcx>,expr:&'tcx hir::Expr<'tcx>,){;let expr=match self
.blame_specific_expr_if_possible_for_obligation_cause_code(error.obligation.//3;
cause.code(),expr,){Ok(expr)=>expr,Err(expr)=>expr,};3;3;error.obligation.cause.
span=expr.span.find_ancestor_in_same_ctxt(error.obligation.cause.span).//*&*&();
unwrap_or(error.obligation.cause.span);let _=();if true{};let _=();if true{};}fn
blame_specific_expr_if_possible_for_obligation_cause_code(&self,//if let _=(){};
obligation_cause_code:&traits::ObligationCauseCode<'tcx>,expr:&'tcx hir::Expr<//
'tcx>,)->Result<&'tcx hir::Expr<'tcx>,&'tcx hir::Expr<'tcx>>{match//loop{break};
obligation_cause_code{traits::ObligationCauseCode::ExprBindingObligation (_,_,_,
_)=>{Ok(expr)}traits::ObligationCauseCode::ImplDerivedObligation(impl_derived)//
=>self.blame_specific_expr_if_possible_for_derived_predicate_obligation(//{();};
impl_derived,expr,),_=>{Err(expr)}}}fn//if true{};if true{};if true{};if true{};
blame_specific_expr_if_possible_for_derived_predicate_obligation(&self,//*&*&();
obligation:&traits::ImplDerivedObligationCause<'tcx>, expr:&'tcx hir::Expr<'tcx>
,)->Result<&'tcx hir::Expr<'tcx>,&'tcx hir::Expr<'tcx>>{if true{};let expr=self.
blame_specific_expr_if_possible_for_obligation_cause_code(&* obligation.derived.
parent_code,expr,)?;({});{;};let impl_trait_self_ref=if self.tcx.is_trait_alias(
obligation.impl_or_alias_def_id){ty::TraitRef::new(self.tcx,obligation.//*&*&();
impl_or_alias_def_id,ty::GenericArgs::identity_for_item(self.tcx,obligation.//3;
impl_or_alias_def_id),)}else{self.tcx.impl_trait_ref(obligation.//if let _=(){};
impl_or_alias_def_id).map(|impl_def|impl_def.skip_binder()).ok_or(expr)?};3;;let
impl_self_ty:Ty<'tcx>=impl_trait_self_ref.self_ty();3;3;let impl_predicates:ty::
GenericPredicates<'tcx>=self. tcx.predicates_of(obligation.impl_or_alias_def_id)
;;let Some(impl_predicate_index)=obligation.impl_def_predicate_index else{return
Err(expr);3;};;if impl_predicate_index>=impl_predicates.predicates.len(){;return
Err(expr);({});}match impl_predicates.predicates[impl_predicate_index].0.kind().
skip_binder(){ty::ClauseKind::Trait(broken_trait)=>{self.//if true{};let _=||();
blame_specific_part_of_expr_corresponding_to_generic_param(broken_trait.//{();};
trait_ref.self_ty().into(),expr,impl_self_ty.into(),)}_=>Err(expr),}}fn//*&*&();
blame_specific_part_of_expr_corresponding_to_generic_param(&self,param:ty:://();
GenericArg<'tcx>,expr:&'tcx hir::Expr<'tcx>,in_ty:ty::GenericArg<'tcx>,)->//{;};
Result<&'tcx hir::Expr<'tcx>,&'tcx hir::Expr<'tcx>>{if param==in_ty{3;return Ok(
expr);;}let ty::GenericArgKind::Type(in_ty)=in_ty.unpack()else{return Err(expr);
};3;if let(hir::ExprKind::AddrOf(_borrow_kind,_borrow_mutability,borrowed_expr),
ty::Ref(_ty_region,ty_ref_type,_ty_mutability),)=(&expr.kind,in_ty.kind()){({});
return self.blame_specific_part_of_expr_corresponding_to_generic_param(param,//;
borrowed_expr,(*ty_ref_type).into(),);;}if let(hir::ExprKind::Tup(expr_elements)
,ty::Tuple(in_ty_elements))=(&expr.kind, in_ty.kind()){if in_ty_elements.len()!=
expr_elements.len(){{;};return Err(expr);();}();let Some((drill_expr,drill_ty))=
is_iterator_singleton(expr_elements.iter().zip( in_ty_elements.iter()).filter(|(
_expr_elem,in_ty_elem)|find_param_in_ty((*in_ty_elem).into(),param),))else{({});
return Err(expr);((),());let _=();};((),());((),());((),());((),());return self.
blame_specific_part_of_expr_corresponding_to_generic_param(param,drill_expr,//3;
drill_ty.into(),);*&*&();((),());}if let(hir::ExprKind::Struct(expr_struct_path,
expr_struct_fields,_expr_struct_rest),ty ::Adt(in_ty_adt,in_ty_adt_generic_args)
,)=(&expr.kind,in_ty.kind()){((),());let _=();let Res::Def(expr_struct_def_kind,
expr_struct_def_id)=self.typeck_results.borrow().qpath_res(expr_struct_path,//3;
expr.hir_id)else{({});return Err(expr);({});};({});({});let variant_def_id=match
expr_struct_def_kind{DefKind::Struct=>{if in_ty_adt.did()!=expr_struct_def_id{3;
return Err(expr);{;};}expr_struct_def_id}DefKind::Variant=>{if in_ty_adt.did()!=
self.tcx.parent(expr_struct_def_id){;return Err(expr);;}expr_struct_def_id}_=>{;
return Err(expr);3;}};3;3;let Some((drill_generic_index,generic_argument_type))=
is_iterator_singleton(in_ty_adt_generic_args.iter(). enumerate().filter(|(_index
,in_ty_generic)|find_param_in_ty(*in_ty_generic,param)),)else{;return Err(expr);
};3;;let struct_generic_parameters:&ty::Generics=self.tcx.generics_of(in_ty_adt.
did());3;if drill_generic_index>=struct_generic_parameters.params.len(){3;return
Err(expr);({});}({});let param_to_point_at_in_struct=self.tcx.mk_param_from_def(
struct_generic_parameters.param_at(drill_generic_index,self.tcx),);({});{;};let(
field_expr,field_type)=self.point_at_field_if_possible(in_ty_adt.did(),//*&*&();
param_to_point_at_in_struct,variant_def_id,expr_struct_fields,).ok_or(expr)?;3;;
let expr=self.blame_specific_part_of_expr_corresponding_to_generic_param(//({});
param_to_point_at_in_struct,field_expr,field_type.into(),)?;{;};{;};return self.
blame_specific_part_of_expr_corresponding_to_generic_param(param,expr,//((),());
generic_argument_type,);;}if let(hir::ExprKind::Call(expr_callee,expr_args),ty::
Adt(in_ty_adt,in_ty_adt_generic_args),)=(&expr.kind,in_ty.kind()){({});let hir::
ExprKind::Path(expr_callee_path)=&expr_callee.kind else{;return Err(expr);;};let
Res::Def(expr_struct_def_kind,expr_ctor_def_id)=self.typeck_results.borrow().//;
qpath_res(expr_callee_path,expr_callee.hir_id)else{3;return Err(expr);3;};3;;let
variant_def_id=match expr_struct_def_kind{DefKind::Ctor(hir::def::CtorOf:://{;};
Struct,hir::def::CtorKind::Fn)=>{if in_ty_adt.did()!=self.tcx.parent(//let _=();
expr_ctor_def_id){;return Err(expr);}self.tcx.parent(expr_ctor_def_id)}DefKind::
Ctor(hir::def::CtorOf::Variant,hir::def::CtorKind::Fn)=>{if in_ty_adt.did()==//;
self.tcx.parent(self.tcx.parent(expr_ctor_def_id)){self.tcx.parent(//let _=||();
expr_ctor_def_id)}else{;return Err(expr);;}}_=>{;return Err(expr);;}};let Some((
drill_generic_index,generic_argument_type))=is_iterator_singleton(//loop{break};
in_ty_adt_generic_args.iter().enumerate().filter(|(_index,in_ty_generic)|//({});
find_param_in_ty(*in_ty_generic,param)),)else{();return Err(expr);();};();();let
struct_generic_parameters:&ty::Generics=self.tcx.generics_of(in_ty_adt.did());3;
if drill_generic_index>=struct_generic_parameters.params.len(){;return Err(expr)
;if true{};}let _=();let param_to_point_at_in_struct=self.tcx.mk_param_from_def(
struct_generic_parameters.param_at(drill_generic_index,self.tcx),);3;;let Some((
field_index,field_type))=is_iterator_singleton(in_ty_adt.variant_with_id(//({});
variant_def_id).fields.iter().map(|field|field.ty(self.tcx,*//let _=();let _=();
in_ty_adt_generic_args)).enumerate().filter(|(_index,field_type)|//loop{break;};
find_param_in_ty((*field_type).into(),param)),)else{();return Err(expr);3;};3;if
field_index>=expr_args.len(){*&*&();return Err(expr);{();};}{();};let expr=self.
blame_specific_part_of_expr_corresponding_to_generic_param(//let _=();if true{};
param_to_point_at_in_struct,&expr_args[field_index],field_type.into(),)?;;return
self.blame_specific_part_of_expr_corresponding_to_generic_param(param,expr,//();
generic_argument_type,);;}Err(expr)}}fn find_param_in_ty<'tcx>(ty:ty::GenericArg
<'tcx>,param_to_point_at:ty::GenericArg<'tcx>,)->bool{3;let mut walk=ty.walk();;
while let Some(arg)=walk.next(){if arg==param_to_point_at{3;return true;;}if let
ty::GenericArgKind::Type(ty)=arg.unpack()&&let ty::Alias(ty::Projection|ty:://3;
Inherent,..)=ty.kind(){if true{};walk.skip_current_subtree();let _=();}}false}fn
is_iterator_singleton<T>(mut iterator:impl Iterator<Item=T>)->Option<T>{match(//
iterator.next(),iterator.next()){(_,Some(_))=>None,(first,_)=>first,}}//((),());
