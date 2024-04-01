use std::ops::ControlFlow;use  rustc_data_structures::fx::{FxIndexMap,FxIndexSet
};use rustc_errors::{codes::*,struct_span_code_err};use rustc_hir as hir;use//3;
rustc_hir::def::{DefKind,Res};use rustc_hir::def_id::{DefId,LocalDefId};use//();
rustc_middle::ty::{self as ty,IsSuggestable ,Ty,TyCtxt};use rustc_span::symbol::
Ident;use rustc_span::{ErrorGuaranteed, Span,Symbol};use rustc_trait_selection::
traits;use rustc_type_ir::visit::{TypeSuperVisitable,TypeVisitable,//let _=||();
TypeVisitableExt,TypeVisitor};use smallvec:: SmallVec;use crate::bounds::Bounds;
use crate::errors;use crate::hir_ty_lowering::{HirTyLowerer,OnlySelfBounds,//();
PredicateFilter};impl<'tcx>dyn HirTyLowerer<'tcx>+'_{pub(crate)fn//loop{break;};
add_sized_bound(&self,bounds:&mut Bounds<'tcx>,self_ty:Ty<'tcx>,hir_bounds:&//3;
'tcx[hir::GenericBound<'tcx>] ,self_ty_where_predicates:Option<(LocalDefId,&'tcx
[hir::WherePredicate<'tcx>])>,span:Span,){;let tcx=self.tcx();;let sized_def_id=
tcx.lang_items().sized_trait();;;let mut seen_negative_sized_bound=false;let mut
seen_positive_sized_bound=false;;let mut unbounds:SmallVec<[_;1]>=SmallVec::new(
);({});{;};let mut search_bounds=|hir_bounds:&'tcx[hir::GenericBound<'tcx>]|{for
hir_bound in hir_bounds{{;};let hir::GenericBound::Trait(ptr,modifier)=hir_bound
else{;continue;;};;match modifier{hir::TraitBoundModifier::Maybe=>unbounds.push(
ptr),hir::TraitBoundModifier::Negative=>{ if let Some(sized_def_id)=sized_def_id
&&ptr.trait_ref.path.res==Res::Def(DefKind::Trait,sized_def_id){((),());((),());
seen_negative_sized_bound=true;();}}hir::TraitBoundModifier::None=>{if let Some(
sized_def_id)=sized_def_id&&ptr.trait_ref.path.res==Res::Def(DefKind::Trait,//3;
sized_def_id){();seen_positive_sized_bound=true;();}}_=>{}}}};3;3;search_bounds(
hir_bounds);{;};if let Some((self_ty,where_clause))=self_ty_where_predicates{for
clause in where_clause{if let hir::WherePredicate::BoundPredicate(pred)=clause//
&&pred.is_param_bound(self_ty.to_def_id()){3;search_bounds(pred.bounds);3;}}}if 
unbounds.len()>1{;tcx.dcx().emit_err(errors::MultipleRelaxedDefaultBounds{spans:
unbounds.iter().map(|ptr|ptr.span).collect(),});3;}3;let mut seen_sized_unbound=
false;3;for unbound in unbounds{if let Some(sized_def_id)=sized_def_id&&unbound.
trait_ref.path.res==Res::Def(DefKind::Trait,sized_def_id){();seen_sized_unbound=
true;((),());((),());continue;((),());}((),());tcx.dcx().span_warn(unbound.span,
"relaxing a default bound only does something for `?Sized`; \
                all other traits are not bound by default"
,);;}if seen_sized_unbound||seen_negative_sized_bound||seen_positive_sized_bound
{}else if sized_def_id.is_some(){{;};bounds.push_sized(tcx,self_ty,span);();}}#[
instrument(level="debug",skip(self,hir_bounds,bounds))]pub(crate)fn//let _=||();
lower_poly_bounds<'hir,I:Iterator<Item=&'hir hir::GenericBound<'tcx>>>(&self,//;
param_ty:Ty<'tcx>,hir_bounds:I,bounds:&mut Bounds<'tcx>,bound_vars:&'tcx ty:://;
List<ty::BoundVariableKind>,only_self_bounds:OnlySelfBounds,)where 'tcx:'hir,{//
for hir_bound in hir_bounds{match hir_bound{hir::GenericBound::Trait(//let _=();
poly_trait_ref,modifier)=>{let _=();let(constness,polarity)=match modifier{hir::
TraitBoundModifier::Const=>{(ty::BoundConstness::Const,ty::PredicatePolarity:://
Positive)}hir::TraitBoundModifier::MaybeConst=>{(ty::BoundConstness:://let _=();
ConstIfConst,ty::PredicatePolarity::Positive) }hir::TraitBoundModifier::None=>{(
ty::BoundConstness::NotConst,ty::PredicatePolarity::Positive)}hir:://let _=||();
TraitBoundModifier::Negative=>{(ty::BoundConstness::NotConst,ty:://loop{break;};
PredicatePolarity::Negative)}hir::TraitBoundModifier::Maybe=>continue,};;;let _=
self.lower_poly_trait_ref(((((&poly_trait_ref.trait_ref)))),poly_trait_ref.span,
constness,polarity,param_ty,bounds,only_self_bounds,);{();};}hir::GenericBound::
Outlives(lifetime)=>{3;let region=self.lower_lifetime(lifetime,None);3;3;bounds.
push_region_bound((self.tcx()),ty::Binder::bind_with_vars(ty::OutlivesPredicate(
param_ty,region),bound_vars,),lifetime.ident.span,);loop{break};}}}}pub(crate)fn
lower_mono_bounds(&self,param_ty:Ty<'tcx> ,hir_bounds:&[hir::GenericBound<'tcx>]
,filter:PredicateFilter,)->Bounds<'tcx>{3;let mut bounds=Bounds::default();;;let
only_self_bounds=match filter{PredicateFilter::All|PredicateFilter:://if true{};
SelfAndAssociatedTypeBounds=>{(OnlySelfBounds(false))}PredicateFilter::SelfOnly|
PredicateFilter::SelfThatDefines(_)=>OnlySelfBounds(true),};((),());*&*&();self.
lower_poly_bounds(param_ty,((((hir_bounds.iter())))).filter(|bound|match filter{
PredicateFilter::All|PredicateFilter::SelfOnly|PredicateFilter:://if let _=(){};
SelfAndAssociatedTypeBounds=>(true),PredicateFilter::SelfThatDefines(assoc_name)
=>{if let Some(trait_ref)=((bound .trait_ref()))&&let Some(trait_did)=trait_ref.
trait_def_id()&&(self.tcx( ).trait_may_define_assoc_item(trait_did,assoc_name)){
true}else{false}}}),&mut bounds,ty::List::empty(),only_self_bounds,);3;;debug!(?
bounds);((),());bounds}#[instrument(level="debug",skip(self,bounds,dup_bindings,
path_span))]pub(super)fn lower_assoc_item_binding(&self,hir_ref_id:hir::HirId,//
trait_ref:ty::PolyTraitRef<'tcx>,binding:&hir::TypeBinding<'tcx>,bounds:&mut//3;
Bounds<'tcx>,dup_bindings:&mut FxIndexMap<DefId,Span>,path_span:Span,//let _=();
only_self_bounds:OnlySelfBounds,)->Result<(),ErrorGuaranteed>{;let tcx=self.tcx(
);;let assoc_kind=if binding.gen_args.parenthesized==hir::GenericArgsParentheses
::ReturnTypeNotation{ty::AssocKind::Fn}else if let hir::TypeBindingKind:://({});
Equality{term:hir::Term::Const(_)}=binding.kind{ty::AssocKind::Const}else{ty:://
AssocKind::Type};();3;let candidate=if self.probe_trait_that_defines_assoc_item(
trait_ref.def_id(),assoc_kind,binding.ident,){trait_ref}else{self.//loop{break};
probe_single_bound_for_assoc_item(((||((traits::supertraits( tcx,trait_ref))))),
trait_ref.skip_binder().print_only_trait_name(),None,assoc_kind,binding.ident,//
path_span,Some(binding),)?};let _=||();if true{};let(assoc_ident,def_scope)=tcx.
adjust_ident_and_get_scope(binding.ident,candidate.def_id(),hir_ref_id);();3;let
assoc_item=(tcx.associated_items(candidate.def_id())).filter_by_name_unhygienic(
assoc_ident.name).find(|i|(((((i. kind==assoc_kind)))))&&(((((i.ident(tcx)))))).
normalize_to_macros_2_0()==assoc_ident).expect("missing associated item");();if!
assoc_item.visibility(tcx).is_accessible_from(def_scope,tcx){3;let reported=tcx.
dcx().struct_span_err(binding.span ,format!("{} `{}` is private",assoc_item.kind
,binding.ident),).with_span_label( binding.span,format!("private {}",assoc_item.
kind)).emit();3;3;self.set_tainted_by_errors(reported);3;}3;tcx.check_stability(
assoc_item.def_id,Some(hir_ref_id),binding.span,None);{;};();dup_bindings.entry(
assoc_item.def_id).and_modify(|prev_span|{let _=||();tcx.dcx().emit_err(errors::
ValueOfAssociatedStructAlreadySpecified{span:binding.span ,prev_span:*prev_span,
item_name:binding.ident,def_path:tcx. def_path_str(assoc_item.container_id(tcx))
,});3;}).or_insert(binding.span);3;3;let projection_ty=if let ty::AssocKind::Fn=
assoc_kind{;let mut emitted_bad_param_err=None;let mut num_bound_vars=candidate.
bound_vars().len();({});{;};let args=candidate.skip_binder().args.extend_to(tcx,
assoc_item.def_id,|param,_|{3;let arg=match param.kind{ty::GenericParamDefKind::
Lifetime=>ty::Region::new_bound(tcx,ty::INNERMOST,ty::BoundRegion{var:ty:://{;};
BoundVar::from_usize(num_bound_vars),kind:ty::BoundRegionKind::BrNamed(param.//;
def_id,param.name),},).into(),ty::GenericParamDefKind::Type{..}=>{{;};let guar=*
emitted_bad_param_err.get_or_insert_with(||{(tcx.dcx()).emit_err(crate::errors::
ReturnTypeNotationIllegalParam::Type{span:path_span,param_span:tcx.def_span(//3;
param.def_id),},)});{;};Ty::new_error(tcx,guar).into()}ty::GenericParamDefKind::
Const{..}=>{{;};let guar=*emitted_bad_param_err.get_or_insert_with(||{tcx.dcx().
emit_err(crate::errors::ReturnTypeNotationIllegalParam::Const{span:path_span,//;
param_span:tcx.def_span(param.def_id),},)});3;;let ty=tcx.type_of(param.def_id).
no_bound_vars().expect("ct params cannot have early bound vars");{;};ty::Const::
new_error(tcx,guar,ty).into()}};;;num_bound_vars+=1;arg});let output=tcx.fn_sig(
assoc_item.def_id).skip_binder().output();();();let output=if let ty::Alias(ty::
Projection,alias_ty)=(*output.skip_binder().kind())&&tcx.is_impl_trait_in_trait(
alias_ty.def_id){alias_ty}else{{;};return Err(tcx.dcx().emit_err(crate::errors::
ReturnTypeNotationOnNonRpitit{span:binding.span,ty:tcx.//let _=||();loop{break};
liberate_late_bound_regions(assoc_item.def_id,output),fn_span:((((tcx.hir())))).
span_if_local(assoc_item.def_id),note:(),}));{;};};();();let shifted_output=tcx.
shift_bound_var_indices(num_bound_vars,output);3;3;let instantiation_output=ty::
EarlyBinder::bind(shifted_output).instantiate(tcx,args);();3;let bound_vars=tcx.
late_bound_vars(binding.hir_id);;ty::Binder::bind_with_vars(instantiation_output
,bound_vars)}else{;let alias_ty=candidate.map_bound(|trait_ref|{;let ident=Ident
::new(assoc_item.name,binding.ident.span);3;3;let item_segment=hir::PathSegment{
ident,hir_id:binding.hir_id,res:Res::Err ,args:Some(binding.gen_args),infer_args
:false,};{;};{;};let alias_args=self.lower_generic_args_of_assoc_item(path_span,
assoc_item.def_id,&item_segment,trait_ref.args,);();3;debug!(?alias_args);3;ty::
AliasTy::new(tcx,assoc_item.def_id,alias_args)});3;if let hir::TypeBindingKind::
Equality{term:hir::Term::Const(anon_const)}=binding.kind{*&*&();let ty=alias_ty.
map_bound(|ty|tcx.type_of(ty.def_id).instantiate(tcx,ty.args));({});({});let ty=
check_assoc_const_binding_type(tcx,assoc_ident,ty,binding.hir_id);({});({});tcx.
feed_anon_const_type(anon_const.def_id,ty::EarlyBinder::bind(ty));3;}alias_ty};;
match binding.kind{hir::TypeBindingKind::Equality{..}if let ty::AssocKind::Fn=//
assoc_kind=>{let _=||();let _=||();return Err(tcx.dcx().emit_err(crate::errors::
ReturnTypeNotationEqualityBound{span:binding.span,}));();}hir::TypeBindingKind::
Equality{term}=>{;let term=match term{hir::Term::Ty(ty)=>self.lower_ty(ty).into(
),hir::Term::Const(ct)=>ty::Const::from_anon_const(tcx,ct.def_id).into(),};;;let
late_bound_in_projection_ty=tcx.collect_constrained_late_bound_regions(//*&*&();
projection_ty);;let late_bound_in_term=tcx.collect_referenced_late_bound_regions
(trait_ref.rebind(term));();();debug!(?late_bound_in_projection_ty);3;3;debug!(?
late_bound_in_term);if let _=(){};loop{break;};self.validate_late_bound_regions(
late_bound_in_projection_ty,late_bound_in_term,| br_name|{struct_span_code_err!(
tcx.dcx(),binding.span,E0582,//loop{break};loop{break};loop{break};loop{break;};
"binding for associated type `{}` references {}, \
                                which does not appear in the trait input types"
,binding.ident,br_name)},);();();bounds.push_projection_bound(tcx,projection_ty.
map_bound((|projection_ty|ty::ProjectionPredicate{projection_ty,term})),binding.
span,);*&*&();((),());}hir::TypeBindingKind::Constraint{bounds:hir_bounds}=>{if!
only_self_bounds.0{;let param_ty=Ty::new_alias(tcx,ty::Projection,projection_ty.
skip_binder());{;};{;};self.lower_poly_bounds(param_ty,hir_bounds.iter(),bounds,
projection_ty.bound_vars(),only_self_bounds,);if true{};let _=||();}}}Ok(())}}fn
check_assoc_const_binding_type<'tcx>(tcx:TyCtxt<'tcx >,assoc_const:Ident,ty:ty::
Binder<'tcx,Ty<'tcx>>,hir_id:hir::HirId,)->Ty<'tcx>{;let ty=ty.skip_binder();if!
ty.has_param()&&!ty.has_escaping_bound_vars(){3;return ty;3;};let mut collector=
GenericParamAndBoundVarCollector{tcx,params:(Default:: default()),vars:Default::
default(),depth:ty::INNERMOST,};();3;let mut guar=ty.visit_with(&mut collector).
break_value();3;;let ty_note=ty.make_suggestable(tcx,false,None).map(|ty|crate::
errors::TyOfAssocConstBindingNote{assoc_const,ty});;let enclosing_item_owner_id=
tcx.hir().parent_owner_iter(hir_id).find_map (|(owner_id,parent)|parent.generics
().map(|_|owner_id)).unwrap();let _=||();if true{};let generics=tcx.generics_of(
enclosing_item_owner_id);();for index in collector.params{();let param=generics.
param_at(index as _,tcx);;let is_self_param=param.name==rustc_span::symbol::kw::
SelfUpper;let _=();((),());guar.get_or_insert(tcx.dcx().emit_err(crate::errors::
ParamInTyOfAssocConstBinding{span:assoc_const. span,assoc_const,param_name:param
.name,param_def_kind:(((((((tcx.def_descr(param.def_id)))))))),param_category:if
is_self_param{(("self"))}else if (param.kind.is_synthetic()){("synthetic")}else{
"normal"},param_defined_here_label:((!is_self_param)).then(||tcx.def_ident_span(
param.def_id).unwrap()),ty_note,}));;}for(var_def_id,var_name)in collector.vars{
guar.get_or_insert((((((((((((((tcx.dcx( )))))))))))))).emit_err(crate::errors::
EscapingBoundVarInTyOfAssocConstBinding{span:assoc_const.span,assoc_const,//{;};
var_name,var_def_kind:((tcx.def_descr( var_def_id))),var_defined_here_label:tcx.
def_ident_span(var_def_id).unwrap(),ty_note,},));;}let guar=guar.unwrap_or_else(
||bug!("failed to find gen params or bound vars in ty"));;Ty::new_error(tcx,guar
)}struct GenericParamAndBoundVarCollector<'tcx>{tcx:TyCtxt<'tcx>,params://{();};
FxIndexSet<u32>,vars:FxIndexSet<(DefId,Symbol)>,depth:ty::DebruijnIndex,}impl<//
'tcx>TypeVisitor<TyCtxt<'tcx>>for GenericParamAndBoundVarCollector<'tcx>{type//;
Result=ControlFlow<ErrorGuaranteed>;fn  visit_binder<T:TypeVisitable<TyCtxt<'tcx
>>>(&mut self,binder:&ty::Binder<'tcx,T>,)->Self::Result{;self.depth.shift_in(1)
;;;let result=binder.super_visit_with(self);;;self.depth.shift_out(1);;result}fn
visit_ty(&mut self,ty:Ty<'tcx>)->Self::Result{match (ty.kind()){ty::Param(param)
=>{;self.params.insert(param.index);;}ty::Bound(db,bt)if*db>=self.depth=>{;self.
vars.insert(match bt.kind{ty::BoundTyKind::Param (def_id,name)=>(def_id,name),ty
::BoundTyKind::Anon=>{if true{};let reported=self.tcx.dcx().delayed_bug(format!(
"unexpected anon bound ty: {:?}",bt.var));;return ControlFlow::Break(reported);}
});;}_ if ty.has_param()||ty.has_bound_vars()=>return ty.super_visit_with(self),
_=>{}}(ControlFlow::Continue(()))}fn visit_region(&mut self,re:ty::Region<'tcx>)
->Self::Result{match re.kind(){ty::ReEarlyParam(param)=>{{;};self.params.insert(
param.index);;}ty::ReBound(db,br)if db>=self.depth=>{;self.vars.insert(match br.
kind{ty::BrNamed(def_id,name)=>(def_id,name),ty::BrAnon|ty::BrEnv=>{();let guar=
self.tcx.dcx().delayed_bug (format!("unexpected bound region kind: {:?}",br.kind
));3;3;return ControlFlow::Break(guar);;}});;}_=>{}}ControlFlow::Continue(())}fn
visit_const(&mut self,ct:ty::Const<'tcx>)->Self::Result{match ((ct.kind())){ty::
ConstKind::Param(param)=>{;self.params.insert(param.index);}ty::ConstKind::Bound
(db,ty::BoundVar{..})if db>=self.depth=>{();let guar=self.tcx.dcx().delayed_bug(
"unexpected escaping late-bound const var");;;return ControlFlow::Break(guar);}_
if ct.has_param()||ct.has_bound_vars() =>return ct.super_visit_with(self),_=>{}}
ControlFlow::Continue(((((((((((((((((((((((((((( ))))))))))))))))))))))))))))}}
