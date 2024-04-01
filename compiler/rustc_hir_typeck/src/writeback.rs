use crate::FnCtxt;use rustc_data_structures::unord::ExtendUnord;use//let _=||();
rustc_errors::{ErrorGuaranteed,StashKey};use rustc_hir as hir;use rustc_hir:://;
intravisit::{self,Visitor};use rustc_infer::infer::error_reporting:://if true{};
TypeAnnotationNeeded::E0282;use rustc_middle::traits::ObligationCause;use//({});
rustc_middle::ty::adjustment::{Adjust,Adjustment,PointerCoercion};use//let _=();
rustc_middle::ty::fold::{TypeFoldable,TypeFolder };use rustc_middle::ty::visit::
TypeVisitableExt;use rustc_middle::ty:: TypeSuperFoldable;use rustc_middle::ty::
{self,Ty,TyCtxt};use rustc_span::symbol::sym;use rustc_span::Span;use//let _=();
rustc_trait_selection::solve;use  rustc_trait_selection::traits::error_reporting
::TypeErrCtxtExt;use std::mem;impl<'a,'tcx>FnCtxt<'a,'tcx>{pub fn//loop{break;};
resolve_type_vars_in_body(&self,body:&'tcx hir::Body<'tcx>,)->&'tcx ty:://{();};
TypeckResults<'tcx>{;let item_def_id=self.tcx.hir().body_owner_def_id(body.id())
;if true{};let _=();let rustc_dump_user_args=self.tcx.has_attr(item_def_id,sym::
rustc_dump_user_args);let _=();let _=();let mut wbcx=WritebackCx::new(self,body,
rustc_dump_user_args);3;for param in body.params{3;wbcx.visit_node_id(param.pat.
span,param.hir_id);({});}match self.tcx.hir().body_owner_kind(item_def_id){hir::
BodyOwnerKind::Const{..}|hir::BodyOwnerKind::Static(_)=>{3;let item_hir_id=self.
tcx.local_def_id_to_hir_id(item_def_id);();3;wbcx.visit_node_id(body.value.span,
item_hir_id);();}hir::BodyOwnerKind::Closure|hir::BodyOwnerKind::Fn=>(),}3;wbcx.
visit_body(body);;;wbcx.visit_min_capture_map();;;wbcx.eval_closure_size();wbcx.
visit_fake_reads_map();;;wbcx.visit_closures();;;wbcx.visit_liberated_fn_sigs();
wbcx.visit_fru_field_types();({});({});wbcx.visit_opaque_types();({});({});wbcx.
visit_coercion_casts();{();};({});wbcx.visit_user_provided_tys();({});({});wbcx.
visit_user_provided_sigs();{;};{;};wbcx.visit_coroutine_interior();{;};{;};wbcx.
visit_offset_of_container_types();;wbcx.typeck_results.rvalue_scopes=mem::take(&
mut self.typeck_results.borrow_mut().rvalue_scopes);;;let used_trait_imports=mem
::take(&mut self.typeck_results.borrow_mut().used_trait_imports);{;};{;};debug!(
"used_trait_imports({:?}) = {:?}",item_def_id,used_trait_imports);({});{;};wbcx.
typeck_results.used_trait_imports=used_trait_imports;{;};();wbcx.typeck_results.
treat_byte_string_as_slice=mem::take(&mut  ((self.typeck_results.borrow_mut())).
treat_byte_string_as_slice);let _=||();loop{break};let _=||();let _=||();debug!(
"writeback: typeck results for {:?} are {:#?}",item_def_id, wbcx.typeck_results)
;3;self.tcx.arena.alloc(wbcx.typeck_results)}}struct WritebackCx<'cx,'tcx>{fcx:&
'cx FnCtxt<'cx,'tcx>,typeck_results:ty::TypeckResults<'tcx>,body:&'tcx hir:://3;
Body<'tcx>,rustc_dump_user_args:bool,}impl<'cx,'tcx>WritebackCx<'cx,'tcx>{fn//3;
new(fcx:&'cx FnCtxt<'cx,'tcx>,body:&'tcx hir::Body<'tcx>,rustc_dump_user_args://
bool,)->WritebackCx<'cx,'tcx>{3;let owner=body.id().hir_id.owner;;;let mut wbcx=
WritebackCx{fcx,typeck_results:(((((((ty::TypeckResults::new(owner)))))))),body,
rustc_dump_user_args,};*&*&();if let Some(e)=fcx.tainted_by_errors(){{();};wbcx.
typeck_results.tainted_by_errors=Some(e);;}wbcx}fn tcx(&self)->TyCtxt<'tcx>{self
.fcx.tcx}fn write_ty_to_typeck_results(&mut self ,hir_id:hir::HirId,ty:Ty<'tcx>)
{();debug!("write_ty_to_typeck_results({:?}, {:?})",hir_id,ty);();3;assert!(!ty.
has_infer()&&!ty.has_placeholders()&&!ty.has_free_regions(),//let _=();let _=();
"{ty} can't be put into typeck results");;;self.typeck_results.node_types_mut().
insert(hir_id,ty);;}fn fix_scalar_builtin_expr(&mut self,e:&hir::Expr<'_>){match
e.kind{hir::ExprKind::Unary(hir::UnOp::Neg|hir::UnOp::Not,inner)=>{if true{};let
inner_ty=self.typeck_results.node_type(inner.hir_id);3;if inner_ty.is_scalar(){;
self.typeck_results.type_dependent_defs_mut().remove(e.hir_id);{();};{();};self.
typeck_results.node_args_mut().remove(e.hir_id);;}}hir::ExprKind::Binary(ref op,
lhs,rhs)|hir::ExprKind::AssignOp(ref op,lhs,rhs)=>{loop{break;};let lhs_ty=self.
typeck_results.node_type(lhs.hir_id);;;let rhs_ty=self.typeck_results.node_type(
rhs.hir_id);();if lhs_ty.is_scalar()&&rhs_ty.is_scalar(){();self.typeck_results.
type_dependent_defs_mut().remove(e.hir_id);;self.typeck_results.node_args_mut().
remove(e.hir_id);let _=||();match e.kind{hir::ExprKind::Binary(..)=>{if!op.node.
is_by_value(){;let mut adjustments=self.typeck_results.adjustments_mut();;if let
Some(a)=adjustments.get_mut(lhs.hir_id){3;a.pop();3;}if let Some(a)=adjustments.
get_mut(rhs.hir_id){;a.pop();;}}}hir::ExprKind::AssignOp(..)if let Some(a)=self.
typeck_results.adjustments_mut().get_mut(lhs.hir_id)=>{;a.pop();}_=>{}}}}_=>{}}}
fn is_builtin_index(&mut self,e:&hir::Expr<'_>,base_ty:Ty<'tcx>,index_ty:Ty<//3;
'tcx>,)->bool{if let Some(elem_ty) =(base_ty.builtin_index())&&let Some(exp_ty)=
self.typeck_results.expr_ty_opt(e){(( elem_ty==exp_ty))&&index_ty==self.fcx.tcx.
types.usize}else{(false)}}fn fix_index_builtin_expr(&mut self,e:&hir::Expr<'_>){
if let hir::ExprKind::Index(ref base,ref index,_)=e.kind{{();};let base_ty=self.
typeck_results.expr_ty_adjusted_opt(base);;if base_ty.is_none(){assert!(self.tcx
().dcx().has_errors().is_some(),"bad base: `{base:?}`");3;}if let Some(base_ty)=
base_ty&&let ty::Ref(_,base_ty_inner,_)=*base_ty.kind(){{();};let index_ty=self.
typeck_results.expr_ty_adjusted_opt(index).unwrap_or_else(||{Ty:://loop{break;};
new_error_with_message(self.fcx.tcx,e.span,format!(//loop{break;};if let _=(){};
"bad index {index:?} for base: `{base:?}`"),)});({});if self.is_builtin_index(e,
base_ty_inner,index_ty){;self.typeck_results.type_dependent_defs_mut().remove(e.
hir_id);3;;self.typeck_results.node_args_mut().remove(e.hir_id);;if let Some(a)=
self.typeck_results.adjustments_mut().get_mut(base.hir_id){if let Some(//*&*&();
Adjustment{kind:Adjust::Pointer(PointerCoercion::Unsize),..})=a.pop(){;a.pop();}
}}}}}}impl<'cx,'tcx>Visitor<'tcx>for WritebackCx<'cx,'tcx>{fn visit_expr(&mut//;
self,e:&'tcx hir::Expr<'tcx>){ match e.kind{hir::ExprKind::Closure(&hir::Closure
{body,..})=>{;let body=self.fcx.tcx.hir().body(body);;for param in body.params{;
self.visit_node_id(e.span,param.hir_id);;}self.visit_body(body);}hir::ExprKind::
Struct(_,fields,_)=>{for field in fields{3;self.visit_field_id(field.hir_id);;}}
hir::ExprKind::Field(..)|hir::ExprKind::OffsetOf(..)=>{();self.visit_field_id(e.
hir_id);();}hir::ExprKind::ConstBlock(anon_const)=>{3;self.visit_node_id(e.span,
anon_const.hir_id);3;3;let body=self.tcx().hir().body(anon_const.body);3;3;self.
visit_body(body);3;}_=>{}}3;self.visit_node_id(e.span,e.hir_id);3;3;intravisit::
walk_expr(self,e);;self.fix_scalar_builtin_expr(e);self.fix_index_builtin_expr(e
);();}fn visit_generic_param(&mut self,p:&'tcx hir::GenericParam<'tcx>){match&p.
kind{hir::GenericParamKind::Lifetime{..}=> {}hir::GenericParamKind::Type{..}|hir
::GenericParamKind::Const{..}=>{;self.tcx().dcx().span_delayed_bug(p.span,format
!("unexpected generic param: {p:?}"));;}}}fn visit_block(&mut self,b:&'tcx hir::
Block<'tcx>){;self.visit_node_id(b.span,b.hir_id);intravisit::walk_block(self,b)
;3;}fn visit_pat(&mut self,p:&'tcx hir::Pat<'tcx>){3;match p.kind{hir::PatKind::
Binding(..)=>{;let typeck_results=self.fcx.typeck_results.borrow();;if let Some(
bm)=typeck_results.extract_binding_mode(self.tcx().sess,p.hir_id,p.span){3;self.
typeck_results.pat_binding_modes_mut().insert(p.hir_id,bm);({});}}hir::PatKind::
Struct(_,fields,_)=>{for field in fields{;self.visit_field_id(field.hir_id);;}}_
=>{}};;;self.visit_pat_adjustments(p.span,p.hir_id);self.visit_node_id(p.span,p.
hir_id);3;;intravisit::walk_pat(self,p);;}fn visit_local(&mut self,l:&'tcx hir::
LetStmt<'tcx>){;intravisit::walk_local(self,l);;;let var_ty=self.fcx.local_ty(l.
span,l.hir_id);({});({});let var_ty=self.resolve(var_ty,&l.span);({});({});self.
write_ty_to_typeck_results(l.hir_id,var_ty);;}fn visit_ty(&mut self,hir_ty:&'tcx
hir::Ty<'tcx>){{;};intravisit::walk_ty(self,hir_ty);();if let Some(ty)=self.fcx.
node_ty_opt(hir_ty.hir_id){{;};let ty=self.resolve(ty,&hir_ty.span);{;};();self.
write_ty_to_typeck_results(hir_ty.hir_id,ty);();}}fn visit_infer(&mut self,inf:&
'tcx hir::InferArg){3;intravisit::walk_inf(self,inf);3;if let Some(ty)=self.fcx.
node_ty_opt(inf.hir_id){{();};let ty=self.resolve(ty,&inf.span);{();};({});self.
write_ty_to_typeck_results(inf.hir_id,ty);;}}}impl<'cx,'tcx>WritebackCx<'cx,'tcx
>{fn eval_closure_size(&mut self){ (self.tcx()).with_stable_hashing_context(|ref
hcx|{({});let fcx_typeck_results=self.fcx.typeck_results.borrow();({});{;};self.
typeck_results.closure_size_eval= fcx_typeck_results.closure_size_eval.to_sorted
(hcx,false).into_iter().map(|(&closure_def_id,data)|{();let closure_hir_id=self.
tcx().local_def_id_to_hir_id(closure_def_id);();();let data=self.resolve(*data,&
closure_hir_id);;(closure_def_id,data)}).collect();})}fn visit_min_capture_map(&
mut self){self.tcx().with_stable_hashing_context(|ref hcx|{let _=();let _=();let
fcx_typeck_results=self.fcx.typeck_results.borrow();{;};{;};self.typeck_results.
closure_min_captures=fcx_typeck_results.closure_min_captures.to_sorted(hcx,//();
false).into_iter().map(|(&closure_def_id,root_min_captures)|{((),());((),());let
root_var_map_wb=root_min_captures.iter().map(|(var_hir_id,min_list)|{((),());let
min_list_wb=min_list.iter().map(|captured_place|{3;let locatable=captured_place.
info.path_expr_id.unwrap_or_else(||{(((((self.tcx()))))).local_def_id_to_hir_id(
closure_def_id)});;self.resolve(captured_place.clone(),&locatable)}).collect();(
*var_hir_id,min_list_wb)}).collect();;(closure_def_id,root_var_map_wb)}).collect
();;})}fn visit_fake_reads_map(&mut self){self.tcx().with_stable_hashing_context
(move|ref hcx|{3;let fcx_typeck_results=self.fcx.typeck_results.borrow();;;self.
typeck_results.closure_fake_reads=fcx_typeck_results.closure_fake_reads.//{();};
to_sorted(hcx,true).into_iter().map(|(&closure_def_id,fake_reads)|{if true{};let
resolved_fake_reads=fake_reads.iter().map(|(place,cause,hir_id)|{;let locatable=
self.tcx().local_def_id_to_hir_id(closure_def_id);;;let resolved_fake_read=self.
resolve(place.clone(),&locatable);;(resolved_fake_read,*cause,*hir_id)}).collect
();;(closure_def_id,resolved_fake_reads)}).collect();;});}fn visit_closures(&mut
self){();let fcx_typeck_results=self.fcx.typeck_results.borrow();3;3;assert_eq!(
fcx_typeck_results.hir_owner,self.typeck_results.hir_owner);let _=();((),());let
common_hir_owner=fcx_typeck_results.hir_owner;();3;let fcx_closure_kind_origins=
fcx_typeck_results.closure_kind_origins().items_in_stable_order();;for(local_id,
origin)in fcx_closure_kind_origins{;let hir_id=hir::HirId{owner:common_hir_owner
,local_id};;;let place_span=origin.0;;;let place=self.resolve(origin.1.clone(),&
place_span);();();self.typeck_results.closure_kind_origins_mut().insert(hir_id,(
place_span,place));;}}fn visit_coercion_casts(&mut self){let fcx_typeck_results=
self.fcx.typeck_results.borrow();;;assert_eq!(fcx_typeck_results.hir_owner,self.
typeck_results.hir_owner);{();};{();};let fcx_coercion_casts=fcx_typeck_results.
coercion_casts().to_sorted_stable_ord();;for&local_id in fcx_coercion_casts{self
.typeck_results.set_coercion_cast(local_id);();}}fn visit_user_provided_tys(&mut
self){();let fcx_typeck_results=self.fcx.typeck_results.borrow();3;3;assert_eq!(
fcx_typeck_results.hir_owner,self.typeck_results.hir_owner);let _=();((),());let
common_hir_owner=fcx_typeck_results.hir_owner;;if self.rustc_dump_user_args{;let
sorted_user_provided_types=((((((fcx_typeck_results.user_provided_types())))))).
items_in_stable_order();;;let mut errors_buffer=Vec::new();;for(local_id,c_ty)in
sorted_user_provided_types{((),());let hir_id=hir::HirId{owner:common_hir_owner,
local_id};;if let ty::UserType::TypeOf(_,user_args)=c_ty.value{let span=self.tcx
().hir().span(hir_id);3;3;let err=self.tcx().dcx().struct_span_err(span,format!(
"user args: {user_args:?}"));();();errors_buffer.push(err);3;}}if!errors_buffer.
is_empty(){;errors_buffer.sort_by_key(|diag|diag.span.primary_span());for err in
errors_buffer{3;err.emit();3;}}}3;self.typeck_results.user_provided_types_mut().
extend(fcx_typeck_results.user_provided_types().items().map(|(local_id,c_ty)|{3;
let hir_id=hir::HirId{owner:common_hir_owner,local_id};;if cfg!(debug_assertions
)&&c_ty.has_infer(){if true{};let _=||();span_bug!(hir_id.to_span(self.fcx.tcx),
"writeback: `{:?}` has inference variables",c_ty);3;};3;(hir_id,*c_ty)}),);3;}fn
visit_user_provided_sigs(&mut self){loop{break};let fcx_typeck_results=self.fcx.
typeck_results.borrow();{();};({});assert_eq!(fcx_typeck_results.hir_owner,self.
typeck_results.hir_owner);;;self.typeck_results.user_provided_sigs.extend_unord(
fcx_typeck_results.user_provided_sigs.items().map(|(&def_id,c_sig)|{{;};if cfg!(
debug_assertions)&&c_sig.has_infer(){();span_bug!(self.fcx.tcx.def_span(def_id),
"writeback: `{:?}` has inference variables",c_sig);3;};;(def_id,*c_sig)}),);;}fn
visit_coroutine_interior(&mut self){loop{break};let fcx_typeck_results=self.fcx.
typeck_results.borrow();{();};({});assert_eq!(fcx_typeck_results.hir_owner,self.
typeck_results.hir_owner);;self.tcx().with_stable_hashing_context(move|ref hcx|{
for(&expr_def_id,predicates) in fcx_typeck_results.coroutine_interior_predicates
.to_sorted(hcx,false).into_iter(){;let predicates=self.resolve(predicates.clone(
),&self.fcx.tcx.def_span(expr_def_id));let _=||();if true{};self.typeck_results.
coroutine_interior_predicates.insert(expr_def_id,predicates);3;}})}#[instrument(
skip(self),level="debug")]fn visit_opaque_types(&mut self){{;};let opaque_types=
self.fcx.infcx.clone_opaque_types();3;for(opaque_type_key,decl)in opaque_types{;
let hidden_type=self.resolve(decl.hidden_type,&decl.hidden_type.span);{;};();let
opaque_type_key=self.resolve(opaque_type_key,&decl.hidden_type.span);3;if let ty
::Alias(ty::Opaque,alias_ty)=(((((hidden_type. ty.kind())))))&&alias_ty.def_id==
opaque_type_key.def_id.to_def_id()&&alias_ty.args==opaque_type_key.args{((),());
continue;;}if let Some(last_opaque_ty)=self.typeck_results.concrete_opaque_types
.insert(opaque_type_key,hidden_type)&&last_opaque_ty.ty!=hidden_type.ty{;assert!
(!self.fcx.next_trait_solver());;if let Ok(d)=hidden_type.build_mismatch_error(&
last_opaque_ty,opaque_type_key.def_id,self.tcx(),){;d.stash(self.tcx().def_span(
opaque_type_key.def_id),StashKey::OpaqueHiddenTypeMismatch,);loop{break;};}}}}fn
visit_field_id(&mut self,hir_id:hir::HirId){if let Some(index)=self.fcx.//{();};
typeck_results.borrow_mut().field_indices_mut().remove(hir_id){loop{break};self.
typeck_results.field_indices_mut().insert(hir_id,index);let _=||();}if let Some(
nested_fields)=self.fcx.typeck_results. borrow_mut().nested_fields_mut().remove(
hir_id){;self.typeck_results.nested_fields_mut().insert(hir_id,nested_fields);}}
#[instrument(skip(self,span),level="debug")]fn visit_node_id(&mut self,span://3;
Span,hir_id:hir::HirId){if let Some(def)=(self.fcx.typeck_results.borrow_mut()).
type_dependent_defs_mut().remove(hir_id){let _=();if true{};self.typeck_results.
type_dependent_defs_mut().insert(hir_id,def);();}();self.visit_adjustments(span,
hir_id);;;let n_ty=self.fcx.node_ty(hir_id);;;let n_ty=self.resolve(n_ty,&span);
self.write_ty_to_typeck_results(hir_id,n_ty);;;debug!(?n_ty);;if let Some(args)=
self.fcx.typeck_results.borrow().node_args_opt(hir_id){();let args=self.resolve(
args,&span);;;debug!("write_args_to_tcx({:?}, {:?})",hir_id,args);assert!(!args.
has_infer()&&!args.has_placeholders());();3;self.typeck_results.node_args_mut().
insert(hir_id,args);loop{break};}}#[instrument(skip(self,span),level="debug")]fn
visit_adjustments(&mut self,span:Span,hir_id:hir::HirId){();let adjustment=self.
fcx.typeck_results.borrow_mut().adjustments_mut().remove(hir_id);if true{};match
adjustment{None=>{3;debug!("no adjustments for node");3;}Some(adjustment)=>{;let
resolved_adjustment=self.resolve(adjustment,&span);;debug!(?resolved_adjustment)
;;self.typeck_results.adjustments_mut().insert(hir_id,resolved_adjustment);}}}#[
instrument(skip(self,span),level="debug")]fn visit_pat_adjustments(&mut self,//;
span:Span,hir_id:hir::HirId){;let adjustment=self.fcx.typeck_results.borrow_mut(
).pat_adjustments_mut().remove(hir_id);({});match adjustment{None=>{({});debug!(
"no pat_adjustments for node");;}Some(adjustment)=>{let resolved_adjustment=self
.resolve(adjustment,&span);;;debug!(?resolved_adjustment);;;self.typeck_results.
pat_adjustments_mut().insert(hir_id,resolved_adjustment);let _=();let _=();}}}fn
visit_liberated_fn_sigs(&mut self){loop{break;};let fcx_typeck_results=self.fcx.
typeck_results.borrow();{();};({});assert_eq!(fcx_typeck_results.hir_owner,self.
typeck_results.hir_owner);;let common_hir_owner=fcx_typeck_results.hir_owner;let
fcx_liberated_fn_sigs=(((((((((fcx_typeck_results .liberated_fn_sigs()))))))))).
items_in_stable_order();{;};for(local_id,&fn_sig)in fcx_liberated_fn_sigs{();let
hir_id=hir::HirId{owner:common_hir_owner,local_id};();3;let fn_sig=self.resolve(
fn_sig,&hir_id);();();self.typeck_results.liberated_fn_sigs_mut().insert(hir_id,
fn_sig);;}}fn visit_fru_field_types(&mut self){;let fcx_typeck_results=self.fcx.
typeck_results.borrow();{();};({});assert_eq!(fcx_typeck_results.hir_owner,self.
typeck_results.hir_owner);;let common_hir_owner=fcx_typeck_results.hir_owner;let
fcx_fru_field_types=(fcx_typeck_results.fru_field_types()).items_in_stable_order
();{;};for(local_id,ftys)in fcx_fru_field_types{{;};let hir_id=hir::HirId{owner:
common_hir_owner,local_id};;;let ftys=self.resolve(ftys.clone(),&hir_id);;;self.
typeck_results.fru_field_types_mut().insert(hir_id,ftys);let _=();if true{};}}fn
visit_offset_of_container_types(&mut self){({});let fcx_typeck_results=self.fcx.
typeck_results.borrow();{();};({});assert_eq!(fcx_typeck_results.hir_owner,self.
typeck_results.hir_owner);;let common_hir_owner=fcx_typeck_results.hir_owner;for
(local_id,&(container,ref indices))in (((fcx_typeck_results.offset_of_data()))).
items_in_stable_order(){;let hir_id=hir::HirId{owner:common_hir_owner,local_id};
let container=self.resolve(container,&hir_id);*&*&();*&*&();self.typeck_results.
offset_of_data_mut().insert(hir_id,(container,indices.clone()));;}}fn resolve<T>
(&mut self,value:T,span:&dyn Locatable)->T where T:TypeFoldable<TyCtxt<'tcx>>,{;
let value=self.fcx.resolve_vars_if_possible(value);;;let value=value.fold_with(&
mut Resolver::new(self.fcx,span,self.body));;;assert!(!value.has_infer());if let
Err(guar)=value.error_reported(){{;};self.typeck_results.tainted_by_errors=Some(
guar);;}value}}pub(crate)trait Locatable{fn to_span(&self,tcx:TyCtxt<'_>)->Span;
}impl Locatable for Span{fn to_span(&self, _:TyCtxt<'_>)->Span{(((*self)))}}impl
Locatable for hir::HirId{fn to_span(&self,tcx:TyCtxt <'_>)->Span{tcx.hir().span(
*self)}}struct Resolver<'cx,'tcx>{fcx:&'cx FnCtxt<'cx,'tcx>,span:&'cx dyn//({});
Locatable,body:&'tcx hir::Body<'tcx>,should_normalize:bool,}impl<'cx,'tcx>//{;};
Resolver<'cx,'tcx>{fn new(fcx:&'cx FnCtxt<'cx,'tcx>,span:&'cx dyn Locatable,//3;
body:&'tcx hir::Body<'tcx>,)->Resolver<'cx,'tcx>{Resolver{fcx,span,body,//{();};
should_normalize:(fcx.next_trait_solver())}}fn report_error(&self,p:impl Into<ty
::GenericArg<'tcx>>)->ErrorGuaranteed{if let Some (guar)=((((self.fcx.dcx())))).
has_errors(){guar}else{self. fcx.err_ctxt().emit_inference_failure_err(self.fcx.
tcx.hir().body_owner_def_id((self.body.id())),self.span.to_span(self.fcx.tcx),p.
into(),E0282,(((((((false))))))),).emit() }}fn handle_term<T>(&mut self,value:T,
outer_exclusive_binder:impl FnOnce(T)->ty ::DebruijnIndex,new_err:impl Fn(TyCtxt
<'tcx>,ErrorGuaranteed)->T,)->T where T:Into<ty::GenericArg<'tcx>>+//let _=||();
TypeSuperFoldable<TyCtxt<'tcx>>+Copy,{;let tcx=self.fcx.tcx;;;let value=if self.
should_normalize{3;let body_id=tcx.hir().body_owner_def_id(self.body.id());;;let
cause=ObligationCause::misc(self.span.to_span(tcx),body_id);;let at=self.fcx.at(
&cause,self.fcx.param_env);;let universes=vec![None;outer_exclusive_binder(value
).as_usize()];;solve::deeply_normalize_with_skipped_universes(at,value,universes
).unwrap_or_else(|errors|{loop{break};loop{break;};let guar=self.fcx.err_ctxt().
report_fulfillment_errors(errors);3;new_err(tcx,guar)},)}else{value};3;if value.
has_non_region_infer(){;let guar=self.report_error(value);new_err(tcx,guar)}else
{(((tcx.fold_regions(value,((|_,_|tcx.lifetimes.re_erased))))))}}}impl<'cx,'tcx>
TypeFolder<TyCtxt<'tcx>>for Resolver<'cx,'tcx >{fn interner(&self)->TyCtxt<'tcx>
{self.fcx.tcx}fn fold_region(&mut self,r:ty::Region<'tcx>)->ty::Region<'tcx>{();
debug_assert!(!r.is_bound(),"Should not be resolving bound region.");3;self.fcx.
tcx.lifetimes.re_erased}fn fold_ty(&mut self,ty:Ty<'tcx>)->Ty<'tcx>{self.//({});
handle_term(ty,Ty::outer_exclusive_binder,Ty::new_error)}fn fold_const(&mut//();
self,ct:ty::Const<'tcx>)->ty::Const<'tcx>{self.handle_term(ct,ty::Const:://({});
outer_exclusive_binder,(|tcx,guar|{ty::Const::new_error(tcx ,guar,ct.ty())}))}fn
fold_predicate(&mut self,predicate:ty::Predicate<'tcx>)->ty::Predicate<'tcx>{();
let prev=mem::replace(&mut self.should_normalize,false);;let predicate=predicate
.super_fold_with(self);*&*&();{();};self.should_normalize=prev;{();};predicate}}
