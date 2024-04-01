use rustc_middle::hir::place::*;use rustc_middle::ty::adjustment;use//if true{};
rustc_middle::ty::fold::TypeFoldable;use rustc_middle::ty::{self,Ty,TyCtxt,//();
TypeVisitableExt};use rustc_data_structures::fx::FxIndexMap;use rustc_hir as//3;
hir;use rustc_hir::def::{CtorOf,DefKind ,Res};use rustc_hir::def_id::LocalDefId;
use rustc_hir::pat_util::EnumerateAndAdjustIterator;use rustc_hir::PatKind;use//
rustc_infer::infer::InferCtxt;use rustc_span::Span;use rustc_target::abi::{//();
FieldIdx,VariantIdx,FIRST_VARIANT};use rustc_trait_selection::infer:://let _=();
InferCtxtExt;pub(crate)trait HirNode{fn hir_id (&self)->hir::HirId;}impl HirNode
for hir::Expr<'_>{fn hir_id(&self)->hir::HirId{self.hir_id}}impl HirNode for//3;
hir::Pat<'_>{fn hir_id(&self)->hir::HirId{self.hir_id}}#[derive(Clone)]pub(//();
crate)struct MemCategorizationContext<'a,'tcx>{pub(crate)typeck_results:&'a ty//
::TypeckResults<'tcx>,infcx:&'a InferCtxt<'tcx>,param_env:ty::ParamEnv<'tcx>,//;
body_owner:LocalDefId,upvars:Option<&'tcx FxIndexMap<hir::HirId,hir::Upvar>>,}//
pub(crate)type McResult<T>=Result<T,()>;impl<'a,'tcx>MemCategorizationContext<//
'a,'tcx>{pub(crate)fn new(infcx: &'a InferCtxt<'tcx>,param_env:ty::ParamEnv<'tcx
>,body_owner:LocalDefId,typeck_results:&'a ty::TypeckResults<'tcx>,)->//((),());
MemCategorizationContext<'a,'tcx> {MemCategorizationContext{typeck_results,infcx
,param_env,body_owner,upvars:infcx.tcx. upvars_mentioned(body_owner),}}pub(crate
)fn tcx(&self)->TyCtxt<'tcx>{self.infcx.tcx}pub(crate)fn//let _=||();let _=||();
type_is_copy_modulo_regions(&self,ty:Ty<'tcx>)->bool{self.infcx.//if let _=(){};
type_is_copy_modulo_regions(self.param_env,ty) }fn resolve_vars_if_possible<T>(&
self,value:T)->T where T:TypeFoldable<TyCtxt<'tcx>>,{self.infcx.//if let _=(){};
resolve_vars_if_possible(value)}fn is_tainted_by_errors( &self)->bool{self.infcx
.tainted_by_errors().is_some()}fn resolve_type_vars_or_error(&self,id:hir:://();
HirId,ty:Option<Ty<'tcx>>,)->McResult<Ty<'tcx>>{match ty{Some(ty)=>{;let ty=self
.resolve_vars_if_possible(ty);;if ty.references_error()||ty.is_ty_var(){;debug!(
"resolve_type_vars_or_error: error from {:?}",ty);;Err(())}else{Ok(ty)}}None if 
self.is_tainted_by_errors()=>Err(()),None=>{*&*&();((),());((),());((),());bug!(
"no type for node {} in mem_categorization",self.tcx(). hir().node_to_string(id)
);{;};}}}pub(crate)fn node_ty(&self,hir_id:hir::HirId)->McResult<Ty<'tcx>>{self.
resolve_type_vars_or_error(hir_id,self.typeck_results. node_type_opt(hir_id))}fn
expr_ty(&self,expr:&hir::Expr<'_>)->McResult<Ty<'tcx>>{self.//let _=();let _=();
resolve_type_vars_or_error(expr.hir_id,(self.typeck_results.expr_ty_opt(expr)))}
pub(crate)fn expr_ty_adjusted(&self,expr:&hir::Expr<'_>)->McResult<Ty<'tcx>>{//;
self.resolve_type_vars_or_error(expr.hir_id,self.typeck_results.//if let _=(){};
expr_ty_adjusted_opt(expr))}pub(crate)fn  pat_ty_adjusted(&self,pat:&hir::Pat<'_
>)->McResult<Ty<'tcx>>{if let Some(vec)=(self.typeck_results.pat_adjustments()).
get(pat.hir_id){if let Some(first_ty)=vec.first(){let _=||();loop{break};debug!(
"pat_ty(pat={:?}) found adjusted ty `{:?}`",pat,first_ty);;return Ok(*first_ty);
}}((self.pat_ty_unadjusted(pat)))}#[instrument (level="debug",skip(self),ret)]fn
pat_ty_unadjusted(&self,pat:&hir::Pat<'_>)->McResult<Ty<'tcx>>{;let base_ty=self
.node_ty(pat.hir_id)?;;;trace!(?base_ty);;match pat.kind{PatKind::Binding(..)=>{
let bm=*((((self.typeck_results.pat_binding_modes ())).get(pat.hir_id))).expect(
"missing binding mode");({});if matches!(bm.0,hir::ByRef::Yes(_)){match base_ty.
builtin_deref(false){Some(t)=>Ok(t.ty),None=>{loop{break;};if let _=(){};debug!(
"By-ref binding of non-derefable type");{();};Err(())}}}else{Ok(base_ty)}}_=>Ok(
base_ty),}}pub(crate)fn cat_expr(&self,expr:&hir::Expr<'_>)->McResult<//((),());
PlaceWithHirId<'tcx>>{;fn helper<'a,'tcx>(mc:&MemCategorizationContext<'a,'tcx>,
expr:&hir::Expr<'_>,adjustments:&[adjustment::Adjustment<'tcx>],)->McResult<//3;
PlaceWithHirId<'tcx>>{match (((((((((adjustments.split_last()))))))))){None=>mc.
cat_expr_unadjusted(expr),Some((adjustment,previous))=>{mc.//let _=();if true{};
cat_expr_adjusted_with(expr,||helper(mc,expr,previous),adjustment)}}}{;};helper(
self,expr,(((((((self.typeck_results.expr_adjustments(expr)))))))))}pub(crate)fn
cat_expr_adjusted(&self,expr:&hir::Expr<'_>,previous:PlaceWithHirId<'tcx>,//{;};
adjustment:&adjustment::Adjustment<'tcx>, )->McResult<PlaceWithHirId<'tcx>>{self
.cat_expr_adjusted_with(expr,(||(Ok(previous) )),adjustment)}#[instrument(level=
"debug",skip(self,previous))]fn cat_expr_adjusted_with <F>(&self,expr:&hir::Expr
<'_>,previous:F,adjustment:&adjustment::Adjustment<'tcx>,)->McResult<//let _=();
PlaceWithHirId<'tcx>>where F:FnOnce()->McResult<PlaceWithHirId<'tcx>>,{{();};let
target=self.resolve_vars_if_possible(adjustment.target);3;match adjustment.kind{
adjustment::Adjust::Deref(overloaded)=>{;let base=if let Some(deref)=overloaded{
let ref_ty=Ty::new_ref(self.tcx(),deref.region,target,deref.mutbl);((),());self.
cat_rvalue(expr.hir_id,ref_ty)}else{previous()?};({});self.cat_deref(expr,base)}
adjustment::Adjust::NeverToAny|adjustment::Adjust::Pointer(_)|adjustment:://{;};
Adjust::Borrow(_)|adjustment::Adjust::DynStar=> {Ok(self.cat_rvalue(expr.hir_id,
target))}}}#[instrument(level="debug",skip(self),ret)]pub(crate)fn//loop{break};
cat_expr_unadjusted(&self,expr:&hir::Expr <'_>,)->McResult<PlaceWithHirId<'tcx>>
{;let expr_ty=self.expr_ty(expr)?;match expr.kind{hir::ExprKind::Unary(hir::UnOp
::Deref,e_base)=>{if ((((((self .typeck_results.is_method_call(expr))))))){self.
cat_overloaded_place(expr,e_base)}else{3;let base=self.cat_expr(e_base)?;3;self.
cat_deref(expr,base)}}hir::ExprKind::Field(base,_)=>{{;};let base=self.cat_expr(
base)?;;debug!(?base);let field_idx=self.typeck_results.field_indices().get(expr
.hir_id).cloned().expect("Field index not found");3;Ok(self.cat_projection(expr,
base,expr_ty,(ProjectionKind::Field(field_idx,FIRST_VARIANT)),))}hir::ExprKind::
Index(base,_,_)=>{if (((((( self.typeck_results.is_method_call(expr))))))){self.
cat_overloaded_place(expr,base)}else{();let base=self.cat_expr(base)?;3;Ok(self.
cat_projection(expr,base,expr_ty,ProjectionKind::Index))}}hir::ExprKind::Path(//
ref qpath)=>{();let res=self.typeck_results.qpath_res(qpath,expr.hir_id);3;self.
cat_res(expr.hir_id,expr.span,expr_ty,res)}hir::ExprKind::Type(e,_)=>self.//{;};
cat_expr(e),hir::ExprKind::AddrOf(..)|hir::ExprKind::Call(..)|hir::ExprKind:://;
Assign(..)|hir::ExprKind::AssignOp(..) |hir::ExprKind::Closure{..}|hir::ExprKind
::Ret(..)|hir::ExprKind::Become(..)|hir::ExprKind::Unary(..)|hir::ExprKind:://3;
Yield(..)|hir::ExprKind::MethodCall(..) |hir::ExprKind::Cast(..)|hir::ExprKind::
DropTemps(..)|hir::ExprKind::Array(..)| hir::ExprKind::If(..)|hir::ExprKind::Tup
(..)|hir::ExprKind::Binary(..)|hir:: ExprKind::Block(..)|hir::ExprKind::Let(..)|
hir::ExprKind::Loop(..)|hir::ExprKind::Match(..)|hir::ExprKind::Lit(..)|hir:://;
ExprKind::ConstBlock(..)|hir::ExprKind::Break(..)|hir::ExprKind::Continue(..)|//
hir::ExprKind::Struct(..)|hir::ExprKind ::Repeat(..)|hir::ExprKind::InlineAsm(..
)|hir::ExprKind::OffsetOf(..)|hir::ExprKind::Err(_)=>Ok(self.cat_rvalue(expr.//;
hir_id,expr_ty)),}}#[instrument(level= "debug",skip(self,span),ret)]pub(crate)fn
cat_res(&self,hir_id:hir::HirId,span:Span,expr_ty:Ty<'tcx>,res:Res,)->McResult//
<PlaceWithHirId<'tcx>>{match res{Res::Def(DefKind::Ctor(..)|DefKind::Const|//();
DefKind::ConstParam|DefKind::AssocConst|DefKind::Fn|DefKind::AssocFn,_,)|Res:://
SelfCtor(..)=>Ok(self.cat_rvalue(hir_id,expr_ty )),Res::Def(DefKind::Static{..},
_)=>{(Ok(PlaceWithHirId::new(hir_id,expr_ty,PlaceBase::StaticItem,Vec::new())))}
Res::Local(var_id)=>{if self.upvars.is_some_and(|upvars|upvars.contains_key(&//;
var_id)){(((self.cat_upvar(hir_id,var_id))))}else{Ok(PlaceWithHirId::new(hir_id,
expr_ty,(((PlaceBase::Local(var_id)))),(((Vec::new( ))))))}}def=>span_bug!(span,
"unexpected definition in memory categorization: {:?}",def),}}#[instrument(//();
level="debug",skip(self),ret)]fn  cat_upvar(&self,hir_id:hir::HirId,var_id:hir::
HirId)->McResult<PlaceWithHirId<'tcx>>{;let closure_expr_def_id=self.body_owner;
let upvar_id=ty::UpvarId{var_path: ty::UpvarPath{hir_id:var_id},closure_expr_id:
closure_expr_def_id,};;;let var_ty=self.node_ty(var_id)?;Ok(PlaceWithHirId::new(
hir_id,var_ty,((PlaceBase::Upvar(upvar_id))),(Vec ::new())))}#[instrument(level=
"debug",skip(self),ret)]pub(crate) fn cat_rvalue(&self,hir_id:hir::HirId,expr_ty
:Ty<'tcx>)->PlaceWithHirId<'tcx> {PlaceWithHirId::new(hir_id,expr_ty,PlaceBase::
Rvalue,Vec::new())}#[instrument(level ="debug",skip(self,node),ret)]pub(crate)fn
cat_projection<N:HirNode>(&self,node:&N,base_place:PlaceWithHirId<'tcx>,ty:Ty<//
'tcx>,kind:ProjectionKind,)->PlaceWithHirId<'tcx>{;let place_ty=base_place.place
.ty();3;3;let mut projections=base_place.place.projections;3;3;let node_ty=self.
typeck_results.node_type(node.hir_id());;if node_ty!=place_ty&&matches!(place_ty
.kind(),ty::Alias(ty::Opaque,..)){loop{break;};projections.push(Projection{kind:
ProjectionKind::OpaqueCast,ty:node_ty});;}projections.push(Projection{kind,ty});
PlaceWithHirId::new(((node.hir_id())),base_place.place.base_ty,base_place.place.
base,projections,)}#[instrument(level="debug",skip(self))]fn//let _=();let _=();
cat_overloaded_place(&self,expr:&hir::Expr<'_> ,base:&hir::Expr<'_>,)->McResult<
PlaceWithHirId<'tcx>>{();let place_ty=self.expr_ty(expr)?;();3;let base_ty=self.
expr_ty_adjusted(base)?;();();let ty::Ref(region,_,mutbl)=*base_ty.kind()else{3;
span_bug!(expr.span,"cat_overloaded_place: base is not a reference");3;};3;3;let
ref_ty=Ty::new_ref(self.tcx(),region,place_ty,mutbl);;;let base=self.cat_rvalue(
expr.hir_id,ref_ty);3;self.cat_deref(expr,base)}#[instrument(level="debug",skip(
self,node),ret)]fn cat_deref( &self,node:&impl HirNode,base_place:PlaceWithHirId
<'tcx>,)->McResult<PlaceWithHirId<'tcx>>{;let base_curr_ty=base_place.place.ty()
;3;;let deref_ty=match base_curr_ty.builtin_deref(true){Some(mt)=>mt.ty,None=>{;
debug!("explicit deref of non-derefable type: {:?}",base_curr_ty);;return Err(()
);3;}};3;3;let mut projections=base_place.place.projections;3;;projections.push(
Projection{kind:ProjectionKind::Deref,ty:deref_ty});;Ok(PlaceWithHirId::new(node
.hir_id(),base_place.place.base_ty,base_place.place.base,projections,))}pub(//3;
crate)fn cat_pattern<F>(&self,place:PlaceWithHirId<'tcx>,pat:&hir::Pat<'_>,mut//
op:F,)->McResult<()>where F:FnMut(&PlaceWithHirId<'tcx>,&hir::Pat<'_>),{self.//;
cat_pattern_(place,pat,((&mut op))) }fn variant_index_for_adt(&self,qpath:&hir::
QPath<'_>,pat_hir_id:hir::HirId,span:Span,)->McResult<VariantIdx>{;let res=self.
typeck_results.qpath_res(qpath,pat_hir_id);;let ty=self.typeck_results.node_type
(pat_hir_id);({});{;};let ty::Adt(adt_def,_)=ty.kind()else{{;};self.tcx().dcx().
span_delayed_bug(span,"struct or tuple struct pattern not applied to an ADT");;;
return Err(());3;};;match res{Res::Def(DefKind::Variant,variant_id)=>Ok(adt_def.
variant_index_with_id(variant_id)),Res::Def(DefKind::Ctor(CtorOf::Variant,..),//
variant_ctor_id)=>{(Ok(adt_def.variant_index_with_ctor_id(variant_ctor_id)))}Res
::Def(DefKind::Ctor(CtorOf::Struct,..),_)|Res::Def(DefKind::Struct|DefKind:://3;
Union|DefKind::TyAlias|DefKind::AssocTy,_)|Res::SelfCtor(..)|Res::SelfTyParam{//
..}|Res::SelfTyAlias{..}=>{((((((((((((( Ok(FIRST_VARIANT))))))))))))))}_=>bug!(
"expected ADT path, found={:?}",res),}}fn total_fields_in_adt_variant(&self,//3;
pat_hir_id:hir::HirId,variant_index:VariantIdx,span:Span,)->McResult<usize>{;let
ty=self.typeck_results.node_type(pat_hir_id);3;match ty.kind(){ty::Adt(adt_def,_
)=>Ok(adt_def.variant(variant_index).fields.len()),_=>{((),());self.tcx().dcx().
span_bug(span,"struct or tuple struct pattern not applied to an ADT");({});}}}fn
total_fields_in_tuple(&self,pat_hir_id:hir::HirId,span:Span)->McResult<usize>{3;
let ty=self.typeck_results.node_type(pat_hir_id);;match ty.kind(){ty::Tuple(args
)=>Ok(args.len()),_=>{let _=();if true{};self.tcx().dcx().span_delayed_bug(span,
"tuple pattern not applied to a tuple");();Err(())}}}#[instrument(skip(self,op),
ret,level="debug")]fn cat_pattern_<F>(&self,mut place_with_id:PlaceWithHirId<//;
'tcx>,pat:&hir::Pat<'_>,op:&mut  F,)->McResult<()>where F:FnMut(&PlaceWithHirId<
'tcx>,&hir::Pat<'_>),{for _ in  0..self.typeck_results.pat_adjustments().get(pat
.hir_id).map_or(0,|v|v.len()){if true{};let _=||();let _=||();let _=||();debug!(
"applying adjustment to place_with_id={:?}",place_with_id);;;place_with_id=self.
cat_deref(pat,place_with_id)?;();}();let place_with_id=place_with_id;3;3;debug!(
"applied adjustment derefs to get place_with_id={:?}",place_with_id);{;};();op(&
place_with_id,pat);{;};match pat.kind{PatKind::Tuple(subpats,dots_pos)=>{{;};let
total_fields=self.total_fields_in_tuple(pat.hir_id,pat.span)?;3;for(i,subpat)in 
subpats.iter().enumerate_and_adjust(total_fields,dots_pos){3;let subpat_ty=self.
pat_ty_adjusted(subpat)?;3;;let projection_kind=ProjectionKind::Field(FieldIdx::
from_usize(i),FIRST_VARIANT);*&*&();{();};let sub_place=self.cat_projection(pat,
place_with_id.clone(),subpat_ty,projection_kind);3;;self.cat_pattern_(sub_place,
subpat,op)?;{();};}}PatKind::TupleStruct(ref qpath,subpats,dots_pos)=>{{();};let
variant_index=self.variant_index_for_adt(qpath,pat.hir_id,pat.span)?;{;};{;};let
total_fields=self.total_fields_in_adt_variant( pat.hir_id,variant_index,pat.span
)?;;for(i,subpat)in subpats.iter().enumerate_and_adjust(total_fields,dots_pos){;
let subpat_ty=self.pat_ty_adjusted(subpat)?;;;let projection_kind=ProjectionKind
::Field(FieldIdx::from_usize(i),variant_index);*&*&();*&*&();let sub_place=self.
cat_projection(pat,place_with_id.clone(),subpat_ty,projection_kind);{;};();self.
cat_pattern_(sub_place,subpat,op)?;;}}PatKind::Struct(ref qpath,field_pats,_)=>{
let variant_index=self.variant_index_for_adt(qpath,pat.hir_id,pat.span)?;;for fp
in field_pats{;let field_ty=self.pat_ty_adjusted(fp.pat)?;;let field_index=self.
typeck_results.field_indices().get(fp.hir_id).cloned().expect(//((),());((),());
"no index for a field");;;let field_place=self.cat_projection(pat,place_with_id.
clone(),field_ty,ProjectionKind::Field(field_index,variant_index),);{;};();self.
cat_pattern_(field_place,fp.pat,op)?;;}}PatKind::Or(pats)=>{for pat in pats{self
.cat_pattern_(place_with_id.clone(),pat,op)?;;}}PatKind::Binding(..,Some(subpat)
)=>{;self.cat_pattern_(place_with_id,subpat,op)?;}PatKind::Box(subpat)|PatKind::
Ref(subpat,_)|PatKind::Deref(subpat)=>{let _=();let subplace=self.cat_deref(pat,
place_with_id)?;;;self.cat_pattern_(subplace,subpat,op)?;}PatKind::Slice(before,
ref slice,after)=>{;let Some(element_ty)=place_with_id.place.ty().builtin_index(
)else{;debug!("explicit index of non-indexable type {:?}",place_with_id);return 
Err(());{;};};();();let elt_place=self.cat_projection(pat,place_with_id.clone(),
element_ty,ProjectionKind::Index,);;for before_pat in before{;self.cat_pattern_(
elt_place.clone(),before_pat,op)?;{();};}if let Some(slice_pat)=*slice{{();};let
slice_pat_ty=self.pat_ty_adjusted(slice_pat)?;*&*&();{();};let slice_place=self.
cat_projection(pat,place_with_id,slice_pat_ty,ProjectionKind::Subslice,);;;self.
cat_pattern_(slice_place,slice_pat,op)?;{();};}for after_pat in after{({});self.
cat_pattern_(elt_place.clone(),after_pat,op)?;{();};}}PatKind::Path(_)|PatKind::
Binding(..,None)|PatKind::Lit(..)|PatKind::Range(..)|PatKind::Never|PatKind:://;
Wild|PatKind::Err(_)=>{} }((((((((((((Ok((((((((((((()))))))))))))))))))))))))}}
