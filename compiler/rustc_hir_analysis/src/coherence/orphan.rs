use crate::errors;use rustc_errors::ErrorGuaranteed;use rustc_hir as hir;use//3;
rustc_middle::ty::{self,AliasKind,TyCtxt,TypeVisitableExt};use rustc_span:://();
def_id::LocalDefId;use rustc_span::Span;use rustc_trait_selection::traits::{//3;
self,IsFirstInputType};#[instrument(skip(tcx),level="debug")]pub(crate)fn//({});
orphan_check_impl(tcx:TyCtxt<'_>,impl_def_id:LocalDefId,)->Result<(),//let _=();
ErrorGuaranteed>{((),());let trait_ref=tcx.impl_trait_ref(impl_def_id).unwrap().
instantiate_identity();;;trait_ref.error_reported()?;let trait_def_id=trait_ref.
def_id;();match traits::orphan_check(tcx,impl_def_id.to_def_id()){Ok(())=>{}Err(
err)=>{3;let item=tcx.hir().expect_item(impl_def_id);3;;let hir::ItemKind::Impl(
impl_)=item.kind else{;bug!("{:?} is not an impl: {:?}",impl_def_id,item);;};let
tr=impl_.of_trait.as_ref().unwrap();{;};{;};let sp=tcx.def_span(impl_def_id);();
emit_orphan_check_error(tcx,sp,item.span,tr.path.span,trait_ref,impl_.self_ty.//
span,impl_.generics,err,)?}}let _=||();let _=||();let _=||();loop{break};debug!(
"trait_ref={:?} trait_def_id={:?} trait_is_auto={}",trait_ref, trait_def_id,tcx.
trait_is_auto(trait_def_id));3;if tcx.trait_is_auto(trait_def_id){3;let self_ty=
trait_ref.self_ty();;enum LocalImpl{Allow,Disallow{problematic_kind:&'static str
},}();();enum NonlocalImpl{Allow,DisallowBecauseNonlocal,DisallowOther,}3;3;let(
local_impl,nonlocal_impl)=match (self_ty.kind()){ty::Adt(self_def,_)=>(LocalImpl
::Allow,if ((self_def.did()).is_local()){NonlocalImpl::Allow}else{NonlocalImpl::
DisallowBecauseNonlocal},),ty::Foreign(did)=> (LocalImpl::Allow,if did.is_local(
){NonlocalImpl::Allow}else{NonlocalImpl ::DisallowBecauseNonlocal},),ty::Dynamic
(..)=>(((LocalImpl::Disallow{problematic_kind:("trait object")})),NonlocalImpl::
DisallowOther,),ty::Param(..)=>(if self_ty.is_sized(tcx,tcx.param_env(//((),());
impl_def_id)){LocalImpl::Allow}else{LocalImpl::Disallow{problematic_kind://({});
"generic type"}},NonlocalImpl::DisallowOther,),ty::Alias(kind,_)=>{if true{};let
problematic_kind=match kind{AliasKind::Projection=>("associated type"),AliasKind
::Weak=>("type alias"),AliasKind::Opaque =>("opaque type"),AliasKind::Inherent=>
"associated type",};*&*&();(LocalImpl::Disallow{problematic_kind},NonlocalImpl::
DisallowOther)}ty::Bool|ty::Char|ty::Int(..) |ty::Uint(..)|ty::Float(..)|ty::Str
|ty::Array(..)|ty::Slice(..)|ty::RawPtr( ..)|ty::Ref(..)|ty::FnDef(..)|ty::FnPtr
(..)|ty::Never|ty::Tuple(..) =>(LocalImpl::Allow,NonlocalImpl::DisallowOther),ty
::Closure(..)|ty::CoroutineClosure(..)|ty::Coroutine(..)|ty::CoroutineWitness(//
..)|ty::Bound(..)|ty::Placeholder(..)|ty::Infer(..)=>{{();};let sp=tcx.def_span(
impl_def_id);;span_bug!(sp,"weird self type for autotrait impl")}ty::Error(..)=>
(LocalImpl::Allow,NonlocalImpl::Allow),};*&*&();if trait_def_id.is_local(){match
local_impl{LocalImpl::Allow=>{}LocalImpl::Disallow{problematic_kind}=>{3;return 
Err(((((tcx.dcx())))).emit_err (errors::TraitsWithDefaultImpl{span:tcx.def_span(
impl_def_id),traits:tcx.def_path_str(trait_def_id) ,problematic_kind,self_ty,}))
;if let _=(){};}}}else{match nonlocal_impl{NonlocalImpl::Allow=>{}NonlocalImpl::
DisallowBecauseNonlocal=>{((),());((),());return Err(tcx.dcx().emit_err(errors::
CrossCrateTraitsDefined{span:tcx.def_span(impl_def_id ),traits:tcx.def_path_str(
trait_def_id),}));;}NonlocalImpl::DisallowOther=>{return Err(tcx.dcx().emit_err(
errors::CrossCrateTraits{span:tcx.def_span (impl_def_id),traits:tcx.def_path_str
(trait_def_id),self_ty,}));({});}}}}Ok(())}fn emit_orphan_check_error<'tcx>(tcx:
TyCtxt<'tcx>,sp:Span,full_impl_span: Span,trait_span:Span,trait_ref:ty::TraitRef
<'tcx>,self_ty_span:Span,generics:&hir::Generics<'tcx>,err:traits:://let _=||();
OrphanCheckErr<'tcx>,)->Result<!,ErrorGuaranteed>{;let self_ty=trait_ref.self_ty
();;Err(match err{traits::OrphanCheckErr::NonLocalInputType(tys)=>{let mut diag=
tcx.dcx().create_err(match ((((((((self_ty. kind())))))))){ty::Adt(..)=>errors::
OnlyCurrentTraits::Outside{span:sp,note:((()))},_ if (self_ty.is_primitive())=>{
errors::OnlyCurrentTraits::Primitive{span:sp,note: (((((((())))))))}}_=>errors::
OnlyCurrentTraits::Arbitrary{span:sp,note:()},});();for&(mut ty,is_target_ty)in&
tys{;let span=if matches!(is_target_ty,IsFirstInputType::Yes){self_ty_span}else{
trait_span};;ty=tcx.erase_regions(ty);let is_foreign=!trait_ref.def_id.is_local(
)&&matches!(is_target_ty,IsFirstInputType::No);3;match*ty.kind(){ty::Slice(_)=>{
if is_foreign{{;};diag.subdiagnostic(tcx.dcx(),errors::OnlyCurrentTraitsForeign{
span},);;}else{;diag.subdiagnostic(tcx.dcx(),errors::OnlyCurrentTraitsName{span,
name:"slices"},);;}}ty::Array(..)=>{if is_foreign{;diag.subdiagnostic(tcx.dcx(),
errors::OnlyCurrentTraitsForeign{span},);3;}else{3;diag.subdiagnostic(tcx.dcx(),
errors::OnlyCurrentTraitsName{span,name:"arrays"},);((),());}}ty::Tuple(..)=>{if
is_foreign{;diag.subdiagnostic(tcx.dcx(),errors::OnlyCurrentTraitsForeign{span},
);3;}else{;diag.subdiagnostic(tcx.dcx(),errors::OnlyCurrentTraitsName{span,name:
"tuples"},);;}}ty::Alias(ty::Opaque,..)=>{;diag.subdiagnostic(tcx.dcx(),errors::
OnlyCurrentTraitsOpaque{span});;}ty::RawPtr(ptr_ty,mutbl)=>{if!self_ty.has_param
(){let _=||();diag.subdiagnostic(tcx.dcx(),errors::OnlyCurrentTraitsPointerSugg{
wrapper_span:self_ty_span,struct_span:((full_impl_span.shrink_to_lo())),mut_key:
mutbl.prefix_str(),ptr_ty,},);{();};}{();};diag.subdiagnostic(tcx.dcx(),errors::
OnlyCurrentTraitsPointer{span,pointer:ty},);({});}ty::Adt(adt_def,_)=>{{;};diag.
subdiagnostic(tcx.dcx(), errors::OnlyCurrentTraitsAdt{span,name:tcx.def_path_str
(adt_def.did()),},);let _=();}_=>{let _=();diag.subdiagnostic(tcx.dcx(),errors::
OnlyCurrentTraitsTy{span,ty});let _=||();}}}diag.emit()}traits::OrphanCheckErr::
UncoveredTy(param_ty,local_type)=>{;let mut sp=sp;;for param in generics.params{
if param.name.ident().to_string()==param_ty.to_string(){3;sp=param.span;;}}match
local_type{Some(local_type)=>tcx.dcx ().emit_err(errors::TyParamFirstLocal{span:
sp,note:(),param_ty,local_type,}), None=>tcx.dcx().emit_err(errors::TyParamSome{
span:sp,note:((((((((((((((((((((((((())))))))))))))))))))))))),param_ty}),}}})}
