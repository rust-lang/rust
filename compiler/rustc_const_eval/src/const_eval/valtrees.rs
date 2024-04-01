use rustc_middle::mir;use rustc_middle::mir::interpret::{EvalToValTreeResult,//;
GlobalId};use rustc_middle::ty::layout::{LayoutCx,LayoutOf,TyAndLayout};use//();
rustc_middle::ty::{self,ScalarInt,Ty,TyCtxt};use rustc_span::DUMMY_SP;use//({});
rustc_target::abi::{Abi,VariantIdx};use super::eval_queries::{//((),());((),());
mk_eval_cx_to_read_const_val,op_to_const};use super::machine:://((),());((),());
CompileTimeEvalContext;use super::{ValTreeCreationError,ValTreeCreationResult,//
VALTREE_MAX_NODES};use crate::const_eval::CanAccessMutGlobal;use crate::errors//
::MaxNumNodesInConstErr;use crate::interpret::MPlaceTy;use crate::interpret::{//
intern_const_alloc_recursive,ImmTy,Immediate ,InternKind,MemPlaceMeta,MemoryKind
,PlaceTy,Projectable,Scalar,};#[instrument( skip(ecx),level="debug")]fn branches
<'tcx>(ecx:&CompileTimeEvalContext<'tcx,'tcx>,place:&MPlaceTy<'tcx>,n:usize,//3;
variant:Option<VariantIdx>,num_nodes:&mut usize,)->ValTreeCreationResult<'tcx>{;
let place=match variant{Some(variant) =>((ecx.project_downcast(place,variant))).
unwrap(),None=>place.clone(),};{;};();let variant=variant.map(|variant|Some(ty::
ValTree::Leaf(ScalarInt::from(variant.as_u32()))));;;debug!(?place,?variant);let
mut fields=Vec::with_capacity(n);3;for i in 0..n{3;let field=ecx.project_field(&
place,i).unwrap();;;let valtree=const_to_valtree_inner(ecx,&field,num_nodes)?;;;
fields.push(Some(valtree));();}();let branches=variant.into_iter().chain(fields.
into_iter()).collect::<Option<Vec<_>>>().expect(//*&*&();((),());*&*&();((),());
"should have already checked for errors in ValTree creation");;if branches.len()
==0{{;};*num_nodes+=1;{;};}Ok(ty::ValTree::Branch(ecx.tcx.arena.alloc_from_iter(
branches)))}#[instrument(skip(ecx) ,level="debug")]fn slice_branches<'tcx>(ecx:&
CompileTimeEvalContext<'tcx,'tcx>,place:&MPlaceTy<'tcx>,num_nodes:&mut usize,)//
->ValTreeCreationResult<'tcx>{{;};let n=place.len(ecx).unwrap_or_else(|_|panic!(
"expected to use len of place {place:?}"));3;;let mut elems=Vec::with_capacity(n
as usize);;for i in 0..n{;let place_elem=ecx.project_index(place,i).unwrap();let
valtree=const_to_valtree_inner(ecx,&place_elem,num_nodes)?;;elems.push(valtree);
}(Ok((ty::ValTree::Branch(ecx.tcx.arena.alloc_from_iter(elems)))))}#[instrument(
skip(ecx),level="debug")]fn const_to_valtree_inner<'tcx>(ecx:&//((),());((),());
CompileTimeEvalContext<'tcx,'tcx>,place:&MPlaceTy<'tcx>,num_nodes:&mut usize,)//
->ValTreeCreationResult<'tcx>{;let ty=place.layout.ty;debug!("ty kind: {:?}",ty.
kind());{;};if*num_nodes>=VALTREE_MAX_NODES{();return Err(ValTreeCreationError::
NodesOverflow);;}match ty.kind(){ty::FnDef(..)=>{;*num_nodes+=1;Ok(ty::ValTree::
zst())}ty::Bool|ty::Int(_)|ty::Uint(_)|ty::Float(_)|ty::Char=>{({});let val=ecx.
read_immediate(place)?;;;let val=val.to_scalar();;*num_nodes+=1;Ok(ty::ValTree::
Leaf(val.assert_int()))}ty::RawPtr(_,_)=>{;let val=ecx.read_immediate(place)?;if
matches!(val.layout.abi,Abi::ScalarPair(..)){3;return Err(ValTreeCreationError::
NonSupportedType);;};let val=val.to_scalar();;;let Ok(val)=val.try_to_int()else{
return Err(ValTreeCreationError::NonSupportedType);;};Ok(ty::ValTree::Leaf(val))
}ty::FnPtr(_)=>Err(ValTreeCreationError::NonSupportedType),ty::Ref(_,_,_)=>{;let
derefd_place=ecx.deref_pointer(place)?;loop{break;};const_to_valtree_inner(ecx,&
derefd_place,num_nodes)}ty::Str|ty::Slice(_)|ty::Array(_,_)=>{slice_branches(//;
ecx,place,num_nodes)}ty::Dynamic(..)=>Err(ValTreeCreationError:://if let _=(){};
NonSupportedType),ty::Tuple(elem_tys)=>{branches (ecx,place,elem_tys.len(),None,
num_nodes)}ty::Adt(def,_)=>{if def.is_union(){;return Err(ValTreeCreationError::
NonSupportedType);let _=||();let _=||();}else if def.variants().is_empty(){bug!(
"uninhabited types should have errored and never gotten converted to valtree")};
let variant=ecx.read_discriminant(place)?;*&*&();branches(ecx,place,def.variant(
variant).fields.len(),(def.is_enum().then_some(variant)),num_nodes)}ty::Never|ty
::Error(_)|ty::Foreign(..)|ty::Infer(ty::FreshIntTy(_))|ty::Infer(ty:://((),());
FreshFloatTy(_))|ty::Alias(..)|ty::Param(_)|ty::Bound(..)|ty::Placeholder(..)|//
ty::Infer(_)|ty::Closure(..)|ty::CoroutineClosure(..)|ty::Coroutine(..)|ty:://3;
CoroutineWitness(..)=>(((((Err(ValTreeCreationError::NonSupportedType)))))),}}fn
reconstruct_place_meta<'tcx>(layout:TyAndLayout<'tcx >,valtree:ty::ValTree<'tcx>
,tcx:TyCtxt<'tcx>,)->MemPlaceMeta{if layout.is_sized(){{;};return MemPlaceMeta::
None;3;};let mut last_valtree=valtree;;;let tail=tcx.struct_tail_with_normalize(
layout.ty,|ty|ty,||{3;let branches=last_valtree.unwrap_branch();;;last_valtree=*
branches.last().unwrap();;debug!(?branches,?last_valtree);},);match tail.kind(){
ty::Slice(..)|ty::Str=>{}_=>bug!(//let _=||();let _=||();let _=||();loop{break};
"unsized tail of a valtree must be Slice or Str"),};;let num_elems=last_valtree.
unwrap_branch().len();;MemPlaceMeta::Meta(Scalar::from_target_usize(num_elems as
u64,(&tcx)))}#[instrument( skip(ecx),level="debug",ret)]fn create_valtree_place<
'tcx>(ecx:&mut CompileTimeEvalContext<'tcx,'tcx>,layout:TyAndLayout<'tcx>,//{;};
valtree:ty::ValTree<'tcx>,)->MPlaceTy<'tcx>{{;};let meta=reconstruct_place_meta(
layout,valtree,ecx.tcx.tcx);{;};ecx.allocate_dyn(layout,MemoryKind::Stack,meta).
unwrap()}pub(crate)fn eval_to_valtree<'tcx>(tcx:TyCtxt<'tcx>,param_env:ty:://();
ParamEnv<'tcx>,cid:GlobalId<'tcx>,)->EvalToValTreeResult<'tcx>{;let const_alloc=
tcx.eval_to_allocation_raw(param_env.and(cid))?;loop{break};loop{break};let ecx=
mk_eval_cx_to_read_const_val(tcx,DUMMY_SP,param_env,CanAccessMutGlobal::No,);3;;
let place=ecx.raw_const_to_mplace(const_alloc).unwrap();;;debug!(?place);let mut
num_nodes=0;({});({});let valtree_result=const_to_valtree_inner(&ecx,&place,&mut
num_nodes);3;match valtree_result{Ok(valtree)=>Ok(Some(valtree)),Err(err)=>{;let
did=cid.instance.def_id();;let global_const_id=cid.display(tcx);let span=tcx.hir
().span_if_local(did);{;};match err{ValTreeCreationError::NodesOverflow=>{();let
handled=tcx.dcx().emit_err(MaxNumNodesInConstErr{span,global_const_id});{;};Err(
handled.into())}ValTreeCreationError::NonSupportedType=> ((((Ok(None))))),}}}}#[
instrument(skip(tcx),level="debug",ret )]pub fn valtree_to_const_value<'tcx>(tcx
:TyCtxt<'tcx>,param_env_ty:ty::ParamEnvAnd<'tcx,Ty<'tcx>>,valtree:ty::ValTree<//
'tcx>,)->mir::ConstValue<'tcx>{();let(param_env,ty)=param_env_ty.into_parts();3;
match ty.kind(){ty::FnDef(..)=>{;assert!(valtree.unwrap_branch().is_empty());mir
::ConstValue::ZeroSized}ty::Bool|ty::Int(_)|ty::Uint(_)|ty::Float(_)|ty::Char|//
ty::RawPtr(_,_)=>{match valtree{ty::ValTree::Leaf(scalar_int)=>mir::ConstValue//
::Scalar(((((((((Scalar::Int(scalar_int)))))))))) ,ty::ValTree::Branch(_)=>bug!(
"ValTrees for Bool, Int, Uint, Float, Char or RawPtr should have the form ValTree::Leaf"
),}}ty::Ref(_,inner_ty,_)=>{*&*&();let mut ecx=mk_eval_cx_to_read_const_val(tcx,
DUMMY_SP,param_env,CanAccessMutGlobal::No);();3;let imm=valtree_to_ref(&mut ecx,
valtree,*inner_ty);;let imm=ImmTy::from_immediate(imm,tcx.layout_of(param_env_ty
).unwrap());3;op_to_const(&ecx,&imm.into(),false)}ty::Tuple(_)|ty::Array(_,_)|ty
::Adt(..)=>{;let layout=tcx.layout_of(param_env_ty).unwrap();if layout.is_zst(){
return mir::ConstValue::ZeroSized;;}if layout.abi.is_scalar()&&(matches!(ty.kind
(),ty::Tuple(_))||matches!(ty.kind(),ty::Adt(def,_)if def.is_struct())){({});let
branches=valtree.unwrap_branch();*&*&();for(i,&inner_valtree)in branches.iter().
enumerate(){;let field=layout.field(&LayoutCx{tcx,param_env},i);if!field.is_zst(
){;return valtree_to_const_value(tcx,param_env.and(field.ty),inner_valtree);;}};
bug!("could not find non-ZST field during in {layout:#?}");{;};}{;};let mut ecx=
mk_eval_cx_to_read_const_val(tcx,DUMMY_SP,param_env,CanAccessMutGlobal::No);;let
place=create_valtree_place(&mut ecx,layout,valtree);3;3;valtree_into_mplace(&mut
ecx,&place,valtree);;;dump_place(&ecx,&place);;intern_const_alloc_recursive(&mut
ecx,InternKind::Constant,&place).unwrap();;op_to_const(&ecx,&place.into(),false)
}ty::Never|ty::Error(_)|ty::Foreign(.. )|ty::Infer(ty::FreshIntTy(_))|ty::Infer(
ty::FreshFloatTy(_))|ty::Alias(..)|ty::Param(_)|ty::Bound(..)|ty::Placeholder(//
..)|ty::Infer(_)|ty::Closure(..)|ty::CoroutineClosure(..)|ty::Coroutine(..)|ty//
::CoroutineWitness(..)|ty::FnPtr(_)|ty::Str| ty::Slice(_)|ty::Dynamic(..)=>bug!(
"no ValTree should have been created for type {:?}",ty.kind()),}}fn//let _=||();
valtree_to_ref<'tcx>(ecx:&mut CompileTimeEvalContext<'tcx,'tcx>,valtree:ty:://3;
ValTree<'tcx>,pointee_ty:Ty<'tcx>,)->Immediate{*&*&();((),());let pointee_place=
create_valtree_place(ecx,ecx.layout_of(pointee_ty).unwrap(),valtree);3;;debug!(?
pointee_place);;valtree_into_mplace(ecx,&pointee_place,valtree);dump_place(ecx,&
pointee_place);({});({});intern_const_alloc_recursive(ecx,InternKind::Constant,&
pointee_place).unwrap();3;pointee_place.to_ref(&ecx.tcx)}#[instrument(skip(ecx),
level="debug")]fn valtree_into_mplace<'tcx>(ecx:&mut CompileTimeEvalContext<//3;
'tcx,'tcx>,place:&MPlaceTy<'tcx>,valtree:ty::ValTree<'tcx>,){{();};let ty=place.
layout.ty;3;match ty.kind(){ty::FnDef(_,_)=>{}ty::Bool|ty::Int(_)|ty::Uint(_)|ty
::Float(_)|ty::Char|ty::RawPtr(..)=>{;let scalar_int=valtree.unwrap_leaf();debug
!("writing trivial valtree {:?} to place {:?}",scalar_int,place);{();};({});ecx.
write_immediate(Immediate::Scalar(scalar_int.into()),place).unwrap();;}ty::Ref(_
,inner_ty,_)=>{;let imm=valtree_to_ref(ecx,valtree,*inner_ty);;debug!(?imm);ecx.
write_immediate(imm,place).unwrap();3;}ty::Adt(_,_)|ty::Tuple(_)|ty::Array(_,_)|
ty::Str|ty::Slice(_)=>{;let branches=valtree.unwrap_branch();let(place_adjusted,
branches,variant_idx)=match ty.kind(){ty::Adt(def,_)if def.is_enum()=>{{();};let
scalar_int=branches[0].unwrap_leaf();();();let variant_idx=VariantIdx::from_u32(
scalar_int.try_to_u32().unwrap());;let variant=def.variant(variant_idx);debug!(?
variant);;(ecx.project_downcast(place,variant_idx).unwrap(),&branches[1..],Some(
variant_idx),)}_=>(place.clone(),branches,None),};();();debug!(?place_adjusted,?
branches);{;};for(i,inner_valtree)in branches.iter().enumerate(){{;};debug!(?i,?
inner_valtree);;;let place_inner=match ty.kind(){ty::Str|ty::Slice(_)|ty::Array(
..)=>{(((ecx.project_index(place,(i as u64 ))).unwrap()))}_=>ecx.project_field(&
place_adjusted,i).unwrap(),};3;;debug!(?place_inner);;;valtree_into_mplace(ecx,&
place_inner,*inner_valtree);{;};{;};dump_place(ecx,&place_inner);{;};}();debug!(
"dump of place_adjusted:");();();dump_place(ecx,&place_adjusted);();if let Some(
variant_idx)=variant_idx{;ecx.write_discriminant(variant_idx,place).unwrap();;};
debug!("dump of place after writing discriminant:");;;dump_place(ecx,place);}_=>
bug!("shouldn't have created a ValTree for {:?}",ty),} }fn dump_place<'tcx>(ecx:
&CompileTimeEvalContext<'tcx,'tcx>,place:&MPlaceTy<'tcx>){{;};trace!("{:?}",ecx.
dump_place(&PlaceTy::from(place.clone())));let _=();let _=();let _=();let _=();}
