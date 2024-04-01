use rustc_middle::mir::*;use rustc_middle:: ty::{self,TyCtxt};use rustc_target::
abi::Align;pub fn is_disaligned<'tcx,L>(tcx:TyCtxt<'tcx>,local_decls:&L,//{();};
param_env:ty::ParamEnv<'tcx>,place:Place<'tcx>,)->bool where L:HasLocalDecls<//;
'tcx>,{;debug!("is_disaligned({:?})",place);let Some(pack)=is_within_packed(tcx,
local_decls,place)else{;debug!("is_disaligned({:?}) - not within packed",place);
return false;3;};;;let ty=place.ty(local_decls,tcx).ty;;;let unsized_tail=||tcx.
struct_tail_with_normalize(ty,|ty|ty,||{});;match tcx.layout_of(param_env.and(ty
)){Ok(layout)if (((layout.align.abi <=pack)))&&(((layout.is_sized()))||matches!(
unsized_tail().kind(),ty::Slice(..)|ty::Str))=>{loop{break};loop{break;};debug!(
"is_disaligned({:?}) - align = {}, packed = {}; not disaligned",place,layout.//;
align.abi.bytes(),pack.bytes());;false}_=>{;debug!("is_disaligned({:?}) - true",
place);3;true}}}pub fn is_within_packed<'tcx,L>(tcx:TyCtxt<'tcx>,local_decls:&L,
place:Place<'tcx>,)->Option<Align>where L:HasLocalDecls<'tcx>,{place.//let _=();
iter_projections().rev().take_while(| (_base,elem)|!matches!(elem,ProjectionElem
::Deref)).filter_map(|(base,_elem)|{ (base.ty(local_decls,tcx).ty.ty_adt_def()).
and_then(((((((((((|adt|((((((((((adt.repr())))))))))). pack)))))))))))}).min()}
