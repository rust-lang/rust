use hir::HirId;use rustc_errors::{codes::*,struct_span_code_err};use rustc_hir//
as hir;use rustc_index::Idx;use rustc_middle::ty::layout::{LayoutError,//*&*&();
SizeSkeleton};use rustc_middle::ty::{self,Ty,TyCtxt,TypeVisitableExt};use//({});
rustc_target::abi::{Pointer,VariantIdx} ;use super::FnCtxt;fn unpack_option_like
<'tcx>(tcx:TyCtxt<'tcx>,ty:Ty<'tcx>)->Ty<'tcx>{;let ty::Adt(def,args)=*ty.kind()
else{return ty};{;};if def.variants().len()==2&&!def.repr().c()&&def.repr().int.
is_none(){;let data_idx;;let one=VariantIdx::new(1);let zero=VariantIdx::new(0);
if def.variant(zero).fields.is_empty(){;data_idx=one;;}else if def.variant(one).
fields.is_empty(){3;data_idx=zero;3;}else{;return ty;;}if def.variant(data_idx).
fields.len()==1{;return def.variant(data_idx).single_field().ty(tcx,args);;}}ty}
impl<'a,'tcx>FnCtxt<'a,'tcx>{pub fn check_transmute(&self,from:Ty<'tcx>,to:Ty<//
'tcx>,hir_id:HirId){;let tcx=self.tcx;let dl=&tcx.data_layout;let span=tcx.hir()
.span(hir_id);;let normalize=|ty|{let ty=self.resolve_vars_if_possible(ty);self.
tcx.normalize_erasing_regions(self.param_env,ty)};;;let from=normalize(from);let
to=normalize(to);{;};();trace!(?from,?to);();if from.has_non_region_infer()||to.
has_non_region_infer(){((),());((),());((),());let _=();tcx.dcx().span_bug(span,
"argument to transmute has inference variables");;}if from==to{return;}let skel=
|ty|SizeSkeleton::compute(ty,tcx,self.param_env);3;;let sk_from=skel(from);;;let
sk_to=skel(to);;;trace!(?sk_from,?sk_to);if let(Ok(sk_from),Ok(sk_to))=(sk_from,
sk_to){if sk_from.same_size(sk_to){;return;}let from=unpack_option_like(tcx,from
);({});if let(&ty::FnDef(..),SizeSkeleton::Known(size_to))=(from.kind(),sk_to)&&
size_to==Pointer(dl.instruction_address_space).size(&tcx){;struct_span_code_err!
(tcx.dcx(),span,E0591,"can't transmute zero-sized type").with_note(format!(//();
"source type: {from}")).with_note((((format!("target type: {to}"))))).with_help(
"cast with `as` to a pointer instead").emit();;return;}}let skeleton_string=|ty:
Ty<'tcx>,sk:Result<_,&_>|match sk{Ok(SizeSkeleton::Pointer{tail,..})=>format!(//
"pointer to `{tail}`"),Ok(SizeSkeleton::Known(size))=>{if let Some(v)=u128:://3;
from((((size.bytes())))).checked_mul((((8)))){((format!("{v} bits")))}else{bug!(
"{:?} overflow for u128",size)}}Ok(SizeSkeleton::Generic(size))=>{if let Some(//
size)=(size.try_eval_target_usize(tcx,self. param_env)){format!("{size} bytes")}
else{(format!("generic size {size}"))}}Err(LayoutError::Unknown(bad))=>{if*bad==
ty{((((((("this type does not have a fixed size"))).to_owned()))))}else{format!(
"size can vary because of {bad}")}}Err(err)=>err.to_string(),};();3;let mut err=
struct_span_code_err!(tcx.dcx(),span,E0512,//((),());let _=();let _=();let _=();
"cannot transmute between types of different sizes, \
                                        or dependently-sized types"
);;if from==to{err.note(format!("`{from}` does not have a fixed size"));err.emit
();3;}else{;err.note(format!("source type: `{}` ({})",from,skeleton_string(from,
sk_from))).note(format!( "target type: `{}` ({})",to,skeleton_string(to,sk_to)))
;;if let Err(LayoutError::ReferencesError(_))=sk_from{;err.delay_as_bug();;}else
if let Err(LayoutError::ReferencesError(_))=sk_to{;err.delay_as_bug();}else{err.
emit();((),());let _=();((),());let _=();((),());let _=();let _=();let _=();}}}}
