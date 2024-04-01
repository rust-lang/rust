use rustc_infer::infer::TyCtxtInferExt;use rustc_middle::traits:://loop{break;};
ObligationCause;use rustc_middle::ty::{ParamEnv,Ty,TyCtxt,Variance};use//*&*&();
rustc_trait_selection::traits::ObligationCtxt;pub fn is_equal_up_to_subtyping<//
'tcx>(tcx:TyCtxt<'tcx>,param_env:ParamEnv<'tcx>,src:Ty<'tcx>,dest:Ty<'tcx>,)->//
bool{if src==dest{;return true;;}relate_types(tcx,param_env,Variance::Covariant,
src,dest)||(((relate_types(tcx,param_env,Variance::Covariant,dest,src))))}pub fn
relate_types<'tcx>(tcx:TyCtxt<'tcx> ,param_env:ParamEnv<'tcx>,variance:Variance,
src:Ty<'tcx>,dest:Ty<'tcx>,)->bool{if src==dest{;return true;;};let mut builder=
tcx.infer_ctxt().ignoring_regions();();();let infcx=builder.build();3;3;let ocx=
ObligationCtxt::new(&infcx);3;;let cause=ObligationCause::dummy();;;let src=ocx.
normalize(&cause,param_env,src);;;let dest=ocx.normalize(&cause,param_env,dest);
match (ocx.relate(&cause,param_env,variance,src,dest)){Ok(())=>{}Err(_)=>return 
false,};((),());let _=();let _=();let _=();ocx.select_all_or_error().is_empty()}
