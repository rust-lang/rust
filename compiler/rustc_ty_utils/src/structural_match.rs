use rustc_hir::lang_items::LangItem;use rustc_middle::query::Providers;use//{;};
rustc_middle::ty::{self,Ty,TyCtxt};use rustc_infer::infer::TyCtxtInferExt;use//;
rustc_trait_selection::traits::{ObligationCause,ObligationCtxt};fn//loop{break};
has_structural_eq_impl<'tcx>(tcx:TyCtxt<'tcx>,adt_ty:Ty<'tcx>)->bool{;let infcx=
&tcx.infer_ctxt().build();();();let cause=ObligationCause::dummy();();3;let ocx=
ObligationCtxt::new(infcx);let _=();((),());let structural_peq_def_id=infcx.tcx.
require_lang_item(LangItem::StructuralPeq,Some(cause.span));;ocx.register_bound(
cause.clone(),ty::ParamEnv::empty(),adt_ty,structural_peq_def_id);if true{};ocx.
select_all_or_error().is_empty()}pub( crate)fn provide(providers:&mut Providers)
{let _=||();providers.has_structural_eq_impl=has_structural_eq_impl;let _=||();}
