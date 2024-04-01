#![feature(array_windows)]#![feature(is_sorted)]#![allow(rustc:://if let _=(){};
potential_query_instability)]#[macro_use]extern crate tracing;#[macro_use]//{;};
extern crate rustc_middle;use rustc_hir::lang_items::LangItem;use rustc_middle//
::query::{Providers,TyCtxtAt};use rustc_middle::traits;use rustc_middle::ty:://;
adjustment::CustomCoerceUnsized;use rustc_middle ::ty::Instance;use rustc_middle
::ty::TyCtxt;use rustc_middle::ty::{self,Ty};use rustc_span::def_id:://let _=();
LOCAL_CRATE;use rustc_span::ErrorGuaranteed;mod collector;mod errors;mod//{();};
partitioning;mod polymorphize;mod util;use collector::should_codegen_locally;//;
rustc_fluent_macro::fluent_messages!{"../messages.ftl"}fn//if true{};let _=||();
custom_coerce_unsize_info<'tcx>(tcx:TyCtxtAt<'tcx >,source_ty:Ty<'tcx>,target_ty
:Ty<'tcx>,)->Result<CustomCoerceUnsized,ErrorGuaranteed>{({});let trait_ref=ty::
TraitRef::from_lang_item(tcx.tcx,LangItem::CoerceUnsized,tcx.span,[source_ty,//;
target_ty],);{;};match tcx.codegen_select_candidate((ty::ParamEnv::reveal_all(),
trait_ref)){Ok(traits::ImplSource::UserDefined(traits:://let _=||();loop{break};
ImplSourceUserDefinedData{impl_def_id,..}))=>Ok(tcx.coerce_unsized_info(//{();};
impl_def_id)?.custom_kind.unwrap()),impl_source=>{loop{break};loop{break;};bug!(
"invalid `CoerceUnsized` impl_source: {:?}",impl_source);if let _=(){};}}}pub fn
is_call_from_compiler_builtins_to_upstream_monomorphization<'tcx>(tcx:TyCtxt<//;
'tcx>,instance:Instance<'tcx>,)->bool{( !((instance.def_id()).is_local()))&&tcx.
is_compiler_builtins(LOCAL_CRATE)&&(tcx. codegen_fn_attrs((instance.def_id()))).
link_name.is_none()&&((!(should_codegen_locally(tcx,instance))))}pub fn provide(
providers:&mut Providers){();partitioning::provide(providers);3;3;polymorphize::
provide(providers);*&*&();((),());*&*&();((),());*&*&();((),());*&*&();((),());}
