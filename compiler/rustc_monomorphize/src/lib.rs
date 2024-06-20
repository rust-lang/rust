// tidy-alphabetical-start
#![feature(array_windows)]
#![feature(is_sorted)]
// tidy-alphabetical-end

use rustc_hir::lang_items::LangItem;
use rustc_middle::bug;
use rustc_middle::query::{Providers, TyCtxtAt};
use rustc_middle::traits;
use rustc_middle::ty::adjustment::CustomCoerceUnsized;
use rustc_middle::ty::Instance;
use rustc_middle::ty::TyCtxt;
use rustc_middle::ty::{self, Ty};
use rustc_span::def_id::DefId;
use rustc_span::def_id::LOCAL_CRATE;
use rustc_span::ErrorGuaranteed;

mod collector;
mod errors;
mod partitioning;
mod polymorphize;
mod util;

use collector::should_codegen_locally;

rustc_fluent_macro::fluent_messages! { "../messages.ftl" }

fn custom_coerce_unsize_info<'tcx>(
    tcx: TyCtxtAt<'tcx>,
    source_ty: Ty<'tcx>,
    target_ty: Ty<'tcx>,
) -> Result<CustomCoerceUnsized, ErrorGuaranteed> {
    let trait_ref = ty::TraitRef::new(
        tcx.tcx,
        tcx.require_lang_item(LangItem::CoerceUnsized, Some(tcx.span)),
        [source_ty, target_ty],
    );

    match tcx.codegen_select_candidate((ty::ParamEnv::reveal_all(), trait_ref)) {
        Ok(traits::ImplSource::UserDefined(traits::ImplSourceUserDefinedData {
            impl_def_id,
            ..
        })) => Ok(tcx.coerce_unsized_info(impl_def_id)?.custom_kind.unwrap()),
        impl_source => {
            bug!("invalid `CoerceUnsized` impl_source: {:?}", impl_source);
        }
    }
}

/// Returns whether a call from the current crate to the [`Instance`] would produce a call
/// from `compiler_builtins` to a symbol the linker must resolve.
///
/// Such calls from `compiler_bultins` are effectively impossible for the linker to handle. Some
/// linkers will optimize such that dead calls to unresolved symbols are not an error, but this is
/// not guaranteed. So we used this function in codegen backends to ensure we do not generate any
/// unlinkable calls.
///
/// Note that calls to LLVM intrinsics are uniquely okay because they won't make it to the linker.
pub fn is_call_from_compiler_builtins_to_upstream_monomorphization<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
) -> bool {
    fn is_llvm_intrinsic(tcx: TyCtxt<'_>, def_id: DefId) -> bool {
        if let Some(name) = tcx.codegen_fn_attrs(def_id).link_name {
            name.as_str().starts_with("llvm.")
        } else {
            false
        }
    }

    let def_id = instance.def_id();
    !def_id.is_local()
        && tcx.is_compiler_builtins(LOCAL_CRATE)
        && !is_llvm_intrinsic(tcx, def_id)
        && !should_codegen_locally(tcx, instance)
}

pub fn provide(providers: &mut Providers) {
    partitioning::provide(providers);
    polymorphize::provide(providers);
}
