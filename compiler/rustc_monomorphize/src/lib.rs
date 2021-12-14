#![feature(array_windows)]
#![feature(bool_to_option)]
#![feature(crate_visibility_modifier)]
#![feature(control_flow_enum)]
#![feature(let_else)]
#![recursion_limit = "256"]

#[macro_use]
extern crate tracing;
#[macro_use]
extern crate rustc_middle;

use rustc_hir::lang_items::LangItem;
use rustc_middle::traits;
use rustc_middle::ty::adjustment::CustomCoerceUnsized;
use rustc_middle::ty::query::Providers;
use rustc_middle::ty::{self, Ty, TyCtxt};

mod collector;
mod partitioning;
mod polymorphize;
mod util;

fn custom_coerce_unsize_info<'tcx>(
    tcx: TyCtxt<'tcx>,
    source_ty: Ty<'tcx>,
    target_ty: Ty<'tcx>,
) -> CustomCoerceUnsized {
    let def_id = tcx.require_lang_item(LangItem::CoerceUnsized, None);

    let trait_ref = ty::Binder::dummy(ty::TraitRef {
        def_id,
        substs: tcx.mk_substs_trait(source_ty, &[target_ty.into()]),
    });

    match tcx.codegen_fulfill_obligation((ty::ParamEnv::reveal_all(), trait_ref)) {
        Ok(traits::ImplSource::UserDefined(traits::ImplSourceUserDefinedData {
            impl_def_id,
            ..
        })) => tcx.coerce_unsized_info(impl_def_id).custom_kind.unwrap(),
        impl_source => {
            bug!("invalid `CoerceUnsized` impl_source: {:?}", impl_source);
        }
    }
}

pub fn provide(providers: &mut Providers) {
    partitioning::provide(providers);
    polymorphize::provide(providers);
}
