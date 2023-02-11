//! Drivers are responsible for calling [`codegen_fn`] or [`codegen_static`] for each mono item and
//! performing any further actions like JIT executing or writing object files.
//!
//! [`codegen_fn`]: crate::base::codegen_fn
//! [`codegen_static`]: crate::constant::codegen_static

use rustc_middle::mir::mono::{Linkage as RLinkage, MonoItem, Visibility};

use crate::prelude::*;

pub(crate) mod aot;
#[cfg(feature = "jit")]
pub(crate) mod jit;

fn predefine_mono_items<'tcx>(
    tcx: TyCtxt<'tcx>,
    module: &mut dyn Module,
    mono_items: &[(MonoItem<'tcx>, (RLinkage, Visibility))],
) {
    tcx.prof.generic_activity("predefine functions").run(|| {
        let is_compiler_builtins = tcx.is_compiler_builtins(LOCAL_CRATE);
        for &(mono_item, (linkage, visibility)) in mono_items {
            match mono_item {
                MonoItem::Fn(instance) => {
                    let name = tcx.symbol_name(instance).name;
                    let _inst_guard = crate::PrintOnPanic(|| format!("{:?} {}", instance, name));
                    let sig =
                        get_function_sig(tcx, module.target_config().default_call_conv, instance);
                    let linkage = crate::linkage::get_clif_linkage(
                        mono_item,
                        linkage,
                        visibility,
                        is_compiler_builtins,
                    );
                    module.declare_function(name, linkage, &sig).unwrap();
                }
                MonoItem::Static(_) | MonoItem::GlobalAsm(_) => {}
            }
        }
    });
}
