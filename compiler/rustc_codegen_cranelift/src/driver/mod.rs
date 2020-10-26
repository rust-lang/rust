//! Drivers are responsible for calling [`codegen_mono_items`] and performing any further actions
//! like JIT executing or writing object files.

use std::any::Any;

use rustc_middle::middle::cstore::EncodedMetadata;
use rustc_middle::mir::mono::{Linkage as RLinkage, MonoItem, Visibility};

use crate::prelude::*;

mod aot;
#[cfg(feature = "jit")]
mod jit;

pub(crate) fn codegen_crate(
    tcx: TyCtxt<'_>,
    metadata: EncodedMetadata,
    need_metadata_module: bool,
    config: crate::BackendConfig,
) -> Box<dyn Any> {
    tcx.sess.abort_if_errors();

    if config.use_jit {
        let is_executable = tcx
            .sess
            .crate_types()
            .contains(&rustc_session::config::CrateType::Executable);
        if !is_executable {
            tcx.sess.fatal("can't jit non-executable crate");
        }

        #[cfg(feature = "jit")]
        let _: ! = jit::run_jit(tcx);

        #[cfg(not(feature = "jit"))]
        tcx.sess
            .fatal("jit support was disabled when compiling rustc_codegen_cranelift");
    }

    aot::run_aot(tcx, metadata, need_metadata_module)
}

fn codegen_mono_items<'tcx>(
    cx: &mut crate::CodegenCx<'tcx, impl Module>,
    mono_items: Vec<(MonoItem<'tcx>, (RLinkage, Visibility))>,
) {
    cx.tcx.sess.time("predefine functions", || {
        for &(mono_item, (linkage, visibility)) in &mono_items {
            match mono_item {
                MonoItem::Fn(instance) => {
                    let (name, sig) = get_function_name_and_sig(
                        cx.tcx,
                        cx.module.isa().triple(),
                        instance,
                        false,
                    );
                    let linkage = crate::linkage::get_clif_linkage(mono_item, linkage, visibility);
                    cx.module.declare_function(&name, linkage, &sig).unwrap();
                }
                MonoItem::Static(_) | MonoItem::GlobalAsm(_) => {}
            }
        }
    });

    for (mono_item, (linkage, visibility)) in mono_items {
        let linkage = crate::linkage::get_clif_linkage(mono_item, linkage, visibility);
        trans_mono_item(cx, mono_item, linkage);
    }
}

fn trans_mono_item<'tcx, M: Module>(
    cx: &mut crate::CodegenCx<'tcx, M>,
    mono_item: MonoItem<'tcx>,
    linkage: Linkage,
) {
    let tcx = cx.tcx;
    match mono_item {
        MonoItem::Fn(inst) => {
            let _inst_guard =
                crate::PrintOnPanic(|| format!("{:?} {}", inst, tcx.symbol_name(inst).name));
            debug_assert!(!inst.substs.needs_infer());
            tcx.sess
                .time("codegen fn", || crate::base::trans_fn(cx, inst, linkage));
        }
        MonoItem::Static(def_id) => {
            crate::constant::codegen_static(&mut cx.constants_cx, def_id);
        }
        MonoItem::GlobalAsm(hir_id) => {
            let item = tcx.hir().expect_item(hir_id);
            if let rustc_hir::ItemKind::GlobalAsm(rustc_hir::GlobalAsm { asm }) = item.kind {
                cx.global_asm.push_str(&*asm.as_str());
                cx.global_asm.push_str("\n\n");
            } else {
                bug!("Expected GlobalAsm found {:?}", item);
            }
        }
    }
}

fn time<R>(tcx: TyCtxt<'_>, name: &'static str, f: impl FnOnce() -> R) -> R {
    if std::env::var("CG_CLIF_DISPLAY_CG_TIME")
        .as_ref()
        .map(|val| &**val)
        == Ok("1")
    {
        println!("[{:<30}: {}] start", tcx.crate_name(LOCAL_CRATE), name);
        let before = std::time::Instant::now();
        let res = tcx.sess.time(name, f);
        let after = std::time::Instant::now();
        println!(
            "[{:<30}: {}] end time: {:?}",
            tcx.crate_name(LOCAL_CRATE),
            name,
            after - before
        );
        res
    } else {
        tcx.sess.time(name, f)
    }
}
