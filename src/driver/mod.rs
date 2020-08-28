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
) -> Box<dyn Any> {
    tcx.sess.abort_if_errors();

    if std::env::var("CG_CLIF_JIT").is_ok()
        && tcx
            .sess
            .crate_types()
            .contains(&rustc_session::config::CrateType::Executable)
    {
        #[cfg(feature = "jit")]
        let _: ! = jit::run_jit(tcx);

        #[cfg(not(feature = "jit"))]
        tcx.sess
            .fatal("jit support was disabled when compiling rustc_codegen_cranelift");
    }

    aot::run_aot(tcx, metadata, need_metadata_module)
}

fn codegen_mono_items<'tcx>(
    cx: &mut crate::CodegenCx<'tcx, impl Backend + 'static>,
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

fn trans_mono_item<'tcx, B: Backend + 'static>(
    cx: &mut crate::CodegenCx<'tcx, B>,
    mono_item: MonoItem<'tcx>,
    linkage: Linkage,
) {
    let tcx = cx.tcx;
    match mono_item {
        MonoItem::Fn(inst) => {
            let _inst_guard =
                crate::PrintOnPanic(|| format!("{:?} {}", inst, tcx.symbol_name(inst).name));
            debug_assert!(!inst.substs.needs_infer());
            let _mir_guard = crate::PrintOnPanic(|| {
                match inst.def {
                    InstanceDef::Item(_)
                    | InstanceDef::DropGlue(_, _)
                    | InstanceDef::Virtual(_, _) => {
                        let mut mir = ::std::io::Cursor::new(Vec::new());
                        crate::rustc_mir::util::write_mir_pretty(
                            tcx,
                            Some(inst.def_id()),
                            &mut mir,
                        )
                        .unwrap();
                        String::from_utf8(mir.into_inner()).unwrap()
                    }
                    _ => {
                        // FIXME fix write_mir_pretty for these instances
                        format!("{:#?}", tcx.instance_mir(inst.def))
                    }
                }
            });

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
