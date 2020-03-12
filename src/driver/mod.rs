use std::any::Any;

use rustc::middle::cstore::EncodedMetadata;
use rustc::mir::mono::{Linkage as RLinkage, Visibility};

use crate::prelude::*;

mod aot;
#[cfg(not(target_arch = "wasm32"))]
mod jit;

pub fn codegen_crate(
    tcx: TyCtxt<'_>,
    metadata: EncodedMetadata,
    need_metadata_module: bool,
) -> Box<dyn Any> {
    tcx.sess.abort_if_errors();

    if std::env::var("CG_CLIF_JIT").is_ok()
        && tcx.sess.crate_types.get().contains(&CrateType::Executable)
    {
        #[cfg(not(target_arch = "wasm32"))]
        let _: ! = jit::run_jit(tcx);

        #[cfg(target_arch = "wasm32")]
        panic!("jit not supported on wasm");
    }

    aot::run_aot(tcx, metadata, need_metadata_module)
}

fn codegen_mono_items<'tcx>(
    tcx: TyCtxt<'tcx>,
    module: &mut Module<impl Backend + 'static>,
    debug_context: Option<&mut DebugContext<'tcx>>,
    mono_items: Vec<(MonoItem<'tcx>, (RLinkage, Visibility))>,
) {
    let mut cx = CodegenCx::new(tcx, module, debug_context);

    tcx.sess.time("predefine functions", || {
        for &(mono_item, (linkage, visibility)) in &mono_items {
            match mono_item {
                MonoItem::Fn(instance) => {
                    let (name, sig) =
                        get_function_name_and_sig(tcx, cx.module.isa().triple(), instance, false);
                    let linkage = crate::linkage::get_clif_linkage(mono_item, linkage, visibility);
                    cx.module.declare_function(&name, linkage, &sig).unwrap();
                }
                MonoItem::Static(_) | MonoItem::GlobalAsm(_) => {}
            }
        }
    });

    for (mono_item, (linkage, visibility)) in mono_items {
        crate::unimpl::try_unimpl(tcx, || {
            let linkage = crate::linkage::get_clif_linkage(mono_item, linkage, visibility);
            trans_mono_item(&mut cx, mono_item, linkage);
        });
    }

    tcx.sess.time("finalize CodegenCx", || cx.finalize());
}

fn trans_mono_item<'clif, 'tcx, B: Backend + 'static>(
    cx: &mut crate::CodegenCx<'clif, 'tcx, B>,
    mono_item: MonoItem<'tcx>,
    linkage: Linkage,
) {
    let tcx = cx.tcx;
    match mono_item {
        MonoItem::Fn(inst) => {
            let _inst_guard =
                PrintOnPanic(|| format!("{:?} {}", inst, tcx.symbol_name(inst).name.as_str()));
            debug_assert!(!inst.substs.needs_infer());
            let _mir_guard = PrintOnPanic(|| {
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

            cx.tcx.sess.time("codegen fn", || crate::base::trans_fn(cx, inst, linkage));
        }
        MonoItem::Static(def_id) => {
            crate::constant::codegen_static(&mut cx.constants_cx, def_id);
        }
        MonoItem::GlobalAsm(hir_id) => {
            let item = tcx.hir().expect_item(hir_id);
            if let rustc_hir::ItemKind::GlobalAsm(rustc_hir::GlobalAsm { asm }) = item.kind {
                // FIXME implement global asm using an external assembler
                if asm.as_str().contains("__rust_probestack") {
                    return;
                } else {
                    tcx
                        .sess
                        .fatal(&format!("Unimplemented global asm mono item \"{}\"", asm));
                }
            } else {
                bug!("Expected GlobalAsm found {:?}", item);
            }
        }
    }
}

fn time<R>(tcx: TyCtxt<'_>, name: &'static str, f: impl FnOnce() -> R) -> R {
    if std::env::var("CG_CLIF_DISPLAY_CG_TIME").is_ok() {
        println!("[{:<30}: {}] start", tcx.crate_name(LOCAL_CRATE), name);
        let before = std::time::Instant::now();
        let res = tcx.sess.time(name, f);
        let after = std::time::Instant::now();
        println!("[{:<30}: {}] end time: {:?}", tcx.crate_name(LOCAL_CRATE), name, after - before);
        res
    } else {
        tcx.sess.time(name, f)
    }
}
