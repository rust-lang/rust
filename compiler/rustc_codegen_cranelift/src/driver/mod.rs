//! Drivers are responsible for calling [`codegen_fn`] or [`codegen_static`] for each mono item and
//! performing any further actions like JIT executing or writing object files.
//!
//! [`codegen_fn`]: crate::base::codegen_fn
//! [`codegen_static`]: crate::constant::codegen_static

use rustc_data_structures::profiling::SelfProfilerRef;
use rustc_middle::middle::codegen_fn_attrs::CodegenFnAttrFlags;
use rustc_middle::mir::mono::{MonoItem, MonoItemData};

use crate::prelude::*;

pub(crate) mod aot;
#[cfg(feature = "jit")]
pub(crate) mod jit;

fn predefine_mono_items<'tcx>(
    tcx: TyCtxt<'tcx>,
    module: &mut dyn Module,
    mono_items: &[(MonoItem<'tcx>, MonoItemData)],
) {
    tcx.prof.generic_activity("predefine functions").run(|| {
        let is_compiler_builtins = tcx.is_compiler_builtins(LOCAL_CRATE);
        for &(mono_item, data) in mono_items {
            match mono_item {
                MonoItem::Fn(instance) => {
                    let name = tcx.symbol_name(instance).name;
                    let _inst_guard = crate::PrintOnPanic(|| format!("{:?} {}", instance, name));
                    let sig =
                        get_function_sig(tcx, module.target_config().default_call_conv, instance);
                    let linkage = crate::linkage::get_clif_linkage(
                        mono_item,
                        data.linkage,
                        data.visibility,
                        is_compiler_builtins,
                    );
                    let is_naked = tcx
                        .codegen_instance_attrs(instance.def)
                        .flags
                        .contains(CodegenFnAttrFlags::NAKED);
                    module
                        .declare_function(
                            name,
                            // Naked functions are defined in a separate object
                            // file from the codegen unit rustc expects them to
                            // be defined in.
                            if is_naked { Linkage::Import } else { linkage },
                            &sig,
                        )
                        .unwrap();
                }
                MonoItem::Static(_) | MonoItem::GlobalAsm(_) => {}
            }
        }
    });
}

struct MeasuremeProfiler(SelfProfilerRef);

struct TimingGuard {
    profiler: std::mem::ManuallyDrop<SelfProfilerRef>,
    inner: Option<rustc_data_structures::profiling::TimingGuard<'static>>,
}

impl Drop for TimingGuard {
    fn drop(&mut self) {
        self.inner.take();
        unsafe {
            std::mem::ManuallyDrop::drop(&mut self.profiler);
        }
    }
}

impl cranelift_codegen::timing::Profiler for MeasuremeProfiler {
    fn start_pass(&self, pass: cranelift_codegen::timing::Pass) -> Box<dyn std::any::Any> {
        let mut timing_guard = Box::new(TimingGuard {
            profiler: std::mem::ManuallyDrop::new(self.0.clone()),
            inner: None,
        });
        timing_guard.inner = Some(
            unsafe { &*(&*timing_guard.profiler as &SelfProfilerRef as *const SelfProfilerRef) }
                .generic_activity(pass.description()),
        );
        timing_guard
    }
}
