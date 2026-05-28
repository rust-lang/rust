use std::sync::Arc;
use std::time::Instant;

use gccjit::{CType, FunctionType, GlobalKind};
use rustc_codegen_ssa::ModuleCodegen;
use rustc_codegen_ssa::base::maybe_create_entry_wrapper;
use rustc_codegen_ssa::mono_item::MonoItemExt;
use rustc_codegen_ssa::traits::DebugInfoCodegenMethods;
use rustc_hir::attrs::{AttributeKind, Linkage};
use rustc_hir::find_attr;
use rustc_middle::dep_graph;
#[cfg(feature = "master")]
use rustc_middle::mono::Visibility;
use rustc_middle::ty::TyCtxt;
use rustc_session::config::DebugInfo;
use rustc_span::Symbol;
#[cfg(feature = "master")]
use rustc_target::spec::SymbolVisibility;

use crate::builder::Builder;
use crate::context::CodegenCx;
use crate::gcc_util::new_context;
use crate::{GccContext, LockedTargetInfo, LtoMode, SyncContext};

#[cfg(feature = "master")]
pub fn visibility_to_gcc(visibility: Visibility) -> gccjit::Visibility {
    match visibility {
        Visibility::Default => gccjit::Visibility::Default,
        Visibility::Hidden => gccjit::Visibility::Hidden,
        Visibility::Protected => gccjit::Visibility::Protected,
    }
}

#[cfg(feature = "master")]
pub fn symbol_visibility_to_gcc(visibility: SymbolVisibility) -> gccjit::Visibility {
    match visibility {
        SymbolVisibility::Hidden => gccjit::Visibility::Hidden,
        SymbolVisibility::Protected => gccjit::Visibility::Protected,
        SymbolVisibility::Interposable => gccjit::Visibility::Default,
    }
}

pub fn global_linkage_to_gcc(linkage: Linkage) -> GlobalKind {
    match linkage {
        Linkage::External => GlobalKind::Imported,
        Linkage::AvailableExternally => GlobalKind::Imported,
        Linkage::LinkOnceAny => unimplemented!(),
        Linkage::LinkOnceODR => unimplemented!(),
        Linkage::WeakAny => unimplemented!(),
        Linkage::WeakODR => unimplemented!(),
        Linkage::Internal => GlobalKind::Internal,
        Linkage::ExternalWeak => GlobalKind::Imported, // FIXME(antoyo): should be weak linkage.
        Linkage::Common => unimplemented!(),
    }
}

pub fn linkage_to_gcc(linkage: Linkage) -> FunctionType {
    match linkage {
        Linkage::External => FunctionType::Exported,
        // FIXME(antoyo): set the attribute externally_visible.
        Linkage::AvailableExternally => FunctionType::Extern,
        Linkage::LinkOnceAny => unimplemented!(),
        Linkage::LinkOnceODR => unimplemented!(),
        Linkage::WeakAny => FunctionType::Exported, // FIXME(antoyo): should be similar to linkonce.
        Linkage::WeakODR => unimplemented!(),
        Linkage::Internal => FunctionType::Internal,
        Linkage::ExternalWeak => unimplemented!(),
        Linkage::Common => unimplemented!(),
    }
}

pub fn compile_codegen_unit(
    tcx: TyCtxt<'_>,
    cgu_name: Symbol,
    target_info: LockedTargetInfo,
    lto_supported: bool,
) -> (ModuleCodegen<GccContext>, u64) {
    let prof_timer = tcx.prof.generic_activity("codegen_module");
    let start_time = Instant::now();

    let dep_node = tcx.codegen_unit(cgu_name).codegen_dep_node(tcx);
    let (module, _) = tcx.dep_graph.with_task(
        dep_node,
        tcx,
        || module_codegen(tcx, cgu_name, target_info, lto_supported),
        Some(dep_graph::hash_result),
    );
    let time_to_codegen = start_time.elapsed();
    drop(prof_timer);

    // We assume that the cost to run GCC on a CGU is proportional to
    // the time we needed for codegenning it.
    let cost = time_to_codegen.as_secs() * 1_000_000_000 + time_to_codegen.subsec_nanos() as u64;

    fn module_codegen(
        tcx: TyCtxt<'_>,
        cgu_name: Symbol,
        target_info: LockedTargetInfo,
        lto_supported: bool,
    ) -> ModuleCodegen<GccContext> {
        let cgu = tcx.codegen_unit(cgu_name);
        // Instantiate monomorphizations without filling out definitions yet...
        let context = new_context(tcx.sess);

        // NOTE: We need to honor the `#![no_builtins]` attribute to prevent GCC from
        // replacing code patterns (like loops) with calls to builtins (like memset).
        // The `-fno-tree-loop-distribute-patterns` flag disables the loop distribution pass
        // that transforms loops into calls to library functions (memset, memcpy, etc.).
        // See GCC handling for more details:
        // https://github.com/rust-lang/gcc/blob/efdd0a7290c22f5438d7c5380105d353ee3e8518/gcc/c-family/c-opts.cc#L953
        let crate_attrs = tcx.hir_attrs(rustc_hir::CRATE_HIR_ID);
        if find_attr!(crate_attrs, AttributeKind::NoBuiltins) {
            context.add_command_line_option("-fno-tree-loop-distribute-patterns");
        }

        // NOTE: The codegen generates unreachable blocks.
        context.set_allow_unreachable_blocks(true);

        {
            // FIXME: to make it less error-prone (calling get_target_info() will add the flag
            // -fsyntax-only), forbid the compilation when get_target_info() is called on a
            // context.
            let f16_type_supported = target_info.supports_target_dependent_type(CType::Float16);
            let f32_type_supported = target_info.supports_target_dependent_type(CType::Float32);
            let f64_type_supported = target_info.supports_target_dependent_type(CType::Float64);
            let f128_type_supported = target_info.supports_target_dependent_type(CType::Float128);
            let u128_type_supported = target_info.supports_target_dependent_type(CType::UInt128t);
            // FIXME: improve this to avoid passing that many arguments.
            let mut cx = CodegenCx::new(
                &context,
                cgu,
                tcx,
                u128_type_supported,
                f16_type_supported,
                f32_type_supported,
                f64_type_supported,
                f128_type_supported,
            );

            let mono_items = cgu.items_in_deterministic_order(tcx);
            for &(mono_item, data) in &mono_items {
                mono_item.predefine::<Builder<'_, '_, '_>>(
                    &mut cx,
                    cgu_name.as_str(),
                    data.linkage,
                    data.visibility,
                );
            }

            // ... and now that we have everything pre-defined, fill out those definitions.
            for &(mono_item, item_data) in &mono_items {
                mono_item.define::<Builder<'_, '_, '_>>(&mut cx, cgu_name.as_str(), item_data);
            }

            // If this codegen unit contains the main function, also create the
            // wrapper here
            maybe_create_entry_wrapper::<Builder<'_, '_, '_>>(&cx, cx.codegen_unit);

            // Finalize debuginfo
            if cx.sess().opts.debuginfo != DebugInfo::None {
                cx.debuginfo_finalize();
            }
        }

        ModuleCodegen::new_regular(
            cgu_name.to_string(),
            GccContext {
                context: Arc::new(SyncContext::new(context)),
                relocation_model: tcx.sess.relocation_model(),
                lto_supported,
                lto_mode: LtoMode::None,
                temp_dir: None,
            },
        )
    }

    (module, cost)
}
