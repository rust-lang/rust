use std::collections::HashSet;
use std::env;
use std::sync::Arc;
use std::time::Instant;

use gccjit::{CType, Context, FunctionType, GlobalKind};
use rustc_codegen_ssa::ModuleCodegen;
use rustc_codegen_ssa::base::maybe_create_entry_wrapper;
use rustc_codegen_ssa::mono_item::MonoItemExt;
use rustc_codegen_ssa::traits::DebugInfoCodegenMethods;
use rustc_middle::dep_graph;
use rustc_middle::mir::mono::Linkage;
#[cfg(feature = "master")]
use rustc_middle::mir::mono::Visibility;
use rustc_middle::ty::TyCtxt;
use rustc_session::config::DebugInfo;
use rustc_span::Symbol;
#[cfg(feature = "master")]
use rustc_target::spec::SymbolVisibility;
use rustc_target::spec::{PanicStrategy, RelocModel};

use crate::builder::Builder;
use crate::context::CodegenCx;
use crate::{GccContext, LockedTargetInfo, SyncContext, gcc_util, new_context};

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
        Linkage::ExternalWeak => GlobalKind::Imported, // TODO(antoyo): should be weak linkage.
        Linkage::Common => unimplemented!(),
    }
}

pub fn linkage_to_gcc(linkage: Linkage) -> FunctionType {
    match linkage {
        Linkage::External => FunctionType::Exported,
        // TODO(antoyo): set the attribute externally_visible.
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
) -> (ModuleCodegen<GccContext>, u64) {
    let prof_timer = tcx.prof.generic_activity("codegen_module");
    let start_time = Instant::now();

    let dep_node = tcx.codegen_unit(cgu_name).codegen_dep_node(tcx);
    let (module, _) = tcx.dep_graph.with_task(
        dep_node,
        tcx,
        (cgu_name, target_info),
        module_codegen,
        Some(dep_graph::hash_result),
    );
    let time_to_codegen = start_time.elapsed();
    drop(prof_timer);

    // We assume that the cost to run GCC on a CGU is proportional to
    // the time we needed for codegenning it.
    let cost = time_to_codegen.as_secs() * 1_000_000_000 + time_to_codegen.subsec_nanos() as u64;

    fn module_codegen(
        tcx: TyCtxt<'_>,
        (cgu_name, target_info): (Symbol, LockedTargetInfo),
    ) -> ModuleCodegen<GccContext> {
        let cgu = tcx.codegen_unit(cgu_name);
        // Instantiate monomorphizations without filling out definitions yet...
        let context = new_context(tcx);

        if tcx.sess.panic_strategy() == PanicStrategy::Unwind {
            context.add_command_line_option("-fexceptions");
            context.add_driver_option("-fexceptions");
        }

        let disabled_features: HashSet<_> = tcx
            .sess
            .opts
            .cg
            .target_feature
            .split(',')
            .filter(|feature| feature.starts_with('-'))
            .map(|string| &string[1..])
            .collect();

        if !disabled_features.contains("avx") && tcx.sess.target.arch == "x86_64" {
            // NOTE: we always enable AVX because the equivalent of llvm.x86.sse2.cmp.pd in GCC for
            // SSE2 is multiple builtins, so we use the AVX __builtin_ia32_cmppd instead.
            // FIXME(antoyo): use the proper builtins for llvm.x86.sse2.cmp.pd and similar.
            context.add_command_line_option("-mavx");
        }

        for arg in &tcx.sess.opts.cg.llvm_args {
            context.add_command_line_option(arg);
        }
        // NOTE: This is needed to compile the file src/intrinsic/archs.rs during a bootstrap of rustc.
        context.add_command_line_option("-fno-var-tracking-assignments");
        // NOTE: an optimization (https://github.com/rust-lang/rustc_codegen_gcc/issues/53).
        context.add_command_line_option("-fno-semantic-interposition");
        // NOTE: Rust relies on LLVM not doing TBAA (https://github.com/rust-lang/unsafe-code-guidelines/issues/292).
        context.add_command_line_option("-fno-strict-aliasing");
        // NOTE: Rust relies on LLVM doing wrapping on overflow.
        context.add_command_line_option("-fwrapv");

        if let Some(model) = tcx.sess.code_model() {
            use rustc_target::spec::CodeModel;

            context.add_command_line_option(match model {
                CodeModel::Tiny => "-mcmodel=tiny",
                CodeModel::Small => "-mcmodel=small",
                CodeModel::Kernel => "-mcmodel=kernel",
                CodeModel::Medium => "-mcmodel=medium",
                CodeModel::Large => "-mcmodel=large",
            });
        }

        add_pic_option(&context, tcx.sess.relocation_model());

        let target_cpu = gcc_util::target_cpu(tcx.sess);
        if target_cpu != "generic" {
            context.add_command_line_option(format!("-march={}", target_cpu));
        }

        if tcx
            .sess
            .opts
            .unstable_opts
            .function_sections
            .unwrap_or(tcx.sess.target.function_sections)
        {
            context.add_command_line_option("-ffunction-sections");
            context.add_command_line_option("-fdata-sections");
        }

        if env::var("CG_GCCJIT_DUMP_RTL").as_deref() == Ok("1") {
            context.add_command_line_option("-fdump-rtl-vregs");
        }
        if env::var("CG_GCCJIT_DUMP_RTL_ALL").as_deref() == Ok("1") {
            context.add_command_line_option("-fdump-rtl-all");
        }
        if env::var("CG_GCCJIT_DUMP_TREE_ALL").as_deref() == Ok("1") {
            context.add_command_line_option("-fdump-tree-all-eh");
        }
        if env::var("CG_GCCJIT_DUMP_IPA_ALL").as_deref() == Ok("1") {
            context.add_command_line_option("-fdump-ipa-all-eh");
        }
        if env::var("CG_GCCJIT_DUMP_CODE").as_deref() == Ok("1") {
            context.set_dump_code_on_compile(true);
        }
        if env::var("CG_GCCJIT_DUMP_GIMPLE").as_deref() == Ok("1") {
            context.set_dump_initial_gimple(true);
        }
        if env::var("CG_GCCJIT_DUMP_EVERYTHING").as_deref() == Ok("1") {
            context.set_dump_everything(true);
        }
        if env::var("CG_GCCJIT_KEEP_INTERMEDIATES").as_deref() == Ok("1") {
            context.set_keep_intermediates(true);
        }
        if env::var("CG_GCCJIT_VERBOSE").as_deref() == Ok("1") {
            context.add_driver_option("-v");
        }

        // NOTE: The codegen generates unreachable blocks.
        context.set_allow_unreachable_blocks(true);

        {
            // TODO: to make it less error-prone (calling get_target_info() will add the flag
            // -fsyntax-only), forbid the compilation when get_target_info() is called on a
            // context.
            let f16_type_supported = target_info.supports_target_dependent_type(CType::Float16);
            let f32_type_supported = target_info.supports_target_dependent_type(CType::Float32);
            let f64_type_supported = target_info.supports_target_dependent_type(CType::Float64);
            let f128_type_supported = target_info.supports_target_dependent_type(CType::Float128);
            let u128_type_supported = target_info.supports_target_dependent_type(CType::UInt128t);
            // TODO: improve this to avoid passing that many arguments.
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
                should_combine_object_files: false,
                temp_dir: None,
            },
        )
    }

    (module, cost)
}

pub fn add_pic_option<'gcc>(context: &Context<'gcc>, relocation_model: RelocModel) {
    match relocation_model {
        rustc_target::spec::RelocModel::Static => {
            context.add_command_line_option("-fno-pie");
            context.add_driver_option("-fno-pie");
        }
        rustc_target::spec::RelocModel::Pic => {
            context.add_command_line_option("-fPIC");
            // NOTE: we use both add_command_line_option and add_driver_option because the usage in
            // this module (compile_codegen_unit) requires add_command_line_option while the usage
            // in the back::write module (codegen) requires add_driver_option.
            context.add_driver_option("-fPIC");
        }
        rustc_target::spec::RelocModel::Pie => {
            context.add_command_line_option("-fPIE");
            context.add_driver_option("-fPIE");
        }
        model => eprintln!("Unsupported relocation model: {:?}", model),
    }
}
