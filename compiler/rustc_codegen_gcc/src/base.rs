use std::collections::HashSet;
use std::env;
use std::time::Instant;

use gccjit::{
    Context,
    FunctionType,
    GlobalKind,
};
use rustc_middle::dep_graph;
use rustc_middle::ty::TyCtxt;
#[cfg(feature="master")]
use rustc_middle::mir::mono::Visibility;
use rustc_middle::mir::mono::Linkage;
use rustc_codegen_ssa::{ModuleCodegen, ModuleKind};
use rustc_codegen_ssa::base::maybe_create_entry_wrapper;
use rustc_codegen_ssa::mono_item::MonoItemExt;
use rustc_codegen_ssa::traits::DebugInfoMethods;
use rustc_session::config::DebugInfo;
use rustc_span::Symbol;

use crate::{LockedTargetInfo, gcc_util};
use crate::GccContext;
use crate::builder::Builder;
use crate::context::CodegenCx;

#[cfg(feature="master")]
pub fn visibility_to_gcc(linkage: Visibility) -> gccjit::Visibility {
    match linkage {
        Visibility::Default => gccjit::Visibility::Default,
        Visibility::Hidden => gccjit::Visibility::Hidden,
        Visibility::Protected => gccjit::Visibility::Protected,
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
        Linkage::Appending => unimplemented!(),
        Linkage::Internal => GlobalKind::Internal,
        Linkage::Private => GlobalKind::Internal,
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
        Linkage::Appending => unimplemented!(),
        Linkage::Internal => FunctionType::Internal,
        Linkage::Private => FunctionType::Internal,
        Linkage::ExternalWeak => unimplemented!(),
        Linkage::Common => unimplemented!(),
    }
}

pub fn compile_codegen_unit(tcx: TyCtxt<'_>, cgu_name: Symbol, target_info: LockedTargetInfo) -> (ModuleCodegen<GccContext>, u64) {
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

    fn module_codegen(tcx: TyCtxt<'_>, (cgu_name, target_info): (Symbol, LockedTargetInfo)) -> ModuleCodegen<GccContext> {
        let cgu = tcx.codegen_unit(cgu_name);
        // Instantiate monomorphizations without filling out definitions yet...
        let context = Context::default();

        context.add_command_line_option("-fexceptions");
        context.add_driver_option("-fexceptions");

        let disabled_features: HashSet<_> = tcx.sess.opts.cg.target_feature.split(',')
            .filter(|feature| feature.starts_with('-'))
            .map(|string| &string[1..])
            .collect();

        if tcx.sess.target.arch == "x86" || tcx.sess.target.arch == "x86_64" {
            context.add_command_line_option("-masm=intel");
        }

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

        if tcx.sess.relocation_model() == rustc_target::spec::RelocModel::Static {
            context.add_command_line_option("-mcmodel=kernel");
            context.add_command_line_option("-fno-pie");
        }

        let target_cpu = gcc_util::target_cpu(tcx.sess);
        if target_cpu != "generic" {
            context.add_command_line_option(&format!("-march={}", target_cpu));
        }

        if tcx.sess.opts.unstable_opts.function_sections.unwrap_or(tcx.sess.target.function_sections) {
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
        context.set_debug_info(true);
        if env::var("CG_GCCJIT_DUMP_EVERYTHING").as_deref() == Ok("1") {
            context.set_dump_everything(true);
        }
        if env::var("CG_GCCJIT_KEEP_INTERMEDIATES").as_deref() == Ok("1") {
            context.set_keep_intermediates(true);
        }

        if env::var("CG_GCCJIT_VERBOSE").as_deref() == Ok("1") {
            context.add_driver_option("-v");
        }

        // NOTE: The codegen generates unrechable blocks.
        context.set_allow_unreachable_blocks(true);

        {
            let cx = CodegenCx::new(&context, cgu, tcx, target_info.supports_128bit_int());

            let mono_items = cgu.items_in_deterministic_order(tcx);
            for &(mono_item, data) in &mono_items {
                mono_item.predefine::<Builder<'_, '_, '_>>(&cx, data.linkage, data.visibility);
            }

            // ... and now that we have everything pre-defined, fill out those definitions.
            for &(mono_item, _) in &mono_items {
                mono_item.define::<Builder<'_, '_, '_>>(&cx);
            }

            // If this codegen unit contains the main function, also create the
            // wrapper here
            maybe_create_entry_wrapper::<Builder<'_, '_, '_>>(&cx);

            // Finalize debuginfo
            if cx.sess().opts.debuginfo != DebugInfo::None {
                cx.debuginfo_finalize();
            }
        }

        ModuleCodegen {
            name: cgu_name.to_string(),
            module_llvm: GccContext {
                context,
                should_combine_object_files: false,
                temp_dir: None,
            },
            kind: ModuleKind::Regular,
        }
    }

    (module, cost)
}
