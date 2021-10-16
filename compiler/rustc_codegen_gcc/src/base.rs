use std::env;
use std::time::Instant;

use gccjit::{
    Context,
    FunctionType,
    GlobalKind,
};
use rustc_middle::dep_graph;
use rustc_middle::middle::exported_symbols;
use rustc_middle::ty::TyCtxt;
use rustc_middle::mir::mono::Linkage;
use rustc_codegen_ssa::{ModuleCodegen, ModuleKind};
use rustc_codegen_ssa::base::maybe_create_entry_wrapper;
use rustc_codegen_ssa::mono_item::MonoItemExt;
use rustc_codegen_ssa::traits::DebugInfoMethods;
use rustc_metadata::EncodedMetadata;
use rustc_session::config::DebugInfo;
use rustc_span::Symbol;

use crate::GccContext;
use crate::builder::Builder;
use crate::context::CodegenCx;

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

pub fn compile_codegen_unit<'tcx>(tcx: TyCtxt<'tcx>, cgu_name: Symbol) -> (ModuleCodegen<GccContext>, u64) {
    let prof_timer = tcx.prof.generic_activity("codegen_module");
    let start_time = Instant::now();

    let dep_node = tcx.codegen_unit(cgu_name).codegen_dep_node(tcx);
    let (module, _) = tcx.dep_graph.with_task(
        dep_node,
        tcx,
        cgu_name,
        module_codegen,
        Some(dep_graph::hash_result),
    );
    let time_to_codegen = start_time.elapsed();
    drop(prof_timer);

    // We assume that the cost to run GCC on a CGU is proportional to
    // the time we needed for codegenning it.
    let cost = time_to_codegen.as_secs() * 1_000_000_000 + time_to_codegen.subsec_nanos() as u64;

    fn module_codegen(tcx: TyCtxt<'_>, cgu_name: Symbol) -> ModuleCodegen<GccContext> {
        let cgu = tcx.codegen_unit(cgu_name);
        // Instantiate monomorphizations without filling out definitions yet...
        //let llvm_module = ModuleLlvm::new(tcx, &cgu_name.as_str());
        let context = Context::default();
        // TODO(antoyo): only set on x86 platforms.
        context.add_command_line_option("-masm=intel");
        for arg in &tcx.sess.opts.cg.llvm_args {
            context.add_command_line_option(arg);
        }
        context.add_command_line_option("-fno-semantic-interposition");
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

        {
            let cx = CodegenCx::new(&context, cgu, tcx);

            let mono_items = cgu.items_in_deterministic_order(tcx);
            for &(mono_item, (linkage, visibility)) in &mono_items {
                mono_item.predefine::<Builder<'_, '_, '_>>(&cx, linkage, visibility);
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
                context
            },
            kind: ModuleKind::Regular,
        }
    }

    (module, cost)
}

pub fn write_compressed_metadata<'tcx>(tcx: TyCtxt<'tcx>, metadata: &EncodedMetadata, gcc_module: &mut GccContext) {
    use snap::write::FrameEncoder;
    use std::io::Write;

    // Historical note:
    //
    // When using link.exe it was seen that the section name `.note.rustc`
    // was getting shortened to `.note.ru`, and according to the PE and COFF
    // specification:
    //
    // > Executable images do not use a string table and do not support
    // > section names longer than 8 characters
    //
    // https://docs.microsoft.com/en-us/windows/win32/debug/pe-format
    //
    // As a result, we choose a slightly shorter name! As to why
    // `.note.rustc` works on MinGW, see
    // https://github.com/llvm/llvm-project/blob/llvmorg-12.0.0/lld/COFF/Writer.cpp#L1190-L1197
    let section_name = if tcx.sess.target.is_like_osx { "__DATA,.rustc" } else { ".rustc" };

    let context = &gcc_module.context;
    let mut compressed = rustc_metadata::METADATA_HEADER.to_vec();
    FrameEncoder::new(&mut compressed).write_all(&metadata.raw_data()).unwrap();

    let name = exported_symbols::metadata_symbol_name(tcx);
    let typ = context.new_array_type(None, context.new_type::<u8>(), compressed.len() as i32);
    let global = context.new_global(None, GlobalKind::Exported, typ, name);
    global.global_set_initializer(&compressed);
    global.set_link_section(section_name);

    // Also generate a .section directive to force no
    // flags, at least for ELF outputs, so that the
    // metadata doesn't get loaded into memory.
    let directive = format!(".section {}", section_name);
    context.add_top_level_asm(None, &directive);
}
