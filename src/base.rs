use std::env;
use std::sync::Once;
use std::time::Instant;

use gccjit::{
    Context,
    FunctionType,
    GlobalKind,
};
use rustc_hir::def_id::LOCAL_CRATE;
use rustc_middle::dep_graph;
use rustc_middle::middle::cstore::EncodedMetadata;
use rustc_middle::ty::TyCtxt;
use rustc_middle::mir::mono::Linkage;
use rustc_codegen_ssa::{ModuleCodegen, ModuleKind};
use rustc_codegen_ssa::base::maybe_create_entry_wrapper;
use rustc_codegen_ssa::mono_item::MonoItemExt;
use rustc_codegen_ssa::traits::DebugInfoMethods;
use rustc_session::config::DebugInfo;
use rustc_span::Symbol;

use crate::{GccContext, create_function_calling_initializers};
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
        Linkage::ExternalWeak => GlobalKind::Imported, // TODO: should be weak linkage.
        Linkage::Common => unimplemented!(),
    }
}

pub fn linkage_to_gcc(linkage: Linkage) -> FunctionType {
    match linkage {
        Linkage::External => FunctionType::Exported,
        Linkage::AvailableExternally => FunctionType::Extern,
        Linkage::LinkOnceAny => unimplemented!(),
        Linkage::LinkOnceODR => unimplemented!(),
        Linkage::WeakAny => FunctionType::Exported, // FIXME: should be similar to linkonce.
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
    let (module, _) = tcx.dep_graph.with_task(dep_node, tcx, cgu_name, module_codegen, dep_graph::hash_result);
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
        // TODO: only set on x86 platforms.
        context.add_command_line_option("-masm=intel");
        for arg in &tcx.sess.opts.cg.llvm_args {
            context.add_command_line_option(arg);
        }
        //context.set_dump_code_on_compile(true);
        if env::var("CG_GCCJIT_DUMP_GIMPLE").as_deref() == Ok("1") {
            context.set_dump_initial_gimple(true);
        }
        context.set_debug_info(true);
        //context.set_dump_everything(true);
        //context.set_keep_intermediates(true);

        {
            let cx = CodegenCx::new(&context, cgu, tcx);

            static START: Once = Once::new();
            START.call_once(|| {
                let initializer_name = format!("__gccGlobalCrateInit{}", tcx.crate_name(LOCAL_CRATE));
                let func = context.new_function(None, FunctionType::Exported, context.new_type::<()>(), &[], initializer_name, false);
                let block = func.new_block("initial");
                create_function_calling_initializers(tcx, &context, block);
                block.end_with_void_return(None);
            });

            //println!("module_codegen: {:?} {:?}", cgu_name, &cx.context as *const _);
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

            cx.global_init_block.end_with_void_return(None);
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

pub fn write_compressed_metadata<'tcx>(_tcx: TyCtxt<'tcx>, _metadata: &EncodedMetadata, _gcc_module: &mut GccContext) {
    // TODO

    /*use flate2::write::DeflateEncoder;
    use flate2::Compression;
    use std::io::Write;

    let (metadata_llcx, metadata_llmod) = (&*llvm_module.llcx, llvm_module.llmod());
    let mut compressed = tcx.metadata_encoding_version();
    DeflateEncoder::new(&mut compressed, Compression::fast())
        .write_all(&metadata.raw_data)
        .unwrap();

    let llmeta = common::bytes_in_context(metadata_llcx, &compressed);
    let llconst = common::struct_in_context(metadata_llcx, &[llmeta], false);
    let name = exported_symbols::metadata_symbol_name(tcx);
    let buf = CString::new(name).unwrap();
    let llglobal =
        unsafe { llvm::LLVMAddGlobal(metadata_llmod, common::val_ty(llconst), buf.as_ptr()) };
    unsafe {
        llvm::LLVMSetInitializer(llglobal, llconst);
        let section_name = metadata::metadata_section_name(&tcx.sess.target.target);
        let name = SmallCStr::new(section_name);
        llvm::LLVMSetSection(llglobal, name.as_ptr());

        // Also generate a .section directive to force no
        // flags, at least for ELF outputs, so that the
        // metadata doesn't get loaded into memory.
        let directive = format!(".section {}", section_name);
        llvm::LLVMSetModuleInlineAsm2(metadata_llmod, directive.as_ptr().cast(), directive.len())
    }*/
}
