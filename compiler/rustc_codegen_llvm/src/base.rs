//! Codegen the MIR to the LLVM IR.
//!
//! Hopefully useful general knowledge about codegen:
//!
//! * There's no way to find out the [`Ty`] type of a [`Value`]. Doing so
//!   would be "trying to get the eggs out of an omelette" (credit:
//!   pcwalton). You can, instead, find out its [`llvm::Type`] by calling [`val_ty`],
//!   but one [`llvm::Type`] corresponds to many [`Ty`]s; for instance, `tup(int, int,
//!   int)` and `rec(x=int, y=int, z=int)` will have the same [`llvm::Type`].
//!
//! [`Ty`]: rustc_middle::ty::Ty
//! [`val_ty`]: common::val_ty

use super::ModuleLlvm;

use crate::attributes;
use crate::builder::Builder;
use crate::common;
use crate::context::CodegenCx;
use crate::llvm;
use crate::value::Value;

use rustc_codegen_ssa::base::maybe_create_entry_wrapper;
use rustc_codegen_ssa::mono_item::MonoItemExt;
use rustc_codegen_ssa::traits::*;
use rustc_codegen_ssa::{ModuleCodegen, ModuleKind};
use rustc_data_structures::small_c_str::SmallCStr;
use rustc_metadata::EncodedMetadata;
use rustc_middle::dep_graph;
use rustc_middle::middle::codegen_fn_attrs::CodegenFnAttrs;
use rustc_middle::middle::exported_symbols;
use rustc_middle::mir::mono::{Linkage, Visibility};
use rustc_middle::ty::TyCtxt;
use rustc_session::config::DebugInfo;
use rustc_span::symbol::Symbol;
use rustc_target::spec::SanitizerSet;

use std::ffi::CString;
use std::time::Instant;

pub fn write_compressed_metadata<'tcx>(
    tcx: TyCtxt<'tcx>,
    metadata: &EncodedMetadata,
    llvm_module: &mut ModuleLlvm,
) {
    use snap::write::FrameEncoder;
    use std::io::Write;

    // Historical note:
    //
    // When using link.exe it was seen that the section name `.note.rustc`
    // was getting shortened to `.note.ru`, and according to the PE and COFF
    // specification:
    //
    // > Executable images do not use a string table and do not support
    // > section names longer than 8 characters
    //
    // https://docs.microsoft.com/en-us/windows/win32/debug/pe-format
    //
    // As a result, we choose a slightly shorter name! As to why
    // `.note.rustc` works on MinGW, see
    // https://github.com/llvm/llvm-project/blob/llvmorg-12.0.0/lld/COFF/Writer.cpp#L1190-L1197
    let section_name = if tcx.sess.target.is_like_osx { "__DATA,.rustc" } else { ".rustc" };

    let (metadata_llcx, metadata_llmod) = (&*llvm_module.llcx, llvm_module.llmod());
    let mut compressed = rustc_metadata::METADATA_HEADER.to_vec();
    FrameEncoder::new(&mut compressed).write_all(metadata.raw_data()).unwrap();

    let llmeta = common::bytes_in_context(metadata_llcx, &compressed);
    let llconst = common::struct_in_context(metadata_llcx, &[llmeta], false);
    let name = exported_symbols::metadata_symbol_name(tcx);
    let buf = CString::new(name).unwrap();
    let llglobal =
        unsafe { llvm::LLVMAddGlobal(metadata_llmod, common::val_ty(llconst), buf.as_ptr()) };
    unsafe {
        llvm::LLVMSetInitializer(llglobal, llconst);
        let name = SmallCStr::new(section_name);
        llvm::LLVMSetSection(llglobal, name.as_ptr());

        // Also generate a .section directive to force no
        // flags, at least for ELF outputs, so that the
        // metadata doesn't get loaded into memory.
        let directive = format!(".section {}", section_name);
        llvm::LLVMSetModuleInlineAsm2(metadata_llmod, directive.as_ptr().cast(), directive.len())
    }
}

pub struct ValueIter<'ll> {
    cur: Option<&'ll Value>,
    step: unsafe extern "C" fn(&'ll Value) -> Option<&'ll Value>,
}

impl Iterator for ValueIter<'ll> {
    type Item = &'ll Value;

    fn next(&mut self) -> Option<&'ll Value> {
        let old = self.cur;
        if let Some(old) = old {
            self.cur = unsafe { (self.step)(old) };
        }
        old
    }
}

pub fn iter_globals(llmod: &'ll llvm::Module) -> ValueIter<'ll> {
    unsafe { ValueIter { cur: llvm::LLVMGetFirstGlobal(llmod), step: llvm::LLVMGetNextGlobal } }
}

pub fn compile_codegen_unit(
    tcx: TyCtxt<'tcx>,
    cgu_name: Symbol,
) -> (ModuleCodegen<ModuleLlvm>, u64) {
    let start_time = Instant::now();

    let dep_node = tcx.codegen_unit(cgu_name).codegen_dep_node(tcx);
    let (module, _) =
        tcx.dep_graph.with_task(dep_node, tcx, cgu_name, module_codegen, dep_graph::hash_result);
    let time_to_codegen = start_time.elapsed();

    // We assume that the cost to run LLVM on a CGU is proportional to
    // the time we needed for codegenning it.
    let cost = time_to_codegen.as_nanos() as u64;

    fn module_codegen(tcx: TyCtxt<'_>, cgu_name: Symbol) -> ModuleCodegen<ModuleLlvm> {
        let cgu = tcx.codegen_unit(cgu_name);
        let _prof_timer = tcx.prof.generic_activity_with_args(
            "codegen_module",
            &[cgu_name.to_string(), cgu.size_estimate().to_string()],
        );
        // Instantiate monomorphizations without filling out definitions yet...
        let llvm_module = ModuleLlvm::new(tcx, &cgu_name.as_str());
        {
            let cx = CodegenCx::new(tcx, cgu, &llvm_module);
            let mono_items = cx.codegen_unit.items_in_deterministic_order(cx.tcx);
            for &(mono_item, (linkage, visibility)) in &mono_items {
                mono_item.predefine::<Builder<'_, '_, '_>>(&cx, linkage, visibility);
            }

            // ... and now that we have everything pre-defined, fill out those definitions.
            for &(mono_item, _) in &mono_items {
                mono_item.define::<Builder<'_, '_, '_>>(&cx);
            }

            // If this codegen unit contains the main function, also create the
            // wrapper here
            if let Some(entry) = maybe_create_entry_wrapper::<Builder<'_, '_, '_>>(&cx) {
                attributes::sanitize(&cx, SanitizerSet::empty(), entry);
            }

            // Run replace-all-uses-with for statics that need it
            for &(old_g, new_g) in cx.statics_to_rauw().borrow().iter() {
                unsafe {
                    let bitcast = llvm::LLVMConstPointerCast(new_g, cx.val_ty(old_g));
                    llvm::LLVMReplaceAllUsesWith(old_g, bitcast);
                    llvm::LLVMDeleteGlobal(old_g);
                }
            }

            // Finalize code coverage by injecting the coverage map. Note, the coverage map will
            // also be added to the `llvm.compiler.used` variable, created next.
            if cx.sess().instrument_coverage() {
                cx.coverageinfo_finalize();
            }

            // Create the llvm.used and llvm.compiler.used variables.
            if !cx.used_statics().borrow().is_empty() {
                cx.create_used_variable()
            }
            if !cx.compiler_used_statics().borrow().is_empty() {
                cx.create_compiler_used_variable()
            }

            // Finalize debuginfo
            if cx.sess().opts.debuginfo != DebugInfo::None {
                cx.debuginfo_finalize();
            }
        }

        ModuleCodegen {
            name: cgu_name.to_string(),
            module_llvm: llvm_module,
            kind: ModuleKind::Regular,
        }
    }

    (module, cost)
}

pub fn set_link_section(llval: &Value, attrs: &CodegenFnAttrs) {
    let sect = match attrs.link_section {
        Some(name) => name,
        None => return,
    };
    unsafe {
        let buf = SmallCStr::new(&sect.as_str());
        llvm::LLVMSetSection(llval, buf.as_ptr());
    }
}

pub fn linkage_to_llvm(linkage: Linkage) -> llvm::Linkage {
    match linkage {
        Linkage::External => llvm::Linkage::ExternalLinkage,
        Linkage::AvailableExternally => llvm::Linkage::AvailableExternallyLinkage,
        Linkage::LinkOnceAny => llvm::Linkage::LinkOnceAnyLinkage,
        Linkage::LinkOnceODR => llvm::Linkage::LinkOnceODRLinkage,
        Linkage::WeakAny => llvm::Linkage::WeakAnyLinkage,
        Linkage::WeakODR => llvm::Linkage::WeakODRLinkage,
        Linkage::Appending => llvm::Linkage::AppendingLinkage,
        Linkage::Internal => llvm::Linkage::InternalLinkage,
        Linkage::Private => llvm::Linkage::PrivateLinkage,
        Linkage::ExternalWeak => llvm::Linkage::ExternalWeakLinkage,
        Linkage::Common => llvm::Linkage::CommonLinkage,
    }
}

pub fn visibility_to_llvm(linkage: Visibility) -> llvm::Visibility {
    match linkage {
        Visibility::Default => llvm::Visibility::Default,
        Visibility::Hidden => llvm::Visibility::Hidden,
        Visibility::Protected => llvm::Visibility::Protected,
    }
}
