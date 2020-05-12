//! Codegen the completed AST to the LLVM IR.
//!
//! Some functions here, such as codegen_block and codegen_expr, return a value --
//! the result of the codegen to LLVM -- while others, such as codegen_fn
//! and mono_item, are called only for the side effect of adding a
//! particular definition to the LLVM IR output we're producing.
//!
//! Hopefully useful general knowledge about codegen:
//!
//! * There's no way to find out the `Ty` type of a Value. Doing so
//!   would be "trying to get the eggs out of an omelette" (credit:
//!   pcwalton). You can, instead, find out its `llvm::Type` by calling `val_ty`,
//!   but one `llvm::Type` corresponds to many `Ty`s; for instance, `tup(int, int,
//!   int)` and `rec(x=int, y=int, z=int)` will have the same `llvm::Type`.

use super::ModuleLlvm;

use crate::attributes;
use crate::builder::Builder;
use crate::common;
use crate::context::CodegenCx;
use crate::llvm;
use crate::metadata;
use crate::value::Value;

use log::debug;
use rustc_codegen_ssa::base::maybe_create_entry_wrapper;
use rustc_codegen_ssa::mono_item::MonoItemExt;
use rustc_codegen_ssa::traits::*;
use rustc_codegen_ssa::{ModuleCodegen, ModuleKind};
use rustc_data_structures::small_c_str::SmallCStr;
use rustc_middle::dep_graph;
use rustc_middle::middle::codegen_fn_attrs::{CodegenFnAttrFlags, CodegenFnAttrs};
use rustc_middle::middle::cstore::EncodedMetadata;
use rustc_middle::middle::exported_symbols;
use rustc_middle::mir::mono::{Linkage, Visibility};
use rustc_middle::ty::TyCtxt;
use rustc_session::config::DebugInfo;
use rustc_span::symbol::Symbol;

use std::ffi::CString;
use std::time::Instant;

pub fn write_compressed_metadata<'tcx>(
    tcx: TyCtxt<'tcx>,
    metadata: &EncodedMetadata,
    llvm_module: &mut ModuleLlvm,
) {
    use flate2::write::DeflateEncoder;
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
    }
}

fn new_global<'ll>(
    name: &[&str],
    llmod: &'ll llvm::Module,
    llvalue: &'ll llvm::Value,
    linkage: llvm::Linkage,
    section: &str,
) -> &'ll llvm::Value {
    let name = CString::new(name.join(".")).unwrap();
    let section = SmallCStr::new(section);

    unsafe {
        let llglobal = llvm::LLVMAddGlobal(llmod, common::val_ty(llvalue), name.as_ptr());

        llvm::LLVMSetInitializer(llglobal, llvalue);
        llvm::LLVMRustSetLinkage(llglobal, linkage);
        llvm::LLVMSetSection(llglobal, section.as_ptr());

        llglobal
    }
}

unsafe fn get_rva<'ll>(
    llctx: &'ll llvm::Context,
    llptr: &'ll llvm::Value,
    llbase: &'ll llvm::Value,
) -> &'ll llvm::Value {
    let llrva_ty = llvm::LLVMInt32TypeInContext(llctx);

    let llbase_val = llvm::LLVMConstPtrToInt(llbase, llrva_ty);
    let llptr_val = llvm::LLVMConstPtrToInt(llptr, llrva_ty);

    llvm::LLVMConstSub(llptr_val, llbase_val)
}

pub fn write_idata_sections<'tcx>(
    _tcx: TyCtxt<'tcx>,
    raw_dylibs: &[RawDylibImports],
    llvm_module: &mut ModuleLlvm,
) {
    let (idata_llctx, idata_llmod) = (&*llvm_module.llcx, llvm_module.llmod());
    let llint32 = unsafe { llvm::LLVMInt32TypeInContext(idata_llctx) };
    let llbyte_ptr = unsafe { llvm::LLVMPointerType(llvm::LLVMInt8TypeInContext(idata_llctx), 0) };

    // import directory table types
    let lldir_ty = unsafe {
        let lldir_ty_name = SmallCStr::new(".win32.image_import_desc");
        let lldir_ty = llvm::LLVMStructCreateNamed(idata_llctx, lldir_ty_name.as_ptr());
        llvm::LLVMStructSetBody(
            lldir_ty,
            [llint32, llint32, llint32, llint32, llint32].as_ptr(),
            5,
            0,
        );

        lldir_ty
    };

    // image base constant for computing RVAs
    let image_base = unsafe {
        let llname = SmallCStr::new("__ImageBase");
        let llty = llvm::LLVMInt8TypeInContext(idata_llctx);

        let llglobal = llvm::LLVMAddGlobal(idata_llmod, llty, llname.as_ptr());
        llvm::LLVMRustSetLinkage(llglobal, llvm::Linkage::ExternalLinkage);

        llglobal
    };

    let mut dir_entries = vec![];

    for raw_dylib in raw_dylibs {
        debug!("creating raw dylib idata secions - {:?}", raw_dylib);

        let name = CString::new(&*raw_dylib.name.as_str()).unwrap();
        let llname = common::bytes_in_context(idata_llctx, name.as_bytes());

        let lldll_name = new_global(
            &["import", &*raw_dylib.name.as_str(), "dll_name"],
            idata_llmod,
            llname,
            llvm::Linkage::PrivateLinkage,
            "idata$7",
        );

        unsafe {
            llvm::LLVMSetGlobalConstant(&lldll_name, 1);

            let mut lookup_table = raw_dylib
                .items
                .iter()
                .map(|item| {
                    match item {
                        RawDylibImportName::Name(s) => {
                            let mut buf = vec![0, 0];
                            buf.extend(s.as_str().as_bytes());

                            if buf.len() % 2 == 1 {
                                buf.push(0);
                            }

                            let llname = common::bytes_in_context(idata_llctx, &buf);

                            let llglobal = new_global(
                                &["import", &*raw_dylib.name.as_str(), "fn", &*s.as_str()],
                                idata_llmod,
                                llname,
                                llvm::Linkage::PrivateLinkage,
                                "idata$6",
                            );

                            llvm::LLVMSetGlobalConstant(&llglobal, 1);
                            llvm::LLVMConstPointerCast(llglobal, llbyte_ptr)
                        }
                        RawDylibImportName::Ordinal(o) => {
                            //FIXME: support 32-bit targets
                            let o = *o as u64 | 0x8000_0000_0000_0000;
                            let llint64 = llvm::LLVMInt64TypeInContext(idata_llctx);
                            let llordinal = llvm::LLVMConstInt(llint64, o, 0);

                            llvm::LLVMConstIntToPtr(llordinal, llbyte_ptr)
                        }
                    }
                })
                .collect::<Vec<_>>();

            lookup_table.push(llvm::LLVMConstNull(llbyte_ptr));
            let lltable =
                llvm::LLVMConstArray(llbyte_ptr, lookup_table.as_ptr(), lookup_table.len() as u32);

            //import lookup table
            let ll_lookup_table = new_global(
                &["import", &*raw_dylib.name.as_str(), "desc"],
                idata_llmod,
                lltable,
                llvm::Linkage::PrivateLinkage,
                "idata$4",
            );

            //import address table - filled in at runtime
            let ll_addr_table = new_global(
                &["import", &*raw_dylib.name.as_str(), "ptr"],
                idata_llmod,
                lltable,
                llvm::Linkage::PrivateLinkage,
                "idata$3",
            );

            let llzero = llvm::LLVMConstInt(llint32, 0, 0);
            let lldir_entry = llvm::LLVMConstStructInContext(
                idata_llctx,
                [
                    get_rva(idata_llctx, ll_lookup_table, image_base),
                    llzero,
                    llzero,
                    get_rva(idata_llctx, lldll_name, image_base),
                    get_rva(idata_llctx, ll_addr_table, image_base),
                ]
                .as_ptr(),
                5,
                0,
            );

            dir_entries.push(lldir_entry);
        }
    }
    unsafe {
        dir_entries.push(llvm::LLVMConstNull(lldir_ty));
        let lldir_table =
            llvm::LLVMConstArray(lldir_ty, dir_entries.as_ptr(), dir_entries.len() as u32);

        let lldir_table = new_global(
            &[".dllimport"],
            idata_llmod,
            lldir_table,
            llvm::Linkage::ExternalLinkage,
            "idata$2",
        );
        llvm::LLVMSetGlobalConstant(&lldir_table, 1);
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
    let prof_timer = tcx.prof.generic_activity_with_arg("codegen_module", cgu_name.to_string());
    let start_time = Instant::now();

    let dep_node = tcx.codegen_unit(cgu_name).codegen_dep_node(tcx);
    let (module, _) =
        tcx.dep_graph.with_task(dep_node, tcx, cgu_name, module_codegen, dep_graph::hash_result);
    let time_to_codegen = start_time.elapsed();
    drop(prof_timer);

    // We assume that the cost to run LLVM on a CGU is proportional to
    // the time we needed for codegenning it.
    let cost = time_to_codegen.as_secs() * 1_000_000_000 + time_to_codegen.subsec_nanos() as u64;

    fn module_codegen(tcx: TyCtxt<'_>, cgu_name: Symbol) -> ModuleCodegen<ModuleLlvm> {
        let cgu = tcx.codegen_unit(cgu_name);
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
                attributes::sanitize(&cx, CodegenFnAttrFlags::empty(), entry);
            }

            // Run replace-all-uses-with for statics that need it
            for &(old_g, new_g) in cx.statics_to_rauw().borrow().iter() {
                unsafe {
                    let bitcast = llvm::LLVMConstPointerCast(new_g, cx.val_ty(old_g));
                    llvm::LLVMReplaceAllUsesWith(old_g, bitcast);
                    llvm::LLVMDeleteGlobal(old_g);
                }
            }

            // Create the llvm.used variable
            // This variable has type [N x i8*] and is stored in the llvm.metadata section
            if !cx.used_statics().borrow().is_empty() {
                cx.create_used_variable()
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
