// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use back::bytecode::{DecodedBytecode, RLIB_BYTECODE_EXTENSION};
use back::write;
use back::symbol_export;
use rustc::session::config;
use errors::{FatalError, Handler};
use llvm;
use llvm::archive_ro::ArchiveRO;
use llvm::{ModuleRef, TargetMachineRef, True, False};
use rustc::middle::exported_symbols::SymbolExportLevel;
use rustc::util::common::time;
use rustc::hir::def_id::LOCAL_CRATE;
use back::write::{ModuleConfig, with_llvm_pmb, CodegenContext};
use {ModuleTranslation, ModuleKind};

use libc;

use std::ffi::CString;
use std::slice;

pub fn crate_type_allows_lto(crate_type: config::CrateType) -> bool {
    match crate_type {
        config::CrateTypeExecutable |
        config::CrateTypeStaticlib  |
        config::CrateTypeCdylib     => true,

        config::CrateTypeDylib     |
        config::CrateTypeRlib      |
        config::CrateTypeProcMacro => false,
    }
}

pub enum LtoModuleTranslation {
    Fat {
        module: Option<ModuleTranslation>,
        _serialized_bitcode: Vec<SerializedModule>,
    },

    // Note the lack of other entries in this enum! Ideally one day this gap is
    // intended to be filled with a "Thin" LTO variant.
}

impl LtoModuleTranslation {
    pub fn name(&self) -> &str {
        match *self {
            LtoModuleTranslation::Fat { .. } => "everything",
        }
    }

    /// Optimize this module within the given codegen context.
    ///
    /// This function is unsafe as it'll return a `ModuleTranslation` still
    /// points to LLVM data structures owned by this `LtoModuleTranslation`.
    /// It's intended that the module returned is immediately code generated and
    /// dropped, and then this LTO module is dropped.
    pub unsafe fn optimize(&mut self, cgcx: &CodegenContext)
        -> Result<ModuleTranslation, FatalError>
    {
        match *self {
            LtoModuleTranslation::Fat { ref mut module, .. } => {
                let trans = module.take().unwrap();
                let config = cgcx.config(trans.kind);
                let llmod = trans.llvm().unwrap().llmod;
                let tm = trans.llvm().unwrap().tm;
                run_pass_manager(cgcx, tm, llmod, config);
                Ok(trans)
            }
        }
    }

    /// A "guage" of how costly it is to optimize this module, used to sort
    /// biggest modules first.
    pub fn cost(&self) -> u64 {
        match *self {
            // Only one module with fat LTO, so the cost doesn't matter.
            LtoModuleTranslation::Fat { .. } => 0,
        }
    }
}

pub fn run(cgcx: &CodegenContext, modules: Vec<ModuleTranslation>)
    -> Result<Vec<LtoModuleTranslation>, FatalError>
{
    let diag_handler = cgcx.create_diag_handler();
    if cgcx.opts.cg.prefer_dynamic {
        diag_handler.struct_err("cannot prefer dynamic linking when performing LTO")
                    .note("only 'staticlib', 'bin', and 'cdylib' outputs are \
                           supported with LTO")
                    .emit();
        return Err(FatalError)
    }

    // Make sure we actually can run LTO
    for crate_type in cgcx.crate_types.iter() {
        if !crate_type_allows_lto(*crate_type) {
            let e = diag_handler.fatal("lto can only be run for executables, cdylibs and \
                                        static library outputs");
            return Err(e)
        }
    }

    let export_threshold =
        symbol_export::crates_export_threshold(&cgcx.crate_types);

    let symbol_filter = &|&(ref name, _, level): &(String, _, SymbolExportLevel)| {
        if level.is_below_threshold(export_threshold) {
            let mut bytes = Vec::with_capacity(name.len() + 1);
            bytes.extend(name.bytes());
            Some(CString::new(bytes).unwrap())
        } else {
            None
        }
    };

    let mut symbol_white_list: Vec<CString> = cgcx.exported_symbols[&LOCAL_CRATE]
        .iter()
        .filter_map(symbol_filter)
        .collect();
    info!("{} symbols in whitelist", symbol_white_list.len());

    // For each of our upstream dependencies, find the corresponding rlib and
    // load the bitcode from the archive. Then merge it into the current LLVM
    // module that we've got.
    let mut upstream_modules = Vec::new();
    for &(cnum, ref path) in cgcx.each_linked_rlib_for_lto.iter() {
        symbol_white_list.extend(
            cgcx.exported_symbols[&cnum]
                .iter()
                .filter_map(symbol_filter));
        info!("{} symbols in whitelist after {}", symbol_white_list.len(), cnum);

        let archive = ArchiveRO::open(&path).expect("wanted an rlib");
        let bytecodes = archive.iter().filter_map(|child| {
            child.ok().and_then(|c| c.name().map(|name| (name, c)))
        }).filter(|&(name, _)| name.ends_with(RLIB_BYTECODE_EXTENSION));
        for (name, data) in bytecodes {
            info!("adding bytecode {}", name);
            let bc_encoded = data.data();

            let (bc, id) = time(cgcx.time_passes, &format!("decode {}", name), || {
                match DecodedBytecode::new(bc_encoded) {
                    Ok(b) => Ok((b.bytecode(), b.identifier().to_string())),
                    Err(e) => Err(diag_handler.fatal(&e)),
                }
            })?;
            let bc = SerializedModule::FromRlib(bc);
            upstream_modules.push((bc, CString::new(id).unwrap()));
        }
    }

    // Internalize everything but the exported symbols of the current module
    let arr: Vec<*const libc::c_char> = symbol_white_list.iter()
                                                         .map(|c| c.as_ptr())
                                                         .collect();

    fat_lto(cgcx, &diag_handler, modules, upstream_modules, &arr)
}

fn fat_lto(cgcx: &CodegenContext,
           diag_handler: &Handler,
           mut modules: Vec<ModuleTranslation>,
           mut serialized_modules: Vec<(SerializedModule, CString)>,
           symbol_white_list: &[*const libc::c_char])
    -> Result<Vec<LtoModuleTranslation>, FatalError>
{
    info!("going for a fat lto");

    // Find the "costliest" module and merge everything into that codegen unit.
    // All the other modules will be serialized and reparsed into the new
    // context, so this hopefully avoids serializing and parsing the largest
    // codegen unit.
    //
    // Additionally use a regular module as the base here to ensure that various
    // file copy operations in the backend work correctly. The only other kind
    // of module here should be an allocator one, and if your crate is smaller
    // than the allocator module then the size doesn't really matter anyway.
    let (_, costliest_module) = modules.iter()
        .enumerate()
        .filter(|&(_, module)| module.kind == ModuleKind::Regular)
        .map(|(i, module)| {
            let cost = unsafe {
                llvm::LLVMRustModuleCost(module.llvm().unwrap().llmod)
            };
            (cost, i)
        })
        .max()
        .expect("must be trans'ing at least one module");
    let module = modules.remove(costliest_module);
    let llmod = module.llvm().expect("can't lto pre-translated modules").llmod;
    info!("using {:?} as a base module", module.llmod_id);

    // For all other modules we translated we'll need to link them into our own
    // bitcode. All modules were translated in their own LLVM context, however,
    // and we want to move everything to the same LLVM context. Currently the
    // way we know of to do that is to serialize them to a string and them parse
    // them later. Not great but hey, that's why it's "fat" LTO, right?
    for module in modules {
        let llvm = module.llvm().expect("can't lto pre-translated modules");
        let buffer = ModuleBuffer::new(llvm.llmod);
        let llmod_id = CString::new(&module.llmod_id[..]).unwrap();
        serialized_modules.push((SerializedModule::Local(buffer), llmod_id));
    }

    // For all serialized bitcode files we parse them and link them in as we did
    // above, this is all mostly handled in C++. Like above, though, we don't
    // know much about the memory management here so we err on the side of being
    // save and persist everything with the original module.
    let mut serialized_bitcode = Vec::new();
    for (bc_decoded, name) in serialized_modules {
        info!("linking {:?}", name);
        time(cgcx.time_passes, &format!("ll link {:?}", name), || unsafe {
            let data = bc_decoded.data();
            if llvm::LLVMRustLinkInExternalBitcode(llmod,
                                                   data.as_ptr() as *const libc::c_char,
                                                   data.len() as libc::size_t) {
                Ok(())
            } else {
                let msg = format!("failed to load bc of {:?}", name);
                Err(write::llvm_err(&diag_handler, msg))
            }
        })?;
        serialized_bitcode.push(bc_decoded);
    }
    cgcx.save_temp_bitcode(&module, "lto.input");

    // Internalize everything that *isn't* in our whitelist to help strip out
    // more modules and such
    unsafe {
        let ptr = symbol_white_list.as_ptr();
        llvm::LLVMRustRunRestrictionPass(llmod,
                                         ptr as *const *const libc::c_char,
                                         symbol_white_list.len() as libc::size_t);
        cgcx.save_temp_bitcode(&module, "lto.after-restriction");
    }

    if cgcx.no_landing_pads {
        unsafe {
            llvm::LLVMRustMarkAllFunctionsNounwind(llmod);
        }
        cgcx.save_temp_bitcode(&module, "lto.after-nounwind");
    }

    Ok(vec![LtoModuleTranslation::Fat {
        module: Some(module),
        _serialized_bitcode: serialized_bitcode,
    }])
}

fn run_pass_manager(cgcx: &CodegenContext,
                    tm: TargetMachineRef,
                    llmod: ModuleRef,
                    config: &ModuleConfig) {

    // Now we have one massive module inside of llmod. Time to run the
    // LTO-specific optimization passes that LLVM provides.
    //
    // This code is based off the code found in llvm's LTO code generator:
    //      tools/lto/LTOCodeGenerator.cpp
    debug!("running the pass manager");
    unsafe {
        let pm = llvm::LLVMCreatePassManager();
        llvm::LLVMRustAddAnalysisPasses(tm, pm, llmod);
        let pass = llvm::LLVMRustFindAndCreatePass("verify\0".as_ptr() as *const _);
        assert!(!pass.is_null());
        llvm::LLVMRustAddPass(pm, pass);

        with_llvm_pmb(llmod, config, &mut |b| {
            llvm::LLVMPassManagerBuilderPopulateLTOPassManager(b, pm,
                /* Internalize = */ False,
                /* RunInliner = */ True);
        });

        let pass = llvm::LLVMRustFindAndCreatePass("verify\0".as_ptr() as *const _);
        assert!(!pass.is_null());
        llvm::LLVMRustAddPass(pm, pass);

        time(cgcx.time_passes, "LTO passes", ||
             llvm::LLVMRunPassManager(pm, llmod));

        llvm::LLVMDisposePassManager(pm);
    }
    debug!("lto done");
}

pub enum SerializedModule {
    Local(ModuleBuffer),
    FromRlib(Vec<u8>),
}

impl SerializedModule {
    fn data(&self) -> &[u8] {
        match *self {
            SerializedModule::Local(ref m) => m.data(),
            SerializedModule::FromRlib(ref m) => m,
        }
    }
}

pub struct ModuleBuffer(*mut llvm::ModuleBuffer);

unsafe impl Send for ModuleBuffer {}
unsafe impl Sync for ModuleBuffer {}

impl ModuleBuffer {
    fn new(m: ModuleRef) -> ModuleBuffer {
        ModuleBuffer(unsafe {
            llvm::LLVMRustModuleBufferCreate(m)
        })
    }

    fn data(&self) -> &[u8] {
        unsafe {
            let ptr = llvm::LLVMRustModuleBufferPtr(self.0);
            let len = llvm::LLVMRustModuleBufferLen(self.0);
            slice::from_raw_parts(ptr, len)
        }
    }
}

impl Drop for ModuleBuffer {
    fn drop(&mut self) {
        unsafe { llvm::LLVMRustModuleBufferFree(self.0); }
    }
}
