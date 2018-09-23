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
use back::symbol_export;
use back::write::{ModuleConfig, with_llvm_pmb, CodegenContext};
use back::write::{self, DiagnosticHandlers, pre_lto_bitcode_filename};
use errors::{FatalError, Handler};
use llvm::archive_ro::ArchiveRO;
use llvm::{True, False};
use llvm;
use memmap;
use rustc::dep_graph::WorkProduct;
use rustc::dep_graph::cgu_reuse_tracker::CguReuse;
use rustc::hir::def_id::LOCAL_CRATE;
use rustc::middle::exported_symbols::SymbolExportLevel;
use rustc::session::config::{self, Lto};
use rustc::util::common::time_ext;
use rustc_data_structures::fx::FxHashMap;
use time_graph::Timeline;
use {ModuleCodegen, ModuleLlvm, ModuleKind};

use libc;

use std::ffi::{CStr, CString};
use std::fs;
use std::ptr;
use std::slice;
use std::sync::Arc;

pub fn crate_type_allows_lto(crate_type: config::CrateType) -> bool {
    match crate_type {
        config::CrateType::Executable |
        config::CrateType::Staticlib  |
        config::CrateType::Cdylib     => true,

        config::CrateType::Dylib     |
        config::CrateType::Rlib      |
        config::CrateType::ProcMacro => false,
    }
}

pub(crate) enum LtoModuleCodegen {
    Fat {
        module: Option<ModuleCodegen>,
        _serialized_bitcode: Vec<SerializedModule>,
    },

    Thin(ThinModule),
}

impl LtoModuleCodegen {
    pub fn name(&self) -> &str {
        match *self {
            LtoModuleCodegen::Fat { .. } => "everything",
            LtoModuleCodegen::Thin(ref m) => m.name(),
        }
    }

    /// Optimize this module within the given codegen context.
    ///
    /// This function is unsafe as it'll return a `ModuleCodegen` still
    /// points to LLVM data structures owned by this `LtoModuleCodegen`.
    /// It's intended that the module returned is immediately code generated and
    /// dropped, and then this LTO module is dropped.
    pub(crate) unsafe fn optimize(&mut self,
                                  cgcx: &CodegenContext,
                                  timeline: &mut Timeline)
        -> Result<ModuleCodegen, FatalError>
    {
        match *self {
            LtoModuleCodegen::Fat { ref mut module, .. } => {
                let module = module.take().unwrap();
                {
                    let config = cgcx.config(module.kind);
                    let llmod = module.module_llvm.llmod();
                    let tm = &*module.module_llvm.tm;
                    run_pass_manager(cgcx, tm, llmod, config, false);
                    timeline.record("fat-done");
                }
                Ok(module)
            }
            LtoModuleCodegen::Thin(ref mut thin) => thin.optimize(cgcx, timeline),
        }
    }

    /// A "gauge" of how costly it is to optimize this module, used to sort
    /// biggest modules first.
    pub fn cost(&self) -> u64 {
        match *self {
            // Only one module with fat LTO, so the cost doesn't matter.
            LtoModuleCodegen::Fat { .. } => 0,
            LtoModuleCodegen::Thin(ref m) => m.cost(),
        }
    }
}

/// Performs LTO, which in the case of full LTO means merging all modules into
/// a single one and returning it for further optimizing. For ThinLTO, it will
/// do the global analysis necessary and return two lists, one of the modules
/// the need optimization and another for modules that can simply be copied over
/// from the incr. comp. cache.
pub(crate) fn run(cgcx: &CodegenContext,
                  modules: Vec<ModuleCodegen>,
                  cached_modules: Vec<(SerializedModule, WorkProduct)>,
                  timeline: &mut Timeline)
    -> Result<(Vec<LtoModuleCodegen>, Vec<WorkProduct>), FatalError>
{
    let diag_handler = cgcx.create_diag_handler();
    let export_threshold = match cgcx.lto {
        // We're just doing LTO for our one crate
        Lto::ThinLocal => SymbolExportLevel::Rust,

        // We're doing LTO for the entire crate graph
        Lto::Fat | Lto::Thin => {
            symbol_export::crates_export_threshold(&cgcx.crate_types)
        }

        Lto::No => panic!("didn't request LTO but we're doing LTO"),
    };

    let symbol_filter = &|&(ref name, level): &(String, SymbolExportLevel)| {
        if level.is_below_threshold(export_threshold) {
            let mut bytes = Vec::with_capacity(name.len() + 1);
            bytes.extend(name.bytes());
            Some(CString::new(bytes).unwrap())
        } else {
            None
        }
    };
    let exported_symbols = cgcx.exported_symbols
        .as_ref().expect("needs exported symbols for LTO");
    let mut symbol_white_list = exported_symbols[&LOCAL_CRATE]
        .iter()
        .filter_map(symbol_filter)
        .collect::<Vec<CString>>();
    timeline.record("whitelist");
    info!("{} symbols to preserve in this crate", symbol_white_list.len());

    // If we're performing LTO for the entire crate graph, then for each of our
    // upstream dependencies, find the corresponding rlib and load the bitcode
    // from the archive.
    //
    // We save off all the bytecode and LLVM module ids for later processing
    // with either fat or thin LTO
    let mut upstream_modules = Vec::new();
    if cgcx.lto != Lto::ThinLocal {
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

        for &(cnum, ref path) in cgcx.each_linked_rlib_for_lto.iter() {
            let exported_symbols = cgcx.exported_symbols
                .as_ref().expect("needs exported symbols for LTO");
            symbol_white_list.extend(
                exported_symbols[&cnum]
                    .iter()
                    .filter_map(symbol_filter));

            let archive = ArchiveRO::open(&path).expect("wanted an rlib");
            let bytecodes = archive.iter().filter_map(|child| {
                child.ok().and_then(|c| c.name().map(|name| (name, c)))
            }).filter(|&(name, _)| name.ends_with(RLIB_BYTECODE_EXTENSION));
            for (name, data) in bytecodes {
                info!("adding bytecode {}", name);
                let bc_encoded = data.data();

                let (bc, id) = time_ext(cgcx.time_passes, None, &format!("decode {}", name), || {
                    match DecodedBytecode::new(bc_encoded) {
                        Ok(b) => Ok((b.bytecode(), b.identifier().to_string())),
                        Err(e) => Err(diag_handler.fatal(&e)),
                    }
                })?;
                let bc = SerializedModule::FromRlib(bc);
                upstream_modules.push((bc, CString::new(id).unwrap()));
            }
            timeline.record(&format!("load: {}", path.display()));
        }
    }

    let symbol_white_list = symbol_white_list.iter()
                                             .map(|c| c.as_ptr())
                                             .collect::<Vec<_>>();
    match cgcx.lto {
        Lto::Fat => {
            assert!(cached_modules.is_empty());
            let opt_jobs = fat_lto(cgcx,
                                  &diag_handler,
                                  modules,
                                  upstream_modules,
                                  &symbol_white_list,
                                  timeline);
            opt_jobs.map(|opt_jobs| (opt_jobs, vec![]))
        }
        Lto::Thin |
        Lto::ThinLocal => {
            if cgcx.opts.debugging_opts.cross_lang_lto.enabled() {
                unreachable!("We should never reach this case if the LTO step \
                              is deferred to the linker");
            }
            thin_lto(cgcx,
                     &diag_handler,
                     modules,
                     upstream_modules,
                     cached_modules,
                     &symbol_white_list,
                     timeline)
        }
        Lto::No => unreachable!(),
    }
}

fn fat_lto(cgcx: &CodegenContext,
           diag_handler: &Handler,
           mut modules: Vec<ModuleCodegen>,
           mut serialized_modules: Vec<(SerializedModule, CString)>,
           symbol_white_list: &[*const libc::c_char],
           timeline: &mut Timeline)
    -> Result<Vec<LtoModuleCodegen>, FatalError>
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
                llvm::LLVMRustModuleCost(module.module_llvm.llmod())
            };
            (cost, i)
        })
        .max()
        .expect("must be codegen'ing at least one module");
    let module = modules.remove(costliest_module);
    let mut serialized_bitcode = Vec::new();
    {
        let (llcx, llmod) = {
            let llvm = &module.module_llvm;
            (&llvm.llcx, llvm.llmod())
        };
        info!("using {:?} as a base module", module.name);

        // The linking steps below may produce errors and diagnostics within LLVM
        // which we'd like to handle and print, so set up our diagnostic handlers
        // (which get unregistered when they go out of scope below).
        let _handler = DiagnosticHandlers::new(cgcx, diag_handler, llcx);

        // For all other modules we codegened we'll need to link them into our own
        // bitcode. All modules were codegened in their own LLVM context, however,
        // and we want to move everything to the same LLVM context. Currently the
        // way we know of to do that is to serialize them to a string and them parse
        // them later. Not great but hey, that's why it's "fat" LTO, right?
        for module in modules {
            let buffer = ModuleBuffer::new(module.module_llvm.llmod());
            let llmod_id = CString::new(&module.name[..]).unwrap();
            serialized_modules.push((SerializedModule::Local(buffer), llmod_id));
        }

        // For all serialized bitcode files we parse them and link them in as we did
        // above, this is all mostly handled in C++. Like above, though, we don't
        // know much about the memory management here so we err on the side of being
        // save and persist everything with the original module.
        let mut linker = Linker::new(llmod);
        for (bc_decoded, name) in serialized_modules {
            info!("linking {:?}", name);
            time_ext(cgcx.time_passes, None, &format!("ll link {:?}", name), || {
                let data = bc_decoded.data();
                linker.add(&data).map_err(|()| {
                    let msg = format!("failed to load bc of {:?}", name);
                    write::llvm_err(&diag_handler, msg)
                })
            })?;
            timeline.record(&format!("link {:?}", name));
            serialized_bitcode.push(bc_decoded);
        }
        drop(linker);
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
        timeline.record("passes");
    }

    Ok(vec![LtoModuleCodegen::Fat {
        module: Some(module),
        _serialized_bitcode: serialized_bitcode,
    }])
}

struct Linker<'a>(&'a mut llvm::Linker<'a>);

impl Linker<'a> {
    fn new(llmod: &'a llvm::Module) -> Self {
        unsafe { Linker(llvm::LLVMRustLinkerNew(llmod)) }
    }

    fn add(&mut self, bytecode: &[u8]) -> Result<(), ()> {
        unsafe {
            if llvm::LLVMRustLinkerAdd(self.0,
                                       bytecode.as_ptr() as *const libc::c_char,
                                       bytecode.len()) {
                Ok(())
            } else {
                Err(())
            }
        }
    }
}

impl Drop for Linker<'a> {
    fn drop(&mut self) {
        unsafe { llvm::LLVMRustLinkerFree(&mut *(self.0 as *mut _)); }
    }
}

/// Prepare "thin" LTO to get run on these modules.
///
/// The general structure of ThinLTO is quite different from the structure of
/// "fat" LTO above. With "fat" LTO all LLVM modules in question are merged into
/// one giant LLVM module, and then we run more optimization passes over this
/// big module after internalizing most symbols. Thin LTO, on the other hand,
/// avoid this large bottleneck through more targeted optimization.
///
/// At a high level Thin LTO looks like:
///
///     1. Prepare a "summary" of each LLVM module in question which describes
///        the values inside, cost of the values, etc.
///     2. Merge the summaries of all modules in question into one "index"
///     3. Perform some global analysis on this index
///     4. For each module, use the index and analysis calculated previously to
///        perform local transformations on the module, for example inlining
///        small functions from other modules.
///     5. Run thin-specific optimization passes over each module, and then code
///        generate everything at the end.
///
/// The summary for each module is intended to be quite cheap, and the global
/// index is relatively quite cheap to create as well. As a result, the goal of
/// ThinLTO is to reduce the bottleneck on LTO and enable LTO to be used in more
/// situations. For example one cheap optimization is that we can parallelize
/// all codegen modules, easily making use of all the cores on a machine.
///
/// With all that in mind, the function here is designed at specifically just
/// calculating the *index* for ThinLTO. This index will then be shared amongst
/// all of the `LtoModuleCodegen` units returned below and destroyed once
/// they all go out of scope.
fn thin_lto(cgcx: &CodegenContext,
            diag_handler: &Handler,
            modules: Vec<ModuleCodegen>,
            serialized_modules: Vec<(SerializedModule, CString)>,
            cached_modules: Vec<(SerializedModule, WorkProduct)>,
            symbol_white_list: &[*const libc::c_char],
            timeline: &mut Timeline)
    -> Result<(Vec<LtoModuleCodegen>, Vec<WorkProduct>), FatalError>
{
    unsafe {
        info!("going for that thin, thin LTO");

        let green_modules: FxHashMap<_, _> = cached_modules
            .iter()
            .map(|&(_, ref wp)| (wp.cgu_name.clone(), wp.clone()))
            .collect();

        let mut thin_buffers = Vec::new();
        let mut module_names = Vec::new();
        let mut thin_modules = Vec::new();

        // FIXME: right now, like with fat LTO, we serialize all in-memory
        //        modules before working with them and ThinLTO. We really
        //        shouldn't do this, however, and instead figure out how to
        //        extract a summary from an in-memory module and then merge that
        //        into the global index. It turns out that this loop is by far
        //        the most expensive portion of this small bit of global
        //        analysis!
        for (i, module) in modules.iter().enumerate() {
            info!("local module: {} - {}", i, module.name);
            let name = CString::new(module.name.clone()).unwrap();
            let buffer = ThinBuffer::new(module.module_llvm.llmod());

            // We emit the module after having serialized it into a ThinBuffer
            // because only then it will contain the ThinLTO module summary.
            if let Some(ref incr_comp_session_dir) = cgcx.incr_comp_session_dir {
                if cgcx.config(module.kind).emit_pre_thin_lto_bc {
                    let path = incr_comp_session_dir
                        .join(pre_lto_bitcode_filename(&module.name));

                    fs::write(&path, buffer.data()).unwrap_or_else(|e| {
                        panic!("Error writing pre-lto-bitcode file `{}`: {}",
                               path.display(),
                               e);
                    });
                }
            }

            thin_modules.push(llvm::ThinLTOModule {
                identifier: name.as_ptr(),
                data: buffer.data().as_ptr(),
                len: buffer.data().len(),
            });
            thin_buffers.push(buffer);
            module_names.push(name);
            timeline.record(&module.name);
        }

        // FIXME: All upstream crates are deserialized internally in the
        //        function below to extract their summary and modules. Note that
        //        unlike the loop above we *must* decode and/or read something
        //        here as these are all just serialized files on disk. An
        //        improvement, however, to make here would be to store the
        //        module summary separately from the actual module itself. Right
        //        now this is store in one large bitcode file, and the entire
        //        file is deflate-compressed. We could try to bypass some of the
        //        decompression by storing the index uncompressed and only
        //        lazily decompressing the bytecode if necessary.
        //
        //        Note that truly taking advantage of this optimization will
        //        likely be further down the road. We'd have to implement
        //        incremental ThinLTO first where we could actually avoid
        //        looking at upstream modules entirely sometimes (the contents,
        //        we must always unconditionally look at the index).
        let mut serialized = Vec::new();

        let cached_modules = cached_modules.into_iter().map(|(sm, wp)| {
            (sm, CString::new(wp.cgu_name).unwrap())
        });

        for (module, name) in serialized_modules.into_iter().chain(cached_modules) {
            info!("upstream or cached module {:?}", name);
            thin_modules.push(llvm::ThinLTOModule {
                identifier: name.as_ptr(),
                data: module.data().as_ptr(),
                len: module.data().len(),
            });
            serialized.push(module);
            module_names.push(name);
        }

        // Sanity check
        assert_eq!(thin_modules.len(), module_names.len());

        // Delegate to the C++ bindings to create some data here. Once this is a
        // tried-and-true interface we may wish to try to upstream some of this
        // to LLVM itself, right now we reimplement a lot of what they do
        // upstream...
        let data = llvm::LLVMRustCreateThinLTOData(
            thin_modules.as_ptr(),
            thin_modules.len() as u32,
            symbol_white_list.as_ptr(),
            symbol_white_list.len() as u32,
        ).ok_or_else(|| {
            write::llvm_err(&diag_handler, "failed to prepare thin LTO context".to_string())
        })?;

        info!("thin LTO data created");
        timeline.record("data");

        let import_map = if cgcx.incr_comp_session_dir.is_some() {
            ThinLTOImports::from_thin_lto_data(data)
        } else {
            // If we don't compile incrementally, we don't need to load the
            // import data from LLVM.
            assert!(green_modules.is_empty());
            ThinLTOImports::new()
        };
        info!("thin LTO import map loaded");
        timeline.record("import-map-loaded");

        let data = ThinData(data);

        // Throw our data in an `Arc` as we'll be sharing it across threads. We
        // also put all memory referenced by the C++ data (buffers, ids, etc)
        // into the arc as well. After this we'll create a thin module
        // codegen per module in this data.
        let shared = Arc::new(ThinShared {
            data,
            thin_buffers,
            serialized_modules: serialized,
            module_names,
        });

        let mut copy_jobs = vec![];
        let mut opt_jobs = vec![];

        info!("checking which modules can be-reused and which have to be re-optimized.");
        for (module_index, module_name) in shared.module_names.iter().enumerate() {
            let module_name = module_name_to_str(module_name);

            // If the module hasn't changed and none of the modules it imports
            // from has changed, we can re-use the post-ThinLTO version of the
            // module.
            if green_modules.contains_key(module_name) {
                let imports_all_green = import_map.modules_imported_by(module_name)
                    .iter()
                    .all(|imported_module| green_modules.contains_key(imported_module));

                if imports_all_green {
                    let work_product = green_modules[module_name].clone();
                    copy_jobs.push(work_product);
                    info!(" - {}: re-used", module_name);
                    cgcx.cgu_reuse_tracker.set_actual_reuse(module_name,
                                                            CguReuse::PostLto);
                    continue
                }
            }

            info!(" - {}: re-compiled", module_name);
            opt_jobs.push(LtoModuleCodegen::Thin(ThinModule {
                shared: shared.clone(),
                idx: module_index,
            }));
        }

        Ok((opt_jobs, copy_jobs))
    }
}

fn run_pass_manager(cgcx: &CodegenContext,
                    tm: &llvm::TargetMachine,
                    llmod: &llvm::Module,
                    config: &ModuleConfig,
                    thin: bool) {
    // Now we have one massive module inside of llmod. Time to run the
    // LTO-specific optimization passes that LLVM provides.
    //
    // This code is based off the code found in llvm's LTO code generator:
    //      tools/lto/LTOCodeGenerator.cpp
    debug!("running the pass manager");
    unsafe {
        let pm = llvm::LLVMCreatePassManager();
        llvm::LLVMRustAddAnalysisPasses(tm, pm, llmod);

        if config.verify_llvm_ir {
            let pass = llvm::LLVMRustFindAndCreatePass("verify\0".as_ptr() as *const _);
            llvm::LLVMRustAddPass(pm, pass.unwrap());
        }

        // When optimizing for LTO we don't actually pass in `-O0`, but we force
        // it to always happen at least with `-O1`.
        //
        // With ThinLTO we mess around a lot with symbol visibility in a way
        // that will actually cause linking failures if we optimize at O0 which
        // notable is lacking in dead code elimination. To ensure we at least
        // get some optimizations and correctly link we forcibly switch to `-O1`
        // to get dead code elimination.
        //
        // Note that in general this shouldn't matter too much as you typically
        // only turn on ThinLTO when you're compiling with optimizations
        // otherwise.
        let opt_level = config.opt_level.unwrap_or(llvm::CodeGenOptLevel::None);
        let opt_level = match opt_level {
            llvm::CodeGenOptLevel::None => llvm::CodeGenOptLevel::Less,
            level => level,
        };
        with_llvm_pmb(llmod, config, opt_level, false, &mut |b| {
            if thin {
                if !llvm::LLVMRustPassManagerBuilderPopulateThinLTOPassManager(b, pm) {
                    panic!("this version of LLVM does not support ThinLTO");
                }
            } else {
                llvm::LLVMPassManagerBuilderPopulateLTOPassManager(b, pm,
                    /* Internalize = */ False,
                    /* RunInliner = */ True);
            }
        });

        if config.verify_llvm_ir {
            let pass = llvm::LLVMRustFindAndCreatePass("verify\0".as_ptr() as *const _);
            llvm::LLVMRustAddPass(pm, pass.unwrap());
        }

        time_ext(cgcx.time_passes, None, "LTO passes", ||
             llvm::LLVMRunPassManager(pm, llmod));

        llvm::LLVMDisposePassManager(pm);
    }
    debug!("lto done");
}

pub enum SerializedModule {
    Local(ModuleBuffer),
    FromRlib(Vec<u8>),
    FromUncompressedFile(memmap::Mmap),
}

impl SerializedModule {
    fn data(&self) -> &[u8] {
        match *self {
            SerializedModule::Local(ref m) => m.data(),
            SerializedModule::FromRlib(ref m) => m,
            SerializedModule::FromUncompressedFile(ref m) => m,
        }
    }
}

pub struct ModuleBuffer(&'static mut llvm::ModuleBuffer);

unsafe impl Send for ModuleBuffer {}
unsafe impl Sync for ModuleBuffer {}

impl ModuleBuffer {
    pub fn new(m: &llvm::Module) -> ModuleBuffer {
        ModuleBuffer(unsafe {
            llvm::LLVMRustModuleBufferCreate(m)
        })
    }

    pub fn data(&self) -> &[u8] {
        unsafe {
            let ptr = llvm::LLVMRustModuleBufferPtr(self.0);
            let len = llvm::LLVMRustModuleBufferLen(self.0);
            slice::from_raw_parts(ptr, len)
        }
    }
}

impl Drop for ModuleBuffer {
    fn drop(&mut self) {
        unsafe { llvm::LLVMRustModuleBufferFree(&mut *(self.0 as *mut _)); }
    }
}

pub struct ThinModule {
    shared: Arc<ThinShared>,
    idx: usize,
}

struct ThinShared {
    data: ThinData,
    thin_buffers: Vec<ThinBuffer>,
    serialized_modules: Vec<SerializedModule>,
    module_names: Vec<CString>,
}

struct ThinData(&'static mut llvm::ThinLTOData);

unsafe impl Send for ThinData {}
unsafe impl Sync for ThinData {}

impl Drop for ThinData {
    fn drop(&mut self) {
        unsafe {
            llvm::LLVMRustFreeThinLTOData(&mut *(self.0 as *mut _));
        }
    }
}

pub struct ThinBuffer(&'static mut llvm::ThinLTOBuffer);

unsafe impl Send for ThinBuffer {}
unsafe impl Sync for ThinBuffer {}

impl ThinBuffer {
    pub fn new(m: &llvm::Module) -> ThinBuffer {
        unsafe {
            let buffer = llvm::LLVMRustThinLTOBufferCreate(m);
            ThinBuffer(buffer)
        }
    }

    pub fn data(&self) -> &[u8] {
        unsafe {
            let ptr = llvm::LLVMRustThinLTOBufferPtr(self.0) as *const _;
            let len = llvm::LLVMRustThinLTOBufferLen(self.0);
            slice::from_raw_parts(ptr, len)
        }
    }
}

impl Drop for ThinBuffer {
    fn drop(&mut self) {
        unsafe {
            llvm::LLVMRustThinLTOBufferFree(&mut *(self.0 as *mut _));
        }
    }
}

impl ThinModule {
    fn name(&self) -> &str {
        self.shared.module_names[self.idx].to_str().unwrap()
    }

    fn cost(&self) -> u64 {
        // Yes, that's correct, we're using the size of the bytecode as an
        // indicator for how costly this codegen unit is.
        self.data().len() as u64
    }

    fn data(&self) -> &[u8] {
        let a = self.shared.thin_buffers.get(self.idx).map(|b| b.data());
        a.unwrap_or_else(|| {
            let len = self.shared.thin_buffers.len();
            self.shared.serialized_modules[self.idx - len].data()
        })
    }

    unsafe fn optimize(&mut self, cgcx: &CodegenContext, timeline: &mut Timeline)
        -> Result<ModuleCodegen, FatalError>
    {
        let diag_handler = cgcx.create_diag_handler();
        let tm = (cgcx.tm_factory)().map_err(|e| {
            write::llvm_err(&diag_handler, e)
        })?;

        // Right now the implementation we've got only works over serialized
        // modules, so we create a fresh new LLVM context and parse the module
        // into that context. One day, however, we may do this for upstream
        // crates but for locally codegened modules we may be able to reuse
        // that LLVM Context and Module.
        let llcx = llvm::LLVMRustContextCreate(cgcx.fewer_names);
        let llmod_raw = llvm::LLVMRustParseBitcodeForThinLTO(
            llcx,
            self.data().as_ptr(),
            self.data().len(),
            self.shared.module_names[self.idx].as_ptr(),
        ).ok_or_else(|| {
            let msg = "failed to parse bitcode for thin LTO module".to_string();
            write::llvm_err(&diag_handler, msg)
        })? as *const _;
        let module = ModuleCodegen {
            module_llvm: ModuleLlvm {
                llmod_raw,
                llcx,
                tm,
            },
            name: self.name().to_string(),
            kind: ModuleKind::Regular,
        };
        {
            let llmod = module.module_llvm.llmod();
            cgcx.save_temp_bitcode(&module, "thin-lto-input");

            // Before we do much else find the "main" `DICompileUnit` that we'll be
            // using below. If we find more than one though then rustc has changed
            // in a way we're not ready for, so generate an ICE by returning
            // an error.
            let mut cu1 = ptr::null_mut();
            let mut cu2 = ptr::null_mut();
            llvm::LLVMRustThinLTOGetDICompileUnit(llmod, &mut cu1, &mut cu2);
            if !cu2.is_null() {
                let msg = "multiple source DICompileUnits found".to_string();
                return Err(write::llvm_err(&diag_handler, msg))
            }

            // Like with "fat" LTO, get some better optimizations if landing pads
            // are disabled by removing all landing pads.
            if cgcx.no_landing_pads {
                llvm::LLVMRustMarkAllFunctionsNounwind(llmod);
                cgcx.save_temp_bitcode(&module, "thin-lto-after-nounwind");
                timeline.record("nounwind");
            }

            // Up next comes the per-module local analyses that we do for Thin LTO.
            // Each of these functions is basically copied from the LLVM
            // implementation and then tailored to suit this implementation. Ideally
            // each of these would be supported by upstream LLVM but that's perhaps
            // a patch for another day!
            //
            // You can find some more comments about these functions in the LLVM
            // bindings we've got (currently `PassWrapper.cpp`)
            if !llvm::LLVMRustPrepareThinLTORename(self.shared.data.0, llmod) {
                let msg = "failed to prepare thin LTO module".to_string();
                return Err(write::llvm_err(&diag_handler, msg))
            }
            cgcx.save_temp_bitcode(&module, "thin-lto-after-rename");
            timeline.record("rename");
            if !llvm::LLVMRustPrepareThinLTOResolveWeak(self.shared.data.0, llmod) {
                let msg = "failed to prepare thin LTO module".to_string();
                return Err(write::llvm_err(&diag_handler, msg))
            }
            cgcx.save_temp_bitcode(&module, "thin-lto-after-resolve");
            timeline.record("resolve");
            if !llvm::LLVMRustPrepareThinLTOInternalize(self.shared.data.0, llmod) {
                let msg = "failed to prepare thin LTO module".to_string();
                return Err(write::llvm_err(&diag_handler, msg))
            }
            cgcx.save_temp_bitcode(&module, "thin-lto-after-internalize");
            timeline.record("internalize");
            if !llvm::LLVMRustPrepareThinLTOImport(self.shared.data.0, llmod) {
                let msg = "failed to prepare thin LTO module".to_string();
                return Err(write::llvm_err(&diag_handler, msg))
            }
            cgcx.save_temp_bitcode(&module, "thin-lto-after-import");
            timeline.record("import");

            // Ok now this is a bit unfortunate. This is also something you won't
            // find upstream in LLVM's ThinLTO passes! This is a hack for now to
            // work around bugs in LLVM.
            //
            // First discovered in #45511 it was found that as part of ThinLTO
            // importing passes LLVM will import `DICompileUnit` metadata
            // information across modules. This means that we'll be working with one
            // LLVM module that has multiple `DICompileUnit` instances in it (a
            // bunch of `llvm.dbg.cu` members). Unfortunately there's a number of
            // bugs in LLVM's backend which generates invalid DWARF in a situation
            // like this:
            //
            //  https://bugs.llvm.org/show_bug.cgi?id=35212
            //  https://bugs.llvm.org/show_bug.cgi?id=35562
            //
            // While the first bug there is fixed the second ended up causing #46346
            // which was basically a resurgence of #45511 after LLVM's bug 35212 was
            // fixed.
            //
            // This function below is a huge hack around this problem. The function
            // below is defined in `PassWrapper.cpp` and will basically "merge"
            // all `DICompileUnit` instances in a module. Basically it'll take all
            // the objects, rewrite all pointers of `DISubprogram` to point to the
            // first `DICompileUnit`, and then delete all the other units.
            //
            // This is probably mangling to the debug info slightly (but hopefully
            // not too much) but for now at least gets LLVM to emit valid DWARF (or
            // so it appears). Hopefully we can remove this once upstream bugs are
            // fixed in LLVM.
            llvm::LLVMRustThinLTOPatchDICompileUnit(llmod, cu1);
            cgcx.save_temp_bitcode(&module, "thin-lto-after-patch");
            timeline.record("patch");

            // Alright now that we've done everything related to the ThinLTO
            // analysis it's time to run some optimizations! Here we use the same
            // `run_pass_manager` as the "fat" LTO above except that we tell it to
            // populate a thin-specific pass manager, which presumably LLVM treats a
            // little differently.
            info!("running thin lto passes over {}", module.name);
            let config = cgcx.config(module.kind);
            run_pass_manager(cgcx, module.module_llvm.tm, llmod, config, true);
            cgcx.save_temp_bitcode(&module, "thin-lto-after-pm");
            timeline.record("thin-done");
        }

        Ok(module)
    }
}

#[derive(Debug)]
pub struct ThinLTOImports {
    // key = llvm name of importing module, value = list of modules it imports from
    imports: FxHashMap<String, Vec<String>>,
}

impl ThinLTOImports {
    fn new() -> ThinLTOImports {
        ThinLTOImports {
            imports: FxHashMap(),
        }
    }

    fn modules_imported_by(&self, llvm_module_name: &str) -> &[String] {
        self.imports.get(llvm_module_name).map(|v| &v[..]).unwrap_or(&[])
    }

    /// Load the ThinLTO import map from ThinLTOData.
    unsafe fn from_thin_lto_data(data: *const llvm::ThinLTOData) -> ThinLTOImports {
        unsafe extern "C" fn imported_module_callback(payload: *mut libc::c_void,
                                                      importing_module_name: *const libc::c_char,
                                                      imported_module_name: *const libc::c_char) {
            let map = &mut* (payload as *mut ThinLTOImports);
            let importing_module_name = CStr::from_ptr(importing_module_name);
            let importing_module_name = module_name_to_str(&importing_module_name);
            let imported_module_name = CStr::from_ptr(imported_module_name);
            let imported_module_name = module_name_to_str(&imported_module_name);

            if !map.imports.contains_key(importing_module_name) {
                map.imports.insert(importing_module_name.to_owned(), vec![]);
            }

            map.imports
               .get_mut(importing_module_name)
               .unwrap()
               .push(imported_module_name.to_owned());
        }
        let mut map = ThinLTOImports {
            imports: FxHashMap(),
        };
        llvm::LLVMRustGetThinLTOModuleImports(data,
                                              imported_module_callback,
                                              &mut map as *mut _ as *mut libc::c_void);
        map
    }
}

fn module_name_to_str(c_str: &CStr) -> &str {
    match c_str.to_str() {
        Ok(s) => s,
        Err(e) => {
            bug!("Encountered non-utf8 LLVM module name `{}`: {}",
                c_str.to_string_lossy(),
                e)
        }
    }
}
