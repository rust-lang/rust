use crate::back::bytecode::DecodedBytecode;
use crate::back::write::{self, DiagnosticHandlers, with_llvm_pmb, save_temp_bitcode,
    to_llvm_opt_settings};
use crate::llvm::archive_ro::ArchiveRO;
use crate::llvm::{self, True, False};
use crate::{ModuleLlvm, LlvmCodegenBackend};
use rustc_codegen_ssa::back::symbol_export;
use rustc_codegen_ssa::back::write::{ModuleConfig, CodegenContext, FatLTOInput};
use rustc_codegen_ssa::back::lto::{SerializedModule, LtoModuleCodegen, ThinShared, ThinModule};
use rustc_codegen_ssa::traits::*;
use errors::{FatalError, Handler};
use rustc::dep_graph::WorkProduct;
use rustc::dep_graph::cgu_reuse_tracker::CguReuse;
use rustc::hir::def_id::LOCAL_CRATE;
use rustc::middle::exported_symbols::SymbolExportLevel;
use rustc::session::config::{self, Lto};
use rustc::util::common::time_ext;
use rustc_data_structures::fx::FxHashMap;
use rustc_codegen_ssa::{RLIB_BYTECODE_EXTENSION, ModuleCodegen, ModuleKind};

use std::ffi::{CStr, CString};
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

fn prepare_lto(cgcx: &CodegenContext<LlvmCodegenBackend>,
               diag_handler: &Handler)
    -> Result<(Vec<CString>, Vec<(SerializedModule<ModuleBuffer>, CString)>), FatalError>
{
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
    let _timer = cgcx.profile_activity("generate_symbol_white_list_for_thinlto");
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
            let _timer = cgcx.profile_activity(format!("load: {}", path.display()));
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
        }
    }

    Ok((symbol_white_list, upstream_modules))
}

/// Performs fat LTO by merging all modules into a single one and returning it
/// for further optimization.
pub(crate) fn run_fat(cgcx: &CodegenContext<LlvmCodegenBackend>,
                      modules: Vec<FatLTOInput<LlvmCodegenBackend>>,
                      cached_modules: Vec<(SerializedModule<ModuleBuffer>, WorkProduct)>)
    -> Result<LtoModuleCodegen<LlvmCodegenBackend>, FatalError>
{
    let diag_handler = cgcx.create_diag_handler();
    let (symbol_white_list, upstream_modules) = prepare_lto(cgcx, &diag_handler)?;
    let symbol_white_list = symbol_white_list.iter()
                                             .map(|c| c.as_ptr())
                                             .collect::<Vec<_>>();
    fat_lto(
        cgcx,
        &diag_handler,
        modules,
        cached_modules,
        upstream_modules,
        &symbol_white_list,
    )
}

/// Performs thin LTO by performing necessary global analysis and returning two
/// lists, one of the modules that need optimization and another for modules that
/// can simply be copied over from the incr. comp. cache.
pub(crate) fn run_thin(cgcx: &CodegenContext<LlvmCodegenBackend>,
                       modules: Vec<(String, ThinBuffer)>,
                       cached_modules: Vec<(SerializedModule<ModuleBuffer>, WorkProduct)>)
    -> Result<(Vec<LtoModuleCodegen<LlvmCodegenBackend>>, Vec<WorkProduct>), FatalError>
{
    let diag_handler = cgcx.create_diag_handler();
    let (symbol_white_list, upstream_modules) = prepare_lto(cgcx, &diag_handler)?;
    let symbol_white_list = symbol_white_list.iter()
                                             .map(|c| c.as_ptr())
                                             .collect::<Vec<_>>();
    if cgcx.opts.cg.linker_plugin_lto.enabled() {
        unreachable!("We should never reach this case if the LTO step \
                      is deferred to the linker");
    }
    thin_lto(cgcx,
             &diag_handler,
             modules,
             upstream_modules,
             cached_modules,
             &symbol_white_list)
}

pub(crate) fn prepare_thin(
    module: ModuleCodegen<ModuleLlvm>
) -> (String, ThinBuffer) {
    let name = module.name.clone();
    let buffer = ThinBuffer::new(module.module_llvm.llmod());
    (name, buffer)
}

fn fat_lto(cgcx: &CodegenContext<LlvmCodegenBackend>,
           diag_handler: &Handler,
           mut modules: Vec<FatLTOInput<LlvmCodegenBackend>>,
           cached_modules: Vec<(SerializedModule<ModuleBuffer>, WorkProduct)>,
           mut serialized_modules: Vec<(SerializedModule<ModuleBuffer>, CString)>,
           symbol_white_list: &[*const libc::c_char])
    -> Result<LtoModuleCodegen<LlvmCodegenBackend>, FatalError>
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
    let costliest_module = modules.iter()
        .enumerate()
        .filter_map(|(i, module)| {
            match module {
                FatLTOInput::InMemory(m) => Some((i, m)),
                FatLTOInput::Serialized { .. } => None,
            }
        })
        .filter(|&(_, module)| module.kind == ModuleKind::Regular)
        .map(|(i, module)| {
            let cost = unsafe {
                llvm::LLVMRustModuleCost(module.module_llvm.llmod())
            };
            (cost, i)
        })
        .max();

    // If we found a costliest module, we're good to go. Otherwise all our
    // inputs were serialized which could happen in the case, for example, that
    // all our inputs were incrementally reread from the cache and we're just
    // re-executing the LTO passes. If that's the case deserialize the first
    // module and create a linker with it.
    let module: ModuleCodegen<ModuleLlvm> = match costliest_module {
        Some((_cost, i)) => {
            match modules.remove(i) {
                FatLTOInput::InMemory(m) => m,
                FatLTOInput::Serialized { .. } => unreachable!(),
            }
        }
        None => {
            let pos = modules.iter().position(|m| {
                match m {
                    FatLTOInput::InMemory(_) => false,
                    FatLTOInput::Serialized { .. } => true,
                }
            }).expect("must have at least one serialized module");
            let (name, buffer) = match modules.remove(pos) {
                FatLTOInput::Serialized { name, buffer } => (name, buffer),
                FatLTOInput::InMemory(_) => unreachable!(),
            };
            ModuleCodegen {
                module_llvm: ModuleLlvm::parse(cgcx, &name, &buffer, diag_handler)?,
                name,
                kind: ModuleKind::Regular,
            }
        }
    };
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
        serialized_modules.extend(modules.into_iter().map(|module| {
            match module {
                FatLTOInput::InMemory(module) => {
                    let buffer = ModuleBuffer::new(module.module_llvm.llmod());
                    let llmod_id = CString::new(&module.name[..]).unwrap();
                    (SerializedModule::Local(buffer), llmod_id)
                }
                FatLTOInput::Serialized { name, buffer } => {
                    let llmod_id = CString::new(name).unwrap();
                    (SerializedModule::Local(buffer), llmod_id)
                }
            }
        }));
        serialized_modules.extend(cached_modules.into_iter().map(|(buffer, wp)| {
            (buffer, CString::new(wp.cgu_name).unwrap())
        }));

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
                    write::llvm_err(&diag_handler, &msg)
                })
            })?;
            serialized_bitcode.push(bc_decoded);
        }
        drop(linker);
        save_temp_bitcode(&cgcx, &module, "lto.input");

        // Internalize everything that *isn't* in our whitelist to help strip out
        // more modules and such
        unsafe {
            let ptr = symbol_white_list.as_ptr();
            llvm::LLVMRustRunRestrictionPass(llmod,
                                             ptr as *const *const libc::c_char,
                                             symbol_white_list.len() as libc::size_t);
            save_temp_bitcode(&cgcx, &module, "lto.after-restriction");
        }

        if cgcx.no_landing_pads {
            unsafe {
                llvm::LLVMRustMarkAllFunctionsNounwind(llmod);
            }
            save_temp_bitcode(&cgcx, &module, "lto.after-nounwind");
        }
    }

    Ok(LtoModuleCodegen::Fat {
        module: Some(module),
        _serialized_bitcode: serialized_bitcode,
    })
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
fn thin_lto(cgcx: &CodegenContext<LlvmCodegenBackend>,
            diag_handler: &Handler,
            modules: Vec<(String, ThinBuffer)>,
            serialized_modules: Vec<(SerializedModule<ModuleBuffer>, CString)>,
            cached_modules: Vec<(SerializedModule<ModuleBuffer>, WorkProduct)>,
            symbol_white_list: &[*const libc::c_char])
    -> Result<(Vec<LtoModuleCodegen<LlvmCodegenBackend>>, Vec<WorkProduct>), FatalError>
{
    unsafe {
        info!("going for that thin, thin LTO");

        let green_modules: FxHashMap<_, _> = cached_modules
            .iter()
            .map(|&(_, ref wp)| (wp.cgu_name.clone(), wp.clone()))
            .collect();

        let full_scope_len = modules.len() + serialized_modules.len() + cached_modules.len();
        let mut thin_buffers = Vec::with_capacity(modules.len());
        let mut module_names = Vec::with_capacity(full_scope_len);
        let mut thin_modules = Vec::with_capacity(full_scope_len);

        for (i, (name, buffer)) in modules.into_iter().enumerate() {
            info!("local module: {} - {}", i, name);
            let cname = CString::new(name.clone()).unwrap();
            thin_modules.push(llvm::ThinLTOModule {
                identifier: cname.as_ptr(),
                data: buffer.data().as_ptr(),
                len: buffer.data().len(),
            });
            thin_buffers.push(buffer);
            module_names.push(cname);
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
        let mut serialized = Vec::with_capacity(serialized_modules.len() + cached_modules.len());

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
            write::llvm_err(&diag_handler, "failed to prepare thin LTO context")
        })?;

        info!("thin LTO data created");

        let import_map = if cgcx.incr_comp_session_dir.is_some() {
            ThinLTOImports::from_thin_lto_data(data)
        } else {
            // If we don't compile incrementally, we don't need to load the
            // import data from LLVM.
            assert!(green_modules.is_empty());
            ThinLTOImports::default()
        };
        info!("thin LTO import map loaded");

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

pub(crate) fn run_pass_manager(cgcx: &CodegenContext<LlvmCodegenBackend>,
                    module: &ModuleCodegen<ModuleLlvm>,
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
        llvm::LLVMRustAddAnalysisPasses(module.module_llvm.tm, pm, module.module_llvm.llmod());

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
        let opt_level = config.opt_level.map(|x| to_llvm_opt_settings(x).0)
            .unwrap_or(llvm::CodeGenOptLevel::None);
        let opt_level = match opt_level {
            llvm::CodeGenOptLevel::None => llvm::CodeGenOptLevel::Less,
            level => level,
        };
        with_llvm_pmb(module.module_llvm.llmod(), config, opt_level, false, &mut |b| {
            if thin {
                llvm::LLVMRustPassManagerBuilderPopulateThinLTOPassManager(b, pm);
            } else {
                llvm::LLVMPassManagerBuilderPopulateLTOPassManager(b, pm,
                    /* Internalize = */ False,
                    /* RunInliner = */ True);
            }
        });

        // We always generate bitcode through ThinLTOBuffers,
        // which do not support anonymous globals
        if config.bitcode_needed() {
            let pass = llvm::LLVMRustFindAndCreatePass("name-anon-globals\0".as_ptr() as *const _);
            llvm::LLVMRustAddPass(pm, pass.unwrap());
        }

        if config.verify_llvm_ir {
            let pass = llvm::LLVMRustFindAndCreatePass("verify\0".as_ptr() as *const _);
            llvm::LLVMRustAddPass(pm, pass.unwrap());
        }

        time_ext(cgcx.time_passes, None, "LTO passes", ||
             llvm::LLVMRunPassManager(pm, module.module_llvm.llmod()));

        llvm::LLVMDisposePassManager(pm);
    }
    debug!("lto done");
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

    pub fn parse<'a>(
        &self,
        name: &str,
        cx: &'a llvm::Context,
        handler: &Handler,
    ) -> Result<&'a llvm::Module, FatalError> {
        let name = CString::new(name).unwrap();
        parse_module(cx, &name, self.data(), handler)
    }
}

impl ModuleBufferMethods for ModuleBuffer {
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
        unsafe { llvm::LLVMRustModuleBufferFree(&mut *(self.0 as *mut _)); }
    }
}

pub struct ThinData(&'static mut llvm::ThinLTOData);

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
}

impl ThinBufferMethods for ThinBuffer {
    fn data(&self) -> &[u8] {
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

pub unsafe fn optimize_thin_module(
    thin_module: &mut ThinModule<LlvmCodegenBackend>,
    cgcx: &CodegenContext<LlvmCodegenBackend>,
) -> Result<ModuleCodegen<ModuleLlvm>, FatalError> {
    let diag_handler = cgcx.create_diag_handler();
    let tm = (cgcx.tm_factory.0)().map_err(|e| {
        write::llvm_err(&diag_handler, &e)
    })?;

    // Right now the implementation we've got only works over serialized
    // modules, so we create a fresh new LLVM context and parse the module
    // into that context. One day, however, we may do this for upstream
    // crates but for locally codegened modules we may be able to reuse
    // that LLVM Context and Module.
    let llcx = llvm::LLVMRustContextCreate(cgcx.fewer_names);
    let llmod_raw = parse_module(
        llcx,
        &thin_module.shared.module_names[thin_module.idx],
        thin_module.data(),
        &diag_handler,
    )? as *const _;
    let module = ModuleCodegen {
        module_llvm: ModuleLlvm {
            llmod_raw,
            llcx,
            tm,
        },
        name: thin_module.name().to_string(),
        kind: ModuleKind::Regular,
    };
    {
        let llmod = module.module_llvm.llmod();
        save_temp_bitcode(&cgcx, &module, "thin-lto-input");

        // Before we do much else find the "main" `DICompileUnit` that we'll be
        // using below. If we find more than one though then rustc has changed
        // in a way we're not ready for, so generate an ICE by returning
        // an error.
        let mut cu1 = ptr::null_mut();
        let mut cu2 = ptr::null_mut();
        llvm::LLVMRustThinLTOGetDICompileUnit(llmod, &mut cu1, &mut cu2);
        if !cu2.is_null() {
            let msg = "multiple source DICompileUnits found";
            return Err(write::llvm_err(&diag_handler, msg))
        }

        // Like with "fat" LTO, get some better optimizations if landing pads
        // are disabled by removing all landing pads.
        if cgcx.no_landing_pads {
            let _timer = cgcx.profile_activity("LLVM_remove_landing_pads");
            llvm::LLVMRustMarkAllFunctionsNounwind(llmod);
            save_temp_bitcode(&cgcx, &module, "thin-lto-after-nounwind");
        }

        // Up next comes the per-module local analyses that we do for Thin LTO.
        // Each of these functions is basically copied from the LLVM
        // implementation and then tailored to suit this implementation. Ideally
        // each of these would be supported by upstream LLVM but that's perhaps
        // a patch for another day!
        //
        // You can find some more comments about these functions in the LLVM
        // bindings we've got (currently `PassWrapper.cpp`)
        if !llvm::LLVMRustPrepareThinLTORename(thin_module.shared.data.0, llmod) {
            let msg = "failed to prepare thin LTO module";
            return Err(write::llvm_err(&diag_handler, msg))
        }
        save_temp_bitcode(cgcx, &module, "thin-lto-after-rename");
        if !llvm::LLVMRustPrepareThinLTOResolveWeak(thin_module.shared.data.0, llmod) {
            let msg = "failed to prepare thin LTO module";
            return Err(write::llvm_err(&diag_handler, msg))
        }
        save_temp_bitcode(cgcx, &module, "thin-lto-after-resolve");
        if !llvm::LLVMRustPrepareThinLTOInternalize(thin_module.shared.data.0, llmod) {
            let msg = "failed to prepare thin LTO module";
            return Err(write::llvm_err(&diag_handler, msg))
        }
        save_temp_bitcode(cgcx, &module, "thin-lto-after-internalize");
        if !llvm::LLVMRustPrepareThinLTOImport(thin_module.shared.data.0, llmod) {
            let msg = "failed to prepare thin LTO module";
            return Err(write::llvm_err(&diag_handler, msg))
        }
        save_temp_bitcode(cgcx, &module, "thin-lto-after-import");

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
        save_temp_bitcode(cgcx, &module, "thin-lto-after-patch");

        // Alright now that we've done everything related to the ThinLTO
        // analysis it's time to run some optimizations! Here we use the same
        // `run_pass_manager` as the "fat" LTO above except that we tell it to
        // populate a thin-specific pass manager, which presumably LLVM treats a
        // little differently.
        info!("running thin lto passes over {}", module.name);
        let config = cgcx.config(module.kind);
        run_pass_manager(cgcx, &module, config, true);
        save_temp_bitcode(cgcx, &module, "thin-lto-after-pm");
    }
    Ok(module)
}

#[derive(Debug, Default)]
pub struct ThinLTOImports {
    // key = llvm name of importing module, value = list of modules it imports from
    imports: FxHashMap<String, Vec<String>>,
}

impl ThinLTOImports {
    fn modules_imported_by(&self, llvm_module_name: &str) -> &[String] {
        self.imports.get(llvm_module_name).map(|v| &v[..]).unwrap_or(&[])
    }

    /// Loads the ThinLTO import map from ThinLTOData.
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
        let mut map = ThinLTOImports::default();
        llvm::LLVMRustGetThinLTOModuleImports(data,
                                              imported_module_callback,
                                              &mut map as *mut _ as *mut libc::c_void);
        map
    }
}

fn module_name_to_str(c_str: &CStr) -> &str {
    c_str.to_str().unwrap_or_else(|e|
        bug!("Encountered non-utf8 LLVM module name `{}`: {}", c_str.to_string_lossy(), e))
}

fn parse_module<'a>(
    cx: &'a llvm::Context,
    name: &CStr,
    data: &[u8],
    diag_handler: &Handler,
) -> Result<&'a llvm::Module, FatalError> {
    unsafe {
        llvm::LLVMRustParseBitcodeForLTO(
            cx,
            data.as_ptr(),
            data.len(),
            name.as_ptr(),
        ).ok_or_else(|| {
            let msg = "failed to parse bitcode for LTO module";
            write::llvm_err(&diag_handler, msg)
        })
    }
}
