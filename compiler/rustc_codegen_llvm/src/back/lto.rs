use std::collections::BTreeMap;
use std::ffi::{CStr, CString};
use std::fs::File;
use std::path::Path;
use std::ptr::NonNull;
use std::sync::Arc;
use std::{io, iter, slice};

use object::read::archive::ArchiveFile;
use rustc_abi::{Align, Size};
use rustc_codegen_ssa::back::lto::{LtoModuleCodegen, SerializedModule, ThinModule, ThinShared};
use rustc_codegen_ssa::back::symbol_export;
use rustc_codegen_ssa::back::write::{CodegenContext, FatLtoInput};
use rustc_codegen_ssa::traits::*;
use rustc_codegen_ssa::{ModuleCodegen, ModuleKind, looks_like_rust_object_file};
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::memmap::Mmap;
use rustc_errors::{DiagCtxtHandle, FatalError};
use rustc_hir::def_id::LOCAL_CRATE;
use rustc_middle::bug;
use rustc_middle::dep_graph::WorkProduct;
use rustc_middle::middle::exported_symbols::{SymbolExportInfo, SymbolExportLevel};
use rustc_session::config::{self, CrateType, Lto};
use tracing::{debug, info};

use llvm::Linkage::*;

use crate::back::write::{
    self, CodegenDiagnosticsStage, DiagnosticHandlers, bitcode_section_name, save_temp_bitcode,
};
use crate::builder::SBuilder;
use crate::errors::{
    DynamicLinkingWithLTO, LlvmError, LtoBitcodeFromRlib, LtoDisallowed, LtoDylib, LtoProcMacro,
};
use crate::common::AsCCharPtr;
use crate::llvm::AttributePlace::Function;
use crate::llvm::{self, build_string, Linkage};
use crate::{LlvmCodegenBackend, ModuleLlvm, SimpleCx, attributes};

/// We keep track of the computed LTO cache keys from the previous
/// session to determine which CGUs we can reuse.
const THIN_LTO_KEYS_INCR_COMP_FILE_NAME: &str = "thin-lto-past-keys.bin";

fn crate_type_allows_lto(crate_type: CrateType) -> bool {
    match crate_type {
        CrateType::Executable
            | CrateType::Dylib
            | CrateType::Staticlib
            | CrateType::Cdylib
            | CrateType::ProcMacro
            | CrateType::Sdylib => true,
        CrateType::Rlib => false,
    }
}

fn prepare_lto(
    cgcx: &CodegenContext<LlvmCodegenBackend>,
    dcx: DiagCtxtHandle<'_>,
) -> Result<(Vec<CString>, Vec<(SerializedModule<ModuleBuffer>, CString)>), FatalError> {
    let export_threshold = match cgcx.lto {
        // We're just doing LTO for our one crate
        Lto::ThinLocal => SymbolExportLevel::Rust,

        // We're doing LTO for the entire crate graph
        Lto::Fat | Lto::Thin => symbol_export::crates_export_threshold(&cgcx.crate_types),

        Lto::No => panic!("didn't request LTO but we're doing LTO"),
    };

    let symbol_filter = &|&(ref name, info): &(String, SymbolExportInfo)| {
        if info.level.is_below_threshold(export_threshold) || info.used {
            Some(CString::new(name.as_str()).unwrap())
        } else {
            None
        }
    };
    let exported_symbols = cgcx.exported_symbols.as_ref().expect("needs exported symbols for LTO");
    let mut symbols_below_threshold = {
        let _timer = cgcx.prof.generic_activity("LLVM_lto_generate_symbols_below_threshold");
        exported_symbols[&LOCAL_CRATE].iter().filter_map(symbol_filter).collect::<Vec<CString>>()
    };
    info!("{} symbols to preserve in this crate", symbols_below_threshold.len());

    // If we're performing LTO for the entire crate graph, then for each of our
    // upstream dependencies, find the corresponding rlib and load the bitcode
    // from the archive.
    //
    // We save off all the bytecode and LLVM module ids for later processing
    // with either fat or thin LTO
    let mut upstream_modules = Vec::new();
    if cgcx.lto != Lto::ThinLocal {
        // Make sure we actually can run LTO
        for crate_type in cgcx.crate_types.iter() {
            if !crate_type_allows_lto(*crate_type) {
                dcx.emit_err(LtoDisallowed);
                return Err(FatalError);
            } else if *crate_type == CrateType::Dylib {
                if !cgcx.opts.unstable_opts.dylib_lto {
                    dcx.emit_err(LtoDylib);
                    return Err(FatalError);
                }
            } else if *crate_type == CrateType::ProcMacro && !cgcx.opts.unstable_opts.dylib_lto {
                dcx.emit_err(LtoProcMacro);
                return Err(FatalError);
            }
        }

        if cgcx.opts.cg.prefer_dynamic && !cgcx.opts.unstable_opts.dylib_lto {
            dcx.emit_err(DynamicLinkingWithLTO);
            return Err(FatalError);
        }

        for &(cnum, ref path) in cgcx.each_linked_rlib_for_lto.iter() {
            let exported_symbols =
                cgcx.exported_symbols.as_ref().expect("needs exported symbols for LTO");
            {
                let _timer =
                    cgcx.prof.generic_activity("LLVM_lto_generate_symbols_below_threshold");
                symbols_below_threshold
                    .extend(exported_symbols[&cnum].iter().filter_map(symbol_filter));
                }

            let archive_data = unsafe {
                Mmap::map(std::fs::File::open(&path).expect("couldn't open rlib"))
                    .expect("couldn't map rlib")
            };
            let archive = ArchiveFile::parse(&*archive_data).expect("wanted an rlib");
            let obj_files = archive
                .members()
                .filter_map(|child| {
                    child.ok().and_then(|c| {
                        std::str::from_utf8(c.name()).ok().map(|name| (name.trim(), c))
                    })
                })
            .filter(|&(name, _)| looks_like_rust_object_file(name));
            for (name, child) in obj_files {
                info!("adding bitcode from {}", name);
                match get_bitcode_slice_from_object_data(
                    child.data(&*archive_data).expect("corrupt rlib"),
                    cgcx,
                ) {
                    Ok(data) => {
                        let module = SerializedModule::FromRlib(data.to_vec());
                        upstream_modules.push((module, CString::new(name).unwrap()));
                    }
                    Err(e) => {
                        dcx.emit_err(e);
                        return Err(FatalError);
                    }
                }
            }
        }
    }

    // __llvm_profile_counter_bias is pulled in at link time by an undefined reference to
    // __llvm_profile_runtime, therefore we won't know until link time if this symbol
    // should have default visibility.
    symbols_below_threshold.push(c"__llvm_profile_counter_bias".to_owned());
    Ok((symbols_below_threshold, upstream_modules))
}

fn get_bitcode_slice_from_object_data<'a>(
    obj: &'a [u8],
    cgcx: &CodegenContext<LlvmCodegenBackend>,
) -> Result<&'a [u8], LtoBitcodeFromRlib> {
    // We're about to assume the data here is an object file with sections, but if it's raw LLVM IR
    // that won't work. Fortunately, if that's what we have we can just return the object directly,
    // so we sniff the relevant magic strings here and return.
    if obj.starts_with(b"\xDE\xC0\x17\x0B") || obj.starts_with(b"BC\xC0\xDE") {
        return Ok(obj);
    }
    // We drop the "__LLVM," prefix here because on Apple platforms there's a notion of "segment
    // name" which in the public API for sections gets treated as part of the section name, but
    // internally in MachOObjectFile.cpp gets treated separately.
    let section_name = bitcode_section_name(cgcx).to_str().unwrap().trim_start_matches("__LLVM,");
    let mut len = 0;
    let data = unsafe {
        llvm::LLVMRustGetSliceFromObjectDataByName(
            obj.as_ptr(),
            obj.len(),
            section_name.as_ptr(),
            section_name.len(),
            &mut len,
        )
    };
    if !data.is_null() {
        assert!(len != 0);
        let bc = unsafe { slice::from_raw_parts(data, len) };

        // `bc` must be a sub-slice of `obj`.
        assert!(obj.as_ptr() <= bc.as_ptr());
        assert!(bc[bc.len()..bc.len()].as_ptr() <= obj[obj.len()..obj.len()].as_ptr());

        Ok(bc)
    } else {
        assert!(len == 0);
        Err(LtoBitcodeFromRlib {
            llvm_err: llvm::last_error().unwrap_or_else(|| "unknown LLVM error".to_string()),
        })
    }
}

/// Performs fat LTO by merging all modules into a single one and returning it
/// for further optimization.
pub(crate) fn run_fat(
    cgcx: &CodegenContext<LlvmCodegenBackend>,
    modules: Vec<FatLtoInput<LlvmCodegenBackend>>,
    cached_modules: Vec<(SerializedModule<ModuleBuffer>, WorkProduct)>,
) -> Result<LtoModuleCodegen<LlvmCodegenBackend>, FatalError> {
    let dcx = cgcx.create_dcx();
    let dcx = dcx.handle();
    let (symbols_below_threshold, upstream_modules) = prepare_lto(cgcx, dcx)?;
    let symbols_below_threshold =
        symbols_below_threshold.iter().map(|c| c.as_ptr()).collect::<Vec<_>>();
    fat_lto(cgcx, dcx, modules, cached_modules, upstream_modules, &symbols_below_threshold)
}

/// Performs thin LTO by performing necessary global analysis and returning two
/// lists, one of the modules that need optimization and another for modules that
/// can simply be copied over from the incr. comp. cache.
pub(crate) fn run_thin(
    cgcx: &CodegenContext<LlvmCodegenBackend>,
    modules: Vec<(String, ThinBuffer)>,
    cached_modules: Vec<(SerializedModule<ModuleBuffer>, WorkProduct)>,
) -> Result<(Vec<LtoModuleCodegen<LlvmCodegenBackend>>, Vec<WorkProduct>), FatalError> {
    let dcx = cgcx.create_dcx();
    let dcx = dcx.handle();
    let (symbols_below_threshold, upstream_modules) = prepare_lto(cgcx, dcx)?;
    let symbols_below_threshold =
        symbols_below_threshold.iter().map(|c| c.as_ptr()).collect::<Vec<_>>();
    if cgcx.opts.cg.linker_plugin_lto.enabled() {
        unreachable!(
            "We should never reach this case if the LTO step \
                      is deferred to the linker"
        );
    }
    thin_lto(cgcx, dcx, modules, upstream_modules, cached_modules, &symbols_below_threshold)
}

pub(crate) fn prepare_thin(
    module: ModuleCodegen<ModuleLlvm>,
    emit_summary: bool,
) -> (String, ThinBuffer) {
    let name = module.name;
    let buffer = ThinBuffer::new(module.module_llvm.llmod(), true, emit_summary);
    (name, buffer)
}

fn fat_lto(
    cgcx: &CodegenContext<LlvmCodegenBackend>,
    dcx: DiagCtxtHandle<'_>,
    modules: Vec<FatLtoInput<LlvmCodegenBackend>>,
    cached_modules: Vec<(SerializedModule<ModuleBuffer>, WorkProduct)>,
    mut serialized_modules: Vec<(SerializedModule<ModuleBuffer>, CString)>,
    symbols_below_threshold: &[*const libc::c_char],
) -> Result<LtoModuleCodegen<LlvmCodegenBackend>, FatalError> {
    let _timer = cgcx.prof.generic_activity("LLVM_fat_lto_build_monolithic_module");
    info!("going for a fat lto");

    // Sort out all our lists of incoming modules into two lists.
    //
    // * `serialized_modules` (also and argument to this function) contains all
    //   modules that are serialized in-memory.
    // * `in_memory` contains modules which are already parsed and in-memory,
    //   such as from multi-CGU builds.
    //
    // All of `cached_modules` (cached from previous incremental builds) can
    // immediately go onto the `serialized_modules` modules list and then we can
    // split the `modules` array into these two lists.
    let mut in_memory = Vec::new();
    serialized_modules.extend(cached_modules.into_iter().map(|(buffer, wp)| {
        info!("pushing cached module {:?}", wp.cgu_name);
        (buffer, CString::new(wp.cgu_name).unwrap())
    }));
    for module in modules {
        match module {
            FatLtoInput::InMemory(m) => in_memory.push(m),
            FatLtoInput::Serialized { name, buffer } => {
                info!("pushing serialized module {:?}", name);
                let buffer = SerializedModule::Local(buffer);
                serialized_modules.push((buffer, CString::new(name).unwrap()));
            }
        }
    }

    // Find the "costliest" module and merge everything into that codegen unit.
    // All the other modules will be serialized and reparsed into the new
    // context, so this hopefully avoids serializing and parsing the largest
    // codegen unit.
    //
    // Additionally use a regular module as the base here to ensure that various
    // file copy operations in the backend work correctly. The only other kind
    // of module here should be an allocator one, and if your crate is smaller
    // than the allocator module then the size doesn't really matter anyway.
    let costliest_module = in_memory
        .iter()
        .enumerate()
        .filter(|&(_, module)| module.kind == ModuleKind::Regular)
        .map(|(i, module)| {
            let cost = unsafe { llvm::LLVMRustModuleCost(module.module_llvm.llmod()) };
            (cost, i)
        })
    .max();

    // If we found a costliest module, we're good to go. Otherwise all our
    // inputs were serialized which could happen in the case, for example, that
    // all our inputs were incrementally reread from the cache and we're just
    // re-executing the LTO passes. If that's the case deserialize the first
    // module and create a linker with it.
    let module: ModuleCodegen<ModuleLlvm> = match costliest_module {
        Some((_cost, i)) => in_memory.remove(i),
        None => {
            assert!(!serialized_modules.is_empty(), "must have at least one serialized module");
            let (buffer, name) = serialized_modules.remove(0);
            info!("no in-memory regular modules to choose from, parsing {:?}", name);
            let llvm_module = ModuleLlvm::parse(cgcx, &name, buffer.data(), dcx)?;
            ModuleCodegen::new_regular(name.into_string().unwrap(), llvm_module)
        }
    };
    {
        let (llcx, llmod) = {
            let llvm = &module.module_llvm;
            (&llvm.llcx, llvm.llmod())
        };
        info!("using {:?} as a base module", module.name);

        // The linking steps below may produce errors and diagnostics within LLVM
        // which we'd like to handle and print, so set up our diagnostic handlers
        // (which get unregistered when they go out of scope below).
        let _handler =
            DiagnosticHandlers::new(cgcx, dcx, llcx, &module, CodegenDiagnosticsStage::LTO);

        // For all other modules we codegened we'll need to link them into our own
        // bitcode. All modules were codegened in their own LLVM context, however,
        // and we want to move everything to the same LLVM context. Currently the
        // way we know of to do that is to serialize them to a string and them parse
        // them later. Not great but hey, that's why it's "fat" LTO, right?
        for module in in_memory {
            let buffer = ModuleBuffer::new(module.module_llvm.llmod());
            let llmod_id = CString::new(&module.name[..]).unwrap();
            serialized_modules.push((SerializedModule::Local(buffer), llmod_id));
        }
        // Sort the modules to ensure we produce deterministic results.
        serialized_modules.sort_by(|module1, module2| module1.1.cmp(&module2.1));

        // For all serialized bitcode files we parse them and link them in as we did
        // above, this is all mostly handled in C++.
        let mut linker = Linker::new(llmod);
        for (bc_decoded, name) in serialized_modules {
            let _timer = cgcx
                .prof
                .generic_activity_with_arg_recorder("LLVM_fat_lto_link_module", |recorder| {
                    recorder.record_arg(format!("{name:?}"))
                });
            info!("linking {:?}", name);
            let data = bc_decoded.data();
            linker.add(data).map_err(|()| write::llvm_err(dcx, LlvmError::LoadBitcode { name }))?;
        }
        drop(linker);
        save_temp_bitcode(cgcx, &module, "lto.input");

        // Internalize everything below threshold to help strip out more modules and such.
        unsafe {
            let ptr = symbols_below_threshold.as_ptr();
            llvm::LLVMRustRunRestrictionPass(
                llmod,
                ptr as *const *const libc::c_char,
                symbols_below_threshold.len() as libc::size_t,
            );
        }
        save_temp_bitcode(cgcx, &module, "lto.after-restriction");
    }

    Ok(LtoModuleCodegen::Fat(module))
}

pub(crate) struct Linker<'a>(&'a mut llvm::Linker<'a>);

impl<'a> Linker<'a> {
    pub(crate) fn new(llmod: &'a llvm::Module) -> Self {
        unsafe { Linker(llvm::LLVMRustLinkerNew(llmod)) }
    }

    pub(crate) fn add(&mut self, bytecode: &[u8]) -> Result<(), ()> {
        unsafe {
            if llvm::LLVMRustLinkerAdd(
                self.0,
                bytecode.as_ptr() as *const libc::c_char,
                bytecode.len(),
            ) {
                Ok(())
            } else {
                Err(())
            }
        }
    }
}

impl Drop for Linker<'_> {
    fn drop(&mut self) {
        unsafe {
            llvm::LLVMRustLinkerFree(&mut *(self.0 as *mut _));
        }
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
///    1. Prepare a "summary" of each LLVM module in question which describes
///       the values inside, cost of the values, etc.
///    2. Merge the summaries of all modules in question into one "index"
///    3. Perform some global analysis on this index
///    4. For each module, use the index and analysis calculated previously to
///       perform local transformations on the module, for example inlining
///       small functions from other modules.
///    5. Run thin-specific optimization passes over each module, and then code
///       generate everything at the end.
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
fn thin_lto(
    cgcx: &CodegenContext<LlvmCodegenBackend>,
    dcx: DiagCtxtHandle<'_>,
    modules: Vec<(String, ThinBuffer)>,
    serialized_modules: Vec<(SerializedModule<ModuleBuffer>, CString)>,
    cached_modules: Vec<(SerializedModule<ModuleBuffer>, WorkProduct)>,
    symbols_below_threshold: &[*const libc::c_char],
) -> Result<(Vec<LtoModuleCodegen<LlvmCodegenBackend>>, Vec<WorkProduct>), FatalError> {
    let _timer = cgcx.prof.generic_activity("LLVM_thin_lto_global_analysis");
    unsafe {
        info!("going for that thin, thin LTO");

        let green_modules: FxHashMap<_, _> =
            cached_modules.iter().map(|(_, wp)| (wp.cgu_name.clone(), wp.clone())).collect();

        let full_scope_len = modules.len() + serialized_modules.len() + cached_modules.len();
        let mut thin_buffers = Vec::with_capacity(modules.len());
        let mut module_names = Vec::with_capacity(full_scope_len);
        let mut thin_modules = Vec::with_capacity(full_scope_len);

        for (i, (name, buffer)) in modules.into_iter().enumerate() {
            info!("local module: {} - {}", i, name);
            let cname = CString::new(name.as_bytes()).unwrap();
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

        let cached_modules =
            cached_modules.into_iter().map(|(sm, wp)| (sm, CString::new(wp.cgu_name).unwrap()));

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
            thin_modules.len(),
            symbols_below_threshold.as_ptr(),
            symbols_below_threshold.len(),
        )
            .ok_or_else(|| write::llvm_err(dcx, LlvmError::PrepareThinLtoContext))?;

        let data = ThinData(data);

        info!("thin LTO data created");

        let (key_map_path, prev_key_map, curr_key_map) = if let Some(ref incr_comp_session_dir) =
            cgcx.incr_comp_session_dir
        {
            let path = incr_comp_session_dir.join(THIN_LTO_KEYS_INCR_COMP_FILE_NAME);
            // If the previous file was deleted, or we get an IO error
            // reading the file, then we'll just use `None` as the
            // prev_key_map, which will force the code to be recompiled.
            let prev =
                if path.exists() { ThinLTOKeysMap::load_from_file(&path).ok() } else { None };
            let curr = ThinLTOKeysMap::from_thin_lto_modules(&data, &thin_modules, &module_names);
            (Some(path), prev, curr)
        } else {
            // If we don't compile incrementally, we don't need to load the
            // import data from LLVM.
            assert!(green_modules.is_empty());
            let curr = ThinLTOKeysMap::default();
            (None, None, curr)
        };
        info!("thin LTO cache key map loaded");
        info!("prev_key_map: {:#?}", prev_key_map);
        info!("curr_key_map: {:#?}", curr_key_map);

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
            if let (Some(prev_key_map), true) =
                (prev_key_map.as_ref(), green_modules.contains_key(module_name))
            {
                assert!(cgcx.incr_comp_session_dir.is_some());

                // If a module exists in both the current and the previous session,
                // and has the same LTO cache key in both sessions, then we can re-use it
                if prev_key_map.keys.get(module_name) == curr_key_map.keys.get(module_name) {
                    let work_product = green_modules[module_name].clone();
                    copy_jobs.push(work_product);
                    info!(" - {}: re-used", module_name);
                    assert!(cgcx.incr_comp_session_dir.is_some());
                    continue;
                }
            }

            info!(" - {}: re-compiled", module_name);
            opt_jobs.push(LtoModuleCodegen::Thin(ThinModule {
                shared: Arc::clone(&shared),
                idx: module_index,
            }));
        }

        // Save the current ThinLTO import information for the next compilation
        // session, overwriting the previous serialized data (if any).
        if let Some(path) = key_map_path {
            if let Err(err) = curr_key_map.save_to_file(&path) {
                return Err(write::llvm_err(dcx, LlvmError::WriteThinLtoKey { err }));
            }
        }

        Ok((opt_jobs, copy_jobs))
    }
}

fn enable_autodiff_settings(ad: &[config::AutoDiff]) {
    for &val in ad {
        // We intentionally don't use a wildcard, to not forget handling anything new.
        match val {
            config::AutoDiff::PrintPerf => {
                llvm::set_print_perf(true);
            }
            config::AutoDiff::PrintAA => {
                llvm::set_print_activity(true);
            }
            config::AutoDiff::PrintTA => {
                llvm::set_print_type(true);
            }
            config::AutoDiff::Inline => {
                llvm::set_inline(true);
            }
            config::AutoDiff::LooseTypes => {
                llvm::set_loose_types(true);
            }
            config::AutoDiff::PrintSteps => {
                llvm::set_print(true);
            }
            // We handle this in the PassWrapper.cpp
            config::AutoDiff::PrintPasses => {}
            // We handle this in the PassWrapper.cpp
            config::AutoDiff::PrintModBefore => {}
            // We handle this in the PassWrapper.cpp
            config::AutoDiff::PrintModAfter => {}
            // We handle this in the PassWrapper.cpp
            config::AutoDiff::PrintModFinal => {}
            // This is required and already checked
            config::AutoDiff::Enable => {}
            // We handle this below
            config::AutoDiff::NoPostopt => {}
        }
    }
    // This helps with handling enums for now.
    llvm::set_strict_aliasing(false);
    // FIXME(ZuseZ4): Test this, since it was added a long time ago.
    llvm::set_rust_rules(true);
}

fn gen_globals<'ll>(cx: &'ll SimpleCx<'_>) -> (&'ll llvm::Type, &'ll llvm::Value, &'ll llvm::Value, &'ll llvm::Value, &'ll llvm::Value, &'ll llvm::Type) {
    let offload_entry_ty = cx.type_named_struct("struct.__tgt_offload_entry");
    let kernel_arguments_ty = cx.type_named_struct("struct.__tgt_kernel_arguments");
    let tptr = cx.type_ptr();
    let ti64 = cx.type_i64();
    let ti32 = cx.type_i32();
    let ti16 = cx.type_i16();
    let ti8 = cx.type_i8();
    let tarr = cx.type_array(ti32, 3);

    // @0 = private unnamed_addr constant [23 x i8] c";unknown;unknown;0;0;;\00", align 1
    let unknown_txt = ";unknown;unknown;0;0;;";
    let c_entry_name = CString::new(unknown_txt).unwrap();
    let c_val = c_entry_name.as_bytes_with_nul();
    let initializer = crate::common::bytes_in_context(cx.llcx, c_val);
    let at_zero = add_unnamed_global(&cx, &"", initializer, PrivateLinkage);
    llvm::set_alignment(at_zero, Align::ONE);

    // @1 = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 22, ptr @0 }, align 8
    let struct_ident_ty = cx.type_named_struct("struct.ident_t");
    let struct_elems: Vec<&llvm::Value> = vec![cx.get_const_i32(0), cx.get_const_i32(2), cx.get_const_i32(0), cx.get_const_i32(22), at_zero];
    let struct_elems_ty: Vec<_> = struct_elems.iter().map(|&x| cx.val_ty(x)).collect();
    let initializer = crate::common::named_struct(struct_ident_ty, &struct_elems);
    cx.set_struct_body(struct_ident_ty, &struct_elems_ty, false);
    let at_one = add_unnamed_global(&cx, &"", initializer, PrivateLinkage);
    llvm::set_alignment(at_one, Align::EIGHT);

    // coppied from LLVM
    // typedef struct {
    //   uint64_t Reserved;
    //   uint16_t Version;
    //   uint16_t Kind;
    //   uint32_t Flags;
    //   void *Address;
    //   char *SymbolName;
    //   uint64_t Size;
    //   uint64_t Data;
    //   void *AuxAddr;
    // } __tgt_offload_entry;
    let entry_elements = vec![ti64, ti16, ti16, ti32, tptr, tptr, ti64, ti64, tptr];
    let kernel_elements = vec![ti32, ti32, tptr, tptr, tptr, tptr, tptr, tptr, ti64, ti64, tarr, tarr, ti32];

    cx.set_struct_body(offload_entry_ty, &entry_elements, false);
    cx.set_struct_body(kernel_arguments_ty, &kernel_elements, false);
    let global = cx.declare_global("my_struct_global", offload_entry_ty);
    let global = cx.declare_global("my_struct_global2", kernel_arguments_ty);
    //@my_struct_global = external global %struct.__tgt_offload_entry
    //@my_struct_global2 = external global %struct.__tgt_kernel_arguments
    dbg!(&offload_entry_ty);
    dbg!(&kernel_arguments_ty);
    //LLVMTypeRef elements[9] = {i64Ty, i16Ty, i16Ty, i32Ty, ptrTy, ptrTy, i64Ty, i64Ty, ptrTy};
    //LLVMStructSetBody(structTy, elements, 9, 0);

    // New, to test memtransfer
    // ; Function Attrs: nounwind
    // declare void @__tgt_target_data_begin_mapper(ptr, i64, i32, ptr, ptr, ptr, ptr, ptr, ptr) #3
    //
    // ; Function Attrs: nounwind
    // declare void @__tgt_target_data_update_mapper(ptr, i64, i32, ptr, ptr, ptr, ptr, ptr, ptr) #3
    //
    // ; Function Attrs: nounwind
    // declare void @__tgt_target_data_end_mapper(ptr, i64, i32, ptr, ptr, ptr, ptr, ptr, ptr) #3

    let mapper_begin = "__tgt_target_data_begin_mapper";
    let mapper_update = String::from("__tgt_target_data_update_mapper");
    let mapper_end = String::from("__tgt_target_data_end_mapper");
    let args = vec![tptr, ti64, ti32, tptr, tptr, tptr, tptr, tptr, tptr];
    let mapper_fn_ty = cx.type_func(&args, cx.type_void());
    let foo = crate::declare::declare_simple_fn(&cx, &mapper_begin, llvm::CallConv::CCallConv, llvm::UnnamedAddr::No, llvm::Visibility::Default, mapper_fn_ty);
    let bar = crate::declare::declare_simple_fn(&cx, &mapper_update, llvm::CallConv::CCallConv, llvm::UnnamedAddr::No, llvm::Visibility::Default, mapper_fn_ty);
    let baz = crate::declare::declare_simple_fn(&cx, &mapper_end, llvm::CallConv::CCallConv, llvm::UnnamedAddr::No, llvm::Visibility::Default, mapper_fn_ty);
    let nounwind = llvm::AttributeKind::NoUnwind.create_attr(cx.llcx);
    attributes::apply_to_llfn(foo, Function, &[nounwind]);
    attributes::apply_to_llfn(bar, Function, &[nounwind]);
    attributes::apply_to_llfn(baz, Function, &[nounwind]);

    (offload_entry_ty, at_one, foo, bar, baz, mapper_fn_ty)
}

fn add_priv_unnamed_arr<'ll>(cx: &SimpleCx<'ll>, name: &str, vals: &[u64]) -> &'ll llvm::Value {
    let ti64 = cx.type_i64();
    let size_ty = cx.type_array(ti64, vals.len() as u64);
    let mut size_val = Vec::with_capacity(vals.len());
    for &val in vals {
        size_val.push(cx.get_const_i64(val));
    }
    let initializer = cx.const_array(ti64, &size_val);
    add_unnamed_global(cx, name, initializer, PrivateLinkage)
}

fn add_unnamed_global<'ll>(cx: &SimpleCx<'ll>, name: &str, initializer: &'ll llvm::Value, l: Linkage) -> &'ll llvm::Value {
    let llglobal = add_global(cx, name, initializer, l);
    unsafe {llvm::LLVMSetUnnamedAddress(llglobal, llvm::UnnamedAddr::Global)};
    llglobal
}

fn add_global<'ll>(cx: &SimpleCx<'ll>, name: &str, initializer: &'ll llvm::Value, l: Linkage) -> &'ll llvm::Value {
    let c_name = CString::new(name).unwrap();
    let llglobal: &'ll llvm::Value = llvm::add_global(cx.llmod, cx.val_ty(initializer), &c_name);
    llvm::set_global_constant(llglobal, true);
    llvm::set_linkage(llglobal, l);
    llvm::set_initializer(llglobal, initializer);
    llglobal
}



fn gen_define_handling<'ll>(cx: &'ll SimpleCx<'_>, kernel: &'ll llvm::Value, offload_entry_ty: &'ll llvm::Type, num: i64) -> &'ll llvm::Value {
    let types = cx.func_params_types(cx.get_type_of_global(kernel));
    // It seems like non-pointer values are automatically mapped. So here, we focus on pointer (or
    // reference) types.
    let num_ptr_types = types.iter().map(|&x| matches!(cx.type_kind(x), rustc_codegen_ssa::common::TypeKind::Pointer)).count();

    // We do not know their size anymore at this level, so hardcode a placeholder.
    // A follow-up pr will track these from the frontend, where we still have Rust types.
    // Then, we will be able to figure out that e.g. `&[f32;1024]` will result in 32*1024 bytes.
    // I decided that 1024 bytes is a great placeholder value for now.
    let o_sizes = add_priv_unnamed_arr(&cx, &format!(".offload_sizes.{num}"), &vec![1024;num_ptr_types]);
    // Here we figure out whether something needs to be copied to the gpu (=1), from the gpu (=2),
    // or both to and from the gpu (=3). Other values shouldn't affect us for now.
    // A non-mutable reference or pointer will be 1, an array that's not read, but fully overwritten
    // will be 2. For now, everything is 3, untill we have our frontend set up.
    let o_types = add_priv_unnamed_arr(&cx, &format!(".offload_maptypes.{num}"), &vec![3;num_ptr_types]);
    // Next: For each function, generate these three entries. A weak constant,
    // the llvm.rodata entry name, and  the omp_offloading_entries value

    // @.__omp_offloading_86fafab6_c40006a1__Z3fooPSt7complexIdES1_S0_m_l7.region_id = weak constant i8 0
    // @.offloading.entry_name = internal unnamed_addr constant [66 x i8] c"__omp_offloading_86fafab6_c40006a1__Z3fooPSt7complexIdES1_S0_m_l7\00", section ".llvm.rodata.offloading", align 1
    let name = format!(".kernel_{num}.region_id");
    let initializer = cx.get_const_i8(0);
    let region_id = add_unnamed_global(&cx, &name, initializer, WeakAnyLinkage);

    let c_entry_name = CString::new(format!("kernel_{num}")).unwrap();
    let c_val = c_entry_name.as_bytes_with_nul();
    let foo = format!(".offloading.entry_name.{num}");

    let initializer = crate::common::bytes_in_context(cx.llcx, c_val);
    let llglobal = add_unnamed_global(&cx, &foo, initializer, InternalLinkage);
    llvm::set_alignment(llglobal, Align::ONE);
    let c_section_name = CString::new(".llvm.rodata.offloading").unwrap();
    llvm::set_section(llglobal, &c_section_name);


    // Not actively used yet, for calling real kernels
    let name = format!(".offloading.entry.kernel_{num}");
    let ci64_0 = cx.get_const_i64(0);
    let ci16_1 = cx.get_const_i16(1);
    let elems: Vec<&llvm::Value> = vec![ci64_0, ci16_1, ci16_1, cx.get_const_i32(0), region_id, llglobal, ci64_0, ci64_0, cx.const_null(cx.type_ptr())];

    let initializer = crate::common::named_struct(offload_entry_ty, &elems);
    let c_name = CString::new(name).unwrap();
    let llglobal = llvm::add_global(cx.llmod, offload_entry_ty, &c_name);
    llvm::set_global_constant(llglobal, true);
    llvm::set_linkage(llglobal, WeakAnyLinkage);
    llvm::set_initializer(llglobal, initializer);
    llvm::set_alignment(llglobal, Align::ONE);
    let c_section_name = CString::new(".omp_offloading_entries").unwrap();
    llvm::set_section(llglobal, &c_section_name);
    // rustc
    // @.offloading.entry.kernel_3 = weak constant %struct.__tgt_offload_entry { i64 0, i16 1, i16 1, i32 0, ptr @.kernel_3.region_id, ptr @.offloading.entry_name.3, i64 0, i64 0, ptr null }, section ".omp_offloading_entries", align 1
    // clang
    // @.offloading.entry.__omp_offloading_86fafab6_c40006a1__Z3fooPSt7complexIdES1_S0_m_l7 = weak constant %struct.__tgt_offload_entry { i64 0, i16 1, i16 1, i32 0, ptr @.__omp_offloading_86fafab6_c40006a1__Z3fooPSt7complexIdES1_S0_m_l7.region_id, ptr @.offloading.entry_name, i64 0, i64 0, ptr null }, section "omp_offloading_entries", align 1


    //
    // 1. @.offload_sizes.{num} = private unnamed_addr constant [4 x i64] [i64 8, i64 0, i64 16, i64 0]
    // 2. @.offload_maptypes
    // 3. @.__omp_offloading_<hash>_fnc_name_<hash> = weak constant i8 0
    // 4. @.offloading.entry_name = internal unnamed_addr constant [66 x i8] c"__omp_offloading_86fafab6_c40006a1__Z3fooPSt7complexIdES1_S0_m_l7\00", section ".llvm.rodata.offloading", align 1
    // 5. @.offloading.entry.__omp_offloading_86fafab6_c40006a1__Z3fooPSt7complexIdES1_S0_m_l7 = weak constant %struct.__tgt_offload_entry { i64 0, i16 1, i16 1, i32 0, ptr @.__omp_offloading_86fafab6_c40006a1__Z3fooPSt7complexIdES1_S0_m_l7.region_id, ptr @.offloading.entry_name, i64 0, i64 0, ptr null }, section "omp_offloading_entries", align 1
    o_types
}

fn gen_call_handling<'ll>(cx: &'ll SimpleCx<'_>, kernels: &[&'ll llvm::Value], s_ident_t: &'ll llvm::Value, begin: &'ll llvm::Value, update: &'ll llvm::Value, end: &'ll llvm::Value, fn_ty: &'ll llvm::Type, o_types: &[&'ll llvm::Value]) {

    let main_fn = cx.get_function("main");
    if let Some(main_fn) = main_fn {
        let kernel_name = "kernel_1";
        let call = unsafe {llvm::LLVMRustGetFunctionCall(main_fn, kernel_name.as_c_char_ptr(), kernel_name.len())};
        let kernel_call = if call.is_some() {
            dbg!("found kernel call");
            call.unwrap()
        } else {
            return;
        };
        let kernel_call_bb = unsafe {llvm::LLVMGetInstructionParent(kernel_call)};
        let mut builder = SBuilder::build(cx, kernel_call_bb);

        let types = cx.func_params_types(cx.get_type_of_global(kernels[0]));
        dbg!(&types);
        let num_args = types.len() as u64;

        // First we generate a few variables used for the data mappers below.
        unsafe{llvm::LLVMRustPositionBuilderPastAllocas(builder.llbuilder, main_fn)};
        let ty = cx.type_array(cx.type_ptr(), num_args);

        // Baseptr are just the input pointer to the kernel, stored in a local alloca
        let a1 = builder.my_alloca2(ty, Align::EIGHT, ".offload_baseptrs");

        // Ptrs are the result of a gep into the baseptr, at least for our trivial types.
        let a2 = builder.my_alloca2(ty, Align::EIGHT, ".offload_ptrs");

        // These represent the sizes in bytes, e.g. the entry for `&[f64; 16]` will be 8*16.
        let ty2 = cx.type_array(cx.type_i64(), num_args);
        let a4 = builder.my_alloca2(ty2, Align::EIGHT, ".offload_sizes");

        // Now we generate the __tgt_target_data calls
        unsafe {llvm::LLVMRustPositionBefore(builder.llbuilder, kernel_call)};
        dbg!("positioned builder, ready");

        let i32_0 = cx.get_const_i32(0);
        let gep1 = builder.inbounds_gep(ty, a1, &[i32_0, i32_0]);
        let gep2 = builder.inbounds_gep(ty, a2, &[i32_0, i32_0]);
        let gep3 = builder.inbounds_gep(ty2, a4, &[i32_0, i32_0]);

        let nullptr = cx.const_null(cx.type_ptr());
        let o_type = o_types[0];
        let args = vec![s_ident_t, cx.get_const_i64(u64::MAX), cx.get_const_i32(num_args), gep1, gep2, gep3, o_type, nullptr, nullptr];
        builder.call(fn_ty, begin, &args, None);

        unsafe {llvm::LLVMRustPositionAfter(builder.llbuilder, kernel_call)};
        dbg!("re-positioned builder, ready");

        let gep1 = builder.inbounds_gep(ty, a1, &[i32_0, i32_0]);
        let gep2 = builder.inbounds_gep(ty, a2, &[i32_0, i32_0]);
        let gep3 = builder.inbounds_gep(ty2, a4, &[i32_0, i32_0]);

        let nullptr = cx.const_null(cx.type_ptr());
        let o_type = o_types[0];
        let args = vec![s_ident_t, cx.get_const_i64(u64::MAX), cx.get_const_i32(num_args), gep1, gep2, gep3, o_type, nullptr, nullptr];
        builder.call(fn_ty, end, &args, None);

        // 1. set insert point before kernel call.
        // 2. generate all the GEPS and stores.
        // 3. generate __tgt_target_data calls.
        //
        // unchanged: keep kernel call.
        //
        // 4. generate all the GEPS and stores.
        // 5. generate __tgt_target_data calls

        // call void @__tgt_target_data_begin_mapper(ptr @1, i64 -1, i32 3, ptr %27, ptr %28, ptr %29, ptr @.offload_maptypes, ptr null, ptr null)
        // call void @__tgt_target_data_update_mapper(ptr @1, i64 -1, i32 2, ptr %46, ptr %47, ptr %48, ptr @.offload_maptypes.1, ptr null, ptr null)
        // call void @__tgt_target_data_end_mapper(ptr @1, i64 -1, i32 3, ptr %49, ptr %50, ptr %51, ptr @.offload_maptypes, ptr null, ptr null)
        // What is @1? Random but fixed:
        // @0 = private unnamed_addr constant [23 x i8] c";unknown;unknown;0;0;;\00", align 1
        // @1 = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 22, ptr @0 }, align 8
    }
}

pub(crate) fn run_pass_manager(
    cgcx: &CodegenContext<LlvmCodegenBackend>,
    dcx: DiagCtxtHandle<'_>,
    module: &mut ModuleCodegen<ModuleLlvm>,
    thin: bool,
) -> Result<(), FatalError> {
    let _timer = cgcx.prof.generic_activity_with_arg("LLVM_lto_optimize", &*module.name);
    let config = cgcx.config(module.kind);

    // Now we have one massive module inside of llmod. Time to run the
    // LTO-specific optimization passes that LLVM provides.
    //
    // This code is based off the code found in llvm's LTO code generator:
    //      llvm/lib/LTO/LTOCodeGenerator.cpp
    debug!("running the pass manager");
    let opt_stage = if thin { llvm::OptStage::ThinLTO } else { llvm::OptStage::FatLTO };
    let opt_level = config.opt_level.unwrap_or(config::OptLevel::No);

    // The PostAD behavior is the same that we would have if no autodiff was used.
    // It will run the default optimization pipeline. If AD is enabled we select
    // the DuringAD stage, which will disable vectorization and loop unrolling, and
    // schedule two autodiff optimization + differentiation passes.
    // We then run the llvm_optimize function a second time, to optimize the code which we generated
    // in the enzyme differentiation pass.
    let enable_ad = config.autodiff.contains(&config::AutoDiff::Enable);
    let enable_gpu = true;//config.offload.contains(&config::Offload::Enable);
    let stage = if thin {
        write::AutodiffStage::PreAD
    } else {
        if enable_ad { write::AutodiffStage::DuringAD } else { write::AutodiffStage::PostAD }
    };

    if enable_ad {
        enable_autodiff_settings(&config.autodiff);
    }

    unsafe {
        write::llvm_optimize(cgcx, dcx, module, None, config, opt_level, opt_stage, stage)?;
    }

    if cfg!(llvm_enzyme) && enable_gpu && !thin {
        // first we need to add all the fun to the host module
        // %struct.__tgt_offload_entry = type { i64, i16, i16, i32, ptr, ptr, i64, i64, ptr }
        // %struct.__tgt_kernel_arguments = type { i32, i32, ptr, ptr, ptr, ptr, ptr, ptr, i64, i64, [3 x i32], [3 x i32], i32 }
        let cx =
            SimpleCx::new(module.module_llvm.llmod(), &module.module_llvm.llcx, cgcx.pointer_size);
        if cx.get_function("gen_tgt_offload").is_some() {

            let (offload_entry_ty, at_one, begin, update, end, fn_ty) = gen_globals(&cx);

            dbg!("created struct");
            let mut o_types = vec![];
            let mut kernels = vec![];
            for num in 0..9 {
                let kernel = cx.get_function(&format!("kernel_{num}"));
                if let Some(kernel) = kernel{
                    o_types.push(gen_define_handling(&cx, kernel, offload_entry_ty, num));
                    kernels.push(kernel);
                }
            }
            dbg!("gen_call_handling");
            gen_call_handling(&cx, &kernels, at_one, begin, update, end, fn_ty, &o_types);
        } else {
            dbg!("no marker found");
        }
    }

    if cfg!(llvm_enzyme) && enable_ad && !thin {
        let cx =
            SimpleCx::new(module.module_llvm.llmod(), &module.module_llvm.llcx, cgcx.pointer_size);

        for function in cx.get_functions() {
            let enzyme_marker = "enzyme_marker";
            if attributes::has_string_attr(function, enzyme_marker) {
                // Sanity check: Ensure 'noinline' is present before replacing it.
                assert!(
                    !attributes::has_attr(function, Function, llvm::AttributeKind::NoInline),
                    "Expected __enzyme function to have 'noinline' before adding 'alwaysinline'"
                );

                attributes::remove_from_llfn(function, Function, llvm::AttributeKind::NoInline);
                attributes::remove_string_attr_from_llfn(function, enzyme_marker);

                assert!(
                    !attributes::has_string_attr(function, enzyme_marker),
                    "Expected function to not have 'enzyme_marker'"
                );

                let always_inline = llvm::AttributeKind::AlwaysInline.create_attr(cx.llcx);
                attributes::apply_to_llfn(function, Function, &[always_inline]);
            }
        }

        let opt_stage = llvm::OptStage::FatLTO;
        let stage = write::AutodiffStage::PostAD;
        if !config.autodiff.contains(&config::AutoDiff::NoPostopt) {
            unsafe {
                write::llvm_optimize(cgcx, dcx, module, None, config, opt_level, opt_stage, stage)?;
            }
        }

        // This is the final IR, so people should be able to inspect the optimized autodiff output,
        // for manual inspection.
        if config.autodiff.contains(&config::AutoDiff::PrintModFinal) {
            unsafe { llvm::LLVMDumpModule(module.module_llvm.llmod()) };
        }
    }

    debug!("lto done");
    Ok(())
}

pub struct ModuleBuffer(&'static mut llvm::ModuleBuffer);

unsafe impl Send for ModuleBuffer {}
unsafe impl Sync for ModuleBuffer {}

impl ModuleBuffer {
    pub(crate) fn new(m: &llvm::Module) -> ModuleBuffer {
        ModuleBuffer(unsafe { llvm::LLVMRustModuleBufferCreate(m) })
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
        unsafe {
            llvm::LLVMRustModuleBufferFree(&mut *(self.0 as *mut _));
        }
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
    pub(crate) fn new(m: &llvm::Module, is_thin: bool, emit_summary: bool) -> ThinBuffer {
        unsafe {
            let buffer = llvm::LLVMRustThinLTOBufferCreate(m, is_thin, emit_summary);
            ThinBuffer(buffer)
        }
    }

    pub(crate) unsafe fn from_raw_ptr(ptr: *mut llvm::ThinLTOBuffer) -> ThinBuffer {
        let mut ptr = NonNull::new(ptr).unwrap();
        ThinBuffer(unsafe { ptr.as_mut() })
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

    fn thin_link_data(&self) -> &[u8] {
        unsafe {
            let ptr = llvm::LLVMRustThinLTOBufferThinLinkDataPtr(self.0) as *const _;
            let len = llvm::LLVMRustThinLTOBufferThinLinkDataLen(self.0);
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

pub(crate) fn optimize_thin_module(
    thin_module: ThinModule<LlvmCodegenBackend>,
    cgcx: &CodegenContext<LlvmCodegenBackend>,
) -> Result<ModuleCodegen<ModuleLlvm>, FatalError> {
    let dcx = cgcx.create_dcx();
    let dcx = dcx.handle();

    let module_name = &thin_module.shared.module_names[thin_module.idx];

    // Right now the implementation we've got only works over serialized
    // modules, so we create a fresh new LLVM context and parse the module
    // into that context. One day, however, we may do this for upstream
    // crates but for locally codegened modules we may be able to reuse
    // that LLVM Context and Module.
    let module_llvm = ModuleLlvm::parse(cgcx, module_name, thin_module.data(), dcx)?;
    let mut module = ModuleCodegen::new_regular(thin_module.name(), module_llvm);
    // Given that the newly created module lacks a thinlto buffer for embedding, we need to re-add it here.
    if cgcx.config(ModuleKind::Regular).embed_bitcode() {
        module.thin_lto_buffer = Some(thin_module.data().to_vec());
    }
    {
        let target = &*module.module_llvm.tm;
        let llmod = module.module_llvm.llmod();
        save_temp_bitcode(cgcx, &module, "thin-lto-input");

        // Up next comes the per-module local analyses that we do for Thin LTO.
        // Each of these functions is basically copied from the LLVM
        // implementation and then tailored to suit this implementation. Ideally
        // each of these would be supported by upstream LLVM but that's perhaps
        // a patch for another day!
        //
        // You can find some more comments about these functions in the LLVM
        // bindings we've got (currently `PassWrapper.cpp`)
        {
            let _timer =
                cgcx.prof.generic_activity_with_arg("LLVM_thin_lto_rename", thin_module.name());
            unsafe {
                llvm::LLVMRustPrepareThinLTORename(thin_module.shared.data.0, llmod, target.raw())
            };
            save_temp_bitcode(cgcx, &module, "thin-lto-after-rename");
        }

        {
            let _timer = cgcx
                .prof
                .generic_activity_with_arg("LLVM_thin_lto_resolve_weak", thin_module.name());
            if unsafe { !llvm::LLVMRustPrepareThinLTOResolveWeak(thin_module.shared.data.0, llmod) }
            {
                return Err(write::llvm_err(dcx, LlvmError::PrepareThinLtoModule));
            }
            save_temp_bitcode(cgcx, &module, "thin-lto-after-resolve");
        }

        {
            let _timer = cgcx
                .prof
                .generic_activity_with_arg("LLVM_thin_lto_internalize", thin_module.name());
            if unsafe { !llvm::LLVMRustPrepareThinLTOInternalize(thin_module.shared.data.0, llmod) }
            {
                return Err(write::llvm_err(dcx, LlvmError::PrepareThinLtoModule));
            }
            save_temp_bitcode(cgcx, &module, "thin-lto-after-internalize");
        }

        {
            let _timer =
                cgcx.prof.generic_activity_with_arg("LLVM_thin_lto_import", thin_module.name());
            if unsafe {
                !llvm::LLVMRustPrepareThinLTOImport(thin_module.shared.data.0, llmod, target.raw())
            } {
                return Err(write::llvm_err(dcx, LlvmError::PrepareThinLtoModule));
            }
            save_temp_bitcode(cgcx, &module, "thin-lto-after-import");
        }

        // Alright now that we've done everything related to the ThinLTO
        // analysis it's time to run some optimizations! Here we use the same
        // `run_pass_manager` as the "fat" LTO above except that we tell it to
        // populate a thin-specific pass manager, which presumably LLVM treats a
        // little differently.
        {
            info!("running thin lto passes over {}", module.name);
            run_pass_manager(cgcx, dcx, &mut module, true)?;
            save_temp_bitcode(cgcx, &module, "thin-lto-after-pm");
        }
    }
    Ok(module)
}

/// Maps LLVM module identifiers to their corresponding LLVM LTO cache keys
#[derive(Debug, Default)]
struct ThinLTOKeysMap {
    // key = llvm name of importing module, value = LLVM cache key
    keys: BTreeMap<String, String>,
}

impl ThinLTOKeysMap {
    fn save_to_file(&self, path: &Path) -> io::Result<()> {
        use std::io::Write;
        let mut writer = File::create_buffered(path)?;
        // The entries are loaded back into a hash map in `load_from_file()`, so
        // the order in which we write them to file here does not matter.
        for (module, key) in &self.keys {
            writeln!(writer, "{module} {key}")?;
        }
        Ok(())
    }

    fn load_from_file(path: &Path) -> io::Result<Self> {
        use std::io::BufRead;
        let mut keys = BTreeMap::default();
        let file = File::open_buffered(path)?;
        for line in file.lines() {
            let line = line?;
            let mut split = line.split(' ');
            let module = split.next().unwrap();
            let key = split.next().unwrap();
            assert_eq!(split.next(), None, "Expected two space-separated values, found {line:?}");
            keys.insert(module.to_string(), key.to_string());
        }
        Ok(Self { keys })
    }

    fn from_thin_lto_modules(
        data: &ThinData,
        modules: &[llvm::ThinLTOModule],
        names: &[CString],
    ) -> Self {
        let keys = iter::zip(modules, names)
            .map(|(module, name)| {
                let key = build_string(|rust_str| unsafe {
                    llvm::LLVMRustComputeLTOCacheKey(rust_str, module.identifier, data.0);
                })
                .expect("Invalid ThinLTO module key");
                (module_name_to_str(name).to_string(), key)
            })
        .collect();
        Self { keys }
    }
}

fn module_name_to_str(c_str: &CStr) -> &str {
    c_str.to_str().unwrap_or_else(|e| {
        bug!("Encountered non-utf8 LLVM module name `{}`: {}", c_str.to_string_lossy(), e)
    })
}

pub(crate) fn parse_module<'a>(
    cx: &'a llvm::Context,
    name: &CStr,
    data: &[u8],
    dcx: DiagCtxtHandle<'_>,
) -> Result<&'a llvm::Module, FatalError> {
    unsafe {
        llvm::LLVMRustParseBitcodeForLTO(cx, data.as_ptr(), data.len(), name.as_ptr())
            .ok_or_else(|| write::llvm_err(dcx, LlvmError::ParseBitcode))
    }
}
