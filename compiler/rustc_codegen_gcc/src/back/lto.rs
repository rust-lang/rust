/// GCC requires to use the same toolchain for the whole compilation when doing LTO.
/// So, we need the same version/commit of the linker (gcc) and lto front-end binaries (lto1,
/// lto-wrapper, liblto_plugin.so).
// FIXME(antoyo): the executables compiled with LTO are bigger than those compiled without LTO.
// Since it is the opposite for cg_llvm, check if this is normal.
//
// Maybe we embed the bitcode in the final binary?
// It doesn't look like we try to generate fat objects for the final binary.
// Check if the way we combine the object files make it keep the LTO sections on the final link.
// Maybe that's because the combined object files contain the IR (true) and the final link
// does not remove it?
//
// TODO(antoyo): for performance, check which optimizations the C++ frontend enables.
//
// Fix these warnings:
// /usr/bin/ld: warning: type of symbol `_RNvNvNvNtCs5JWOrf9uCus_5rayon11thread_pool19WORKER_THREAD_STATE7___getit5___KEY' changed from 1 to 6 in /tmp/ccKeUSiR.ltrans0.ltrans.o
// /usr/bin/ld: warning: type of symbol `_RNvNvNvNvNtNtNtCsAj5i4SGTR7_3std4sync4mpmc5waker17current_thread_id5DUMMY7___getit5___KEY' changed from 1 to 6 in /tmp/ccKeUSiR.ltrans0.ltrans.o
// /usr/bin/ld: warning: incremental linking of LTO and non-LTO objects; using -flinker-output=nolto-rel which will bypass whole program optimization
use std::ffi::{CStr, CString};
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use gccjit::{Context, OutputKind};
use object::read::archive::ArchiveFile;
use rustc_codegen_ssa::back::lto::{LtoModuleCodegen, SerializedModule, ThinModule, ThinShared};
use rustc_codegen_ssa::back::symbol_export;
use rustc_codegen_ssa::back::write::{CodegenContext, FatLtoInput};
use rustc_codegen_ssa::traits::*;
use rustc_codegen_ssa::{ModuleCodegen, ModuleKind, looks_like_rust_object_file};
use rustc_data_structures::memmap::Mmap;
use rustc_errors::{DiagCtxtHandle, FatalError};
use rustc_hir::def_id::LOCAL_CRATE;
use rustc_middle::bug;
use rustc_middle::dep_graph::WorkProduct;
use rustc_middle::middle::exported_symbols::{SymbolExportInfo, SymbolExportLevel};
use rustc_session::config::{CrateType, Lto};
use rustc_target::spec::RelocModel;
use tempfile::{TempDir, tempdir};

use crate::back::write::save_temp_bitcode;
use crate::errors::{DynamicLinkingWithLTO, LtoBitcodeFromRlib, LtoDisallowed, LtoDylib};
use crate::{GccCodegenBackend, GccContext, SyncContext, to_gcc_opt_level};

pub fn crate_type_allows_lto(crate_type: CrateType) -> bool {
    match crate_type {
        CrateType::Executable
        | CrateType::Dylib
        | CrateType::Staticlib
        | CrateType::Cdylib
        | CrateType::Sdylib => true,
        CrateType::Rlib | CrateType::ProcMacro => false,
    }
}

struct LtoData {
    // TODO(antoyo): use symbols_below_threshold.
    //symbols_below_threshold: Vec<String>,
    upstream_modules: Vec<(SerializedModule<ModuleBuffer>, CString)>,
    tmp_path: TempDir,
}

fn prepare_lto(
    cgcx: &CodegenContext<GccCodegenBackend>,
    dcx: DiagCtxtHandle<'_>,
) -> Result<LtoData, FatalError> {
    let export_threshold = match cgcx.lto {
        // We're just doing LTO for our one crate
        Lto::ThinLocal => SymbolExportLevel::Rust,

        // We're doing LTO for the entire crate graph
        Lto::Fat | Lto::Thin => symbol_export::crates_export_threshold(&cgcx.crate_types),

        Lto::No => panic!("didn't request LTO but we're doing LTO"),
    };

    let tmp_path = match tempdir() {
        Ok(tmp_path) => tmp_path,
        Err(error) => {
            eprintln!("Cannot create temporary directory: {}", error);
            return Err(FatalError);
        }
    };

    let symbol_filter = &|&(ref name, info): &(String, SymbolExportInfo)| {
        if info.level.is_below_threshold(export_threshold) || info.used {
            Some(name.clone())
        } else {
            None
        }
    };
    let exported_symbols = cgcx.exported_symbols.as_ref().expect("needs exported symbols for LTO");
    let mut symbols_below_threshold = {
        let _timer = cgcx.prof.generic_activity("GCC_lto_generate_symbols_below_threshold");
        exported_symbols[&LOCAL_CRATE].iter().filter_map(symbol_filter).collect::<Vec<String>>()
    };
    info!("{} symbols to preserve in this crate", symbols_below_threshold.len());

    // If we're performing LTO for the entire crate graph, then for each of our
    // upstream dependencies, find the corresponding rlib and load the bitcode
    // from the archive.
    //
    // We save off all the bytecode and GCC module file path for later processing
    // with either fat or thin LTO
    let mut upstream_modules = Vec::new();
    if cgcx.lto != Lto::ThinLocal {
        // Make sure we actually can run LTO
        for crate_type in cgcx.crate_types.iter() {
            if !crate_type_allows_lto(*crate_type) {
                dcx.emit_err(LtoDisallowed);
                return Err(FatalError);
            }
            if *crate_type == CrateType::Dylib && !cgcx.opts.unstable_opts.dylib_lto {
                dcx.emit_err(LtoDylib);
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
                let _timer = cgcx.prof.generic_activity("GCC_lto_generate_symbols_below_threshold");
                symbols_below_threshold
                    .extend(exported_symbols[&cnum].iter().filter_map(symbol_filter));
            }

            let archive_data = unsafe {
                Mmap::map(File::open(path).expect("couldn't open rlib")).expect("couldn't map rlib")
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
                let path = tmp_path.path().join(name);
                match save_as_file(child.data(&*archive_data).expect("corrupt rlib"), &path) {
                    Ok(()) => {
                        let buffer = ModuleBuffer::new(path);
                        let module = SerializedModule::Local(buffer);
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

    Ok(LtoData { upstream_modules, tmp_path })
}

fn save_as_file(obj: &[u8], path: &Path) -> Result<(), LtoBitcodeFromRlib> {
    fs::write(path, obj).map_err(|error| LtoBitcodeFromRlib {
        gcc_err: format!("write object file to temp dir: {}", error),
    })
}

/// Performs fat LTO by merging all modules into a single one and returning it
/// for further optimization.
pub(crate) fn run_fat(
    cgcx: &CodegenContext<GccCodegenBackend>,
    modules: Vec<FatLtoInput<GccCodegenBackend>>,
    cached_modules: Vec<(SerializedModule<ModuleBuffer>, WorkProduct)>,
) -> Result<LtoModuleCodegen<GccCodegenBackend>, FatalError> {
    let dcx = cgcx.create_dcx();
    let dcx = dcx.handle();
    let lto_data = prepare_lto(cgcx, dcx)?;
    /*let symbols_below_threshold =
    lto_data.symbols_below_threshold.iter().map(|c| c.as_ptr()).collect::<Vec<_>>();*/
    fat_lto(
        cgcx,
        dcx,
        modules,
        cached_modules,
        lto_data.upstream_modules,
        lto_data.tmp_path,
        //&lto_data.symbols_below_threshold,
    )
}

fn fat_lto(
    cgcx: &CodegenContext<GccCodegenBackend>,
    _dcx: DiagCtxtHandle<'_>,
    modules: Vec<FatLtoInput<GccCodegenBackend>>,
    cached_modules: Vec<(SerializedModule<ModuleBuffer>, WorkProduct)>,
    mut serialized_modules: Vec<(SerializedModule<ModuleBuffer>, CString)>,
    tmp_path: TempDir,
    //symbols_below_threshold: &[String],
) -> Result<LtoModuleCodegen<GccCodegenBackend>, FatalError> {
    let _timer = cgcx.prof.generic_activity("GCC_fat_lto_build_monolithic_module");
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
        .map(|(i, _module)| {
            //let cost = unsafe { llvm::LLVMRustModuleCost(module.module_llvm.llmod()) };
            // TODO(antoyo): compute the cost of a module if GCC allows this.
            (0, i)
        })
        .max();

    // If we found a costliest module, we're good to go. Otherwise all our
    // inputs were serialized which could happen in the case, for example, that
    // all our inputs were incrementally reread from the cache and we're just
    // re-executing the LTO passes. If that's the case deserialize the first
    // module and create a linker with it.
    let mut module: ModuleCodegen<GccContext> = match costliest_module {
        Some((_cost, i)) => in_memory.remove(i),
        None => {
            unimplemented!("Incremental");
            /*assert!(!serialized_modules.is_empty(), "must have at least one serialized module");
            let (buffer, name) = serialized_modules.remove(0);
            info!("no in-memory regular modules to choose from, parsing {:?}", name);
            ModuleCodegen {
                module_llvm: GccContext::parse(cgcx, &name, buffer.data(), dcx)?,
                name: name.into_string().unwrap(),
                kind: ModuleKind::Regular,
            }*/
        }
    };
    {
        info!("using {:?} as a base module", module.name);

        // We cannot load and merge GCC contexts in memory like cg_llvm is doing.
        // Instead, we combine the object files into a single object file.
        for module in in_memory {
            let path = tmp_path.path().to_path_buf().join(&module.name);
            let path = path.to_str().expect("path");
            let context = &module.module_llvm.context;
            let config = cgcx.config(module.kind);
            // NOTE: we need to set the optimization level here in order for LTO to do its job.
            context.set_optimization_level(to_gcc_opt_level(config.opt_level));
            context.add_command_line_option("-flto=auto");
            context.add_command_line_option("-flto-partition=one");
            context.compile_to_file(OutputKind::ObjectFile, path);
            let buffer = ModuleBuffer::new(PathBuf::from(path));
            let llmod_id = CString::new(&module.name[..]).unwrap();
            serialized_modules.push((SerializedModule::Local(buffer), llmod_id));
        }
        // Sort the modules to ensure we produce deterministic results.
        serialized_modules.sort_by(|module1, module2| module1.1.cmp(&module2.1));

        // We add the object files and save in should_combine_object_files that we should combine
        // them into a single object file when compiling later.
        for (bc_decoded, name) in serialized_modules {
            let _timer = cgcx
                .prof
                .generic_activity_with_arg_recorder("GCC_fat_lto_link_module", |recorder| {
                    recorder.record_arg(format!("{:?}", name))
                });
            info!("linking {:?}", name);
            match bc_decoded {
                SerializedModule::Local(ref module_buffer) => {
                    module.module_llvm.should_combine_object_files = true;
                    module
                        .module_llvm
                        .context
                        .add_driver_option(module_buffer.0.to_str().expect("path"));
                }
                SerializedModule::FromRlib(_) => unimplemented!("from rlib"),
                SerializedModule::FromUncompressedFile(_) => {
                    unimplemented!("from uncompressed file")
                }
            }
        }
        save_temp_bitcode(cgcx, &module, "lto.input");

        // Internalize everything below threshold to help strip out more modules and such.
        /*unsafe {
        let ptr = symbols_below_threshold.as_ptr();
        llvm::LLVMRustRunRestrictionPass(
            llmod,
            ptr as *const *const libc::c_char,
            symbols_below_threshold.len() as libc::size_t,
        );*/

        save_temp_bitcode(cgcx, &module, "lto.after-restriction");
        //}
    }

    // NOTE: save the temporary directory used by LTO so that it gets deleted after linking instead
    // of now.
    module.module_llvm.temp_dir = Some(tmp_path);

    Ok(LtoModuleCodegen::Fat(module))
}

pub struct ModuleBuffer(PathBuf);

impl ModuleBuffer {
    pub fn new(path: PathBuf) -> ModuleBuffer {
        ModuleBuffer(path)
    }
}

impl ModuleBufferMethods for ModuleBuffer {
    fn data(&self) -> &[u8] {
        &[]
    }
}

/// Performs thin LTO by performing necessary global analysis and returning two
/// lists, one of the modules that need optimization and another for modules that
/// can simply be copied over from the incr. comp. cache.
pub(crate) fn run_thin(
    cgcx: &CodegenContext<GccCodegenBackend>,
    modules: Vec<(String, ThinBuffer)>,
    cached_modules: Vec<(SerializedModule<ModuleBuffer>, WorkProduct)>,
) -> Result<(Vec<LtoModuleCodegen<GccCodegenBackend>>, Vec<WorkProduct>), FatalError> {
    let dcx = cgcx.create_dcx();
    let dcx = dcx.handle();
    let lto_data = prepare_lto(cgcx, dcx)?;
    if cgcx.opts.cg.linker_plugin_lto.enabled() {
        unreachable!(
            "We should never reach this case if the LTO step \
                      is deferred to the linker"
        );
    }
    thin_lto(
        cgcx,
        dcx,
        modules,
        lto_data.upstream_modules,
        lto_data.tmp_path,
        cached_modules,
        //&lto_data.symbols_below_threshold,
    )
}

pub(crate) fn prepare_thin(
    module: ModuleCodegen<GccContext>,
    _emit_summary: bool,
) -> (String, ThinBuffer) {
    let name = module.name;
    //let buffer = ThinBuffer::new(module.module_llvm.context, true, emit_summary);
    let buffer = ThinBuffer::new(&module.module_llvm.context);
    (name, buffer)
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
    cgcx: &CodegenContext<GccCodegenBackend>,
    _dcx: DiagCtxtHandle<'_>,
    modules: Vec<(String, ThinBuffer)>,
    serialized_modules: Vec<(SerializedModule<ModuleBuffer>, CString)>,
    tmp_path: TempDir,
    cached_modules: Vec<(SerializedModule<ModuleBuffer>, WorkProduct)>,
    //_symbols_below_threshold: &[String],
) -> Result<(Vec<LtoModuleCodegen<GccCodegenBackend>>, Vec<WorkProduct>), FatalError> {
    let _timer = cgcx.prof.generic_activity("LLVM_thin_lto_global_analysis");
    info!("going for that thin, thin LTO");

    /*let green_modules: FxHashMap<_, _> =
    cached_modules.iter().map(|(_, wp)| (wp.cgu_name.clone(), wp.clone())).collect();*/

    let full_scope_len = modules.len() + serialized_modules.len() + cached_modules.len();
    let mut thin_buffers = Vec::with_capacity(modules.len());
    let mut module_names = Vec::with_capacity(full_scope_len);
    //let mut thin_modules = Vec::with_capacity(full_scope_len);

    for (i, (name, buffer)) in modules.into_iter().enumerate() {
        info!("local module: {} - {}", i, name);
        let cname = CString::new(name.as_bytes()).unwrap();
        /*thin_modules.push(llvm::ThinLTOModule {
            identifier: cname.as_ptr(),
            data: buffer.data().as_ptr(),
            len: buffer.data().len(),
        });*/
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
        /*thin_modules.push(llvm::ThinLTOModule {
            identifier: name.as_ptr(),
            data: module.data().as_ptr(),
            len: module.data().len(),
        });*/

        match module {
            SerializedModule::Local(_) => {
                //let path = module_buffer.0.to_str().expect("path");
                //let my_path = PathBuf::from(path);
                //let exists = my_path.exists();
                /*module.module_llvm.should_combine_object_files = true;
                module
                .module_llvm
                .context
                .add_driver_option(module_buffer.0.to_str().expect("path"));*/
            }
            SerializedModule::FromRlib(_) => unimplemented!("from rlib"),
            SerializedModule::FromUncompressedFile(_) => {
                unimplemented!("from uncompressed file")
            }
        }

        serialized.push(module);
        module_names.push(name);
    }

    // Sanity check
    //assert_eq!(thin_modules.len(), module_names.len());

    // Delegate to the C++ bindings to create some data here. Once this is a
    // tried-and-true interface we may wish to try to upstream some of this
    // to LLVM itself, right now we reimplement a lot of what they do
    // upstream...
    /*let data = llvm::LLVMRustCreateThinLTOData(
        thin_modules.as_ptr(),
        thin_modules.len() as u32,
        symbols_below_threshold.as_ptr(),
        symbols_below_threshold.len() as u32,
    )
    .ok_or_else(|| write::llvm_err(dcx, LlvmError::PrepareThinLtoContext))?;
    */

    let data = ThinData; //(Arc::new(tmp_path))/*(data)*/;

    info!("thin LTO data created");

    /*let (key_map_path, prev_key_map, curr_key_map) =
        if let Some(ref incr_comp_session_dir) = cgcx.incr_comp_session_dir {
            let path = incr_comp_session_dir.join(THIN_LTO_KEYS_INCR_COMP_FILE_NAME);
            // If the previous file was deleted, or we get an IO error
            // reading the file, then we'll just use `None` as the
            // prev_key_map, which will force the code to be recompiled.
            let prev =
                if path.exists() { ThinLTOKeysMap::load_from_file(&path).ok() } else { None };
            let curr = ThinLTOKeysMap::from_thin_lto_modules(&data, &thin_modules, &module_names);
            (Some(path), prev, curr)
        }
        else {
            // If we don't compile incrementally, we don't need to load the
            // import data from LLVM.
            assert!(green_modules.is_empty());
            let curr = ThinLTOKeysMap::default();
            (None, None, curr)
        };
    info!("thin LTO cache key map loaded");
    info!("prev_key_map: {:#?}", prev_key_map);
    info!("curr_key_map: {:#?}", curr_key_map);*/

    // Throw our data in an `Arc` as we'll be sharing it across threads. We
    // also put all memory referenced by the C++ data (buffers, ids, etc)
    // into the arc as well. After this we'll create a thin module
    // codegen per module in this data.
    let shared =
        Arc::new(ThinShared { data, thin_buffers, serialized_modules: serialized, module_names });

    let copy_jobs = vec![];
    let mut opt_jobs = vec![];

    info!("checking which modules can be-reused and which have to be re-optimized.");
    for (module_index, module_name) in shared.module_names.iter().enumerate() {
        let module_name = module_name_to_str(module_name);
        /*if let (Some(prev_key_map), true) =
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
        }*/

        info!(" - {}: re-compiled", module_name);
        opt_jobs
            .push(LtoModuleCodegen::Thin(ThinModule { shared: shared.clone(), idx: module_index }));
    }

    // Save the current ThinLTO import information for the next compilation
    // session, overwriting the previous serialized data (if any).
    /*if let Some(path) = key_map_path {
        if let Err(err) = curr_key_map.save_to_file(&path) {
            return Err(write::llvm_err(dcx, LlvmError::WriteThinLtoKey { err }));
        }
    }*/

    // NOTE: save the temporary directory used by LTO so that it gets deleted after linking instead
    // of now.
    //module.module_llvm.temp_dir = Some(tmp_path);
    // TODO: save the directory so that it gets deleted later.
    std::mem::forget(tmp_path);

    Ok((opt_jobs, copy_jobs))
}

pub fn optimize_thin_module(
    thin_module: ThinModule<GccCodegenBackend>,
    _cgcx: &CodegenContext<GccCodegenBackend>,
) -> Result<ModuleCodegen<GccContext>, FatalError> {
    //let dcx = cgcx.create_dcx();

    //let module_name = &thin_module.shared.module_names[thin_module.idx];
    /*let tm_factory_config = TargetMachineFactoryConfig::new(cgcx, module_name.to_str().unwrap());
    let tm = (cgcx.tm_factory)(tm_factory_config).map_err(|e| write::llvm_err(&dcx, e))?;*/

    // Right now the implementation we've got only works over serialized
    // modules, so we create a fresh new LLVM context and parse the module
    // into that context. One day, however, we may do this for upstream
    // crates but for locally codegened modules we may be able to reuse
    // that LLVM Context and Module.
    //let llcx = llvm::LLVMRustContextCreate(cgcx.fewer_names);
    //let llmod_raw = parse_module(llcx, module_name, thin_module.data(), &dcx)? as *const _;
    let mut should_combine_object_files = false;
    let context = match thin_module.shared.thin_buffers.get(thin_module.idx) {
        Some(thin_buffer) => Arc::clone(&thin_buffer.context),
        None => {
            let context = Context::default();
            let len = thin_module.shared.thin_buffers.len();
            let module = &thin_module.shared.serialized_modules[thin_module.idx - len];
            match *module {
                SerializedModule::Local(ref module_buffer) => {
                    let path = module_buffer.0.to_str().expect("path");
                    context.add_driver_option(path);
                    should_combine_object_files = true;
                    /*module.module_llvm.should_combine_object_files = true;
                    module
                        .module_llvm
                        .context
                        .add_driver_option(module_buffer.0.to_str().expect("path"));*/
                }
                SerializedModule::FromRlib(_) => unimplemented!("from rlib"),
                SerializedModule::FromUncompressedFile(_) => {
                    unimplemented!("from uncompressed file")
                }
            }
            Arc::new(SyncContext::new(context))
        }
    };
    let module = ModuleCodegen::new_regular(
        thin_module.name().to_string(),
        GccContext {
            context,
            should_combine_object_files,
            // TODO(antoyo): use the correct relocation model here.
            relocation_model: RelocModel::Pic,
            temp_dir: None,
        },
    );
    /*{
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
            unsafe { llvm::LLVMRustPrepareThinLTORename(thin_module.shared.data.0, llmod, target) };
            save_temp_bitcode(cgcx, &module, "thin-lto-after-rename");
        }

        {
            let _timer = cgcx
                .prof
                .generic_activity_with_arg("LLVM_thin_lto_resolve_weak", thin_module.name());
            if !llvm::LLVMRustPrepareThinLTOResolveWeak(thin_module.shared.data.0, llmod) {
                return Err(write::llvm_err(&dcx, LlvmError::PrepareThinLtoModule));
            }
            save_temp_bitcode(cgcx, &module, "thin-lto-after-resolve");
        }

        {
            let _timer = cgcx
                .prof
                .generic_activity_with_arg("LLVM_thin_lto_internalize", thin_module.name());
            if !llvm::LLVMRustPrepareThinLTOInternalize(thin_module.shared.data.0, llmod) {
                return Err(write::llvm_err(&dcx, LlvmError::PrepareThinLtoModule));
            }
            save_temp_bitcode(cgcx, &module, "thin-lto-after-internalize");
        }

        {
            let _timer =
                cgcx.prof.generic_activity_with_arg("LLVM_thin_lto_import", thin_module.name());
            if !llvm::LLVMRustPrepareThinLTOImport(thin_module.shared.data.0, llmod, target) {
                return Err(write::llvm_err(&dcx, LlvmError::PrepareThinLtoModule));
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
            run_pass_manager(cgcx, &dcx, &mut module, true)?;
            save_temp_bitcode(cgcx, &module, "thin-lto-after-pm");
        }
    }*/
    Ok(module)
}

pub struct ThinBuffer {
    context: Arc<SyncContext>,
}

impl ThinBuffer {
    pub(crate) fn new(context: &Arc<SyncContext>) -> Self {
        Self { context: Arc::clone(context) }
    }
}

impl ThinBufferMethods for ThinBuffer {
    fn data(&self) -> &[u8] {
        &[]
    }

    fn thin_link_data(&self) -> &[u8] {
        unimplemented!();
    }
}

pub struct ThinData; //(Arc<TempDir>);

fn module_name_to_str(c_str: &CStr) -> &str {
    c_str.to_str().unwrap_or_else(|e| {
        bug!("Encountered non-utf8 GCC module name `{}`: {}", c_str.to_string_lossy(), e)
    })
}
