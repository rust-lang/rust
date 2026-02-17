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
// cSpell:disable
// Fix these warnings:
// /usr/bin/ld: warning: type of symbol `_RNvNvNvNtCs5JWOrf9uCus_5rayon11thread_pool19WORKER_THREAD_STATE7___getit5___KEY' changed from 1 to 6 in /tmp/ccKeUSiR.ltrans0.ltrans.o
// /usr/bin/ld: warning: type of symbol `_RNvNvNvNvNtNtNtCsAj5i4SGTR7_3std4sync4mpmc5waker17current_thread_id5DUMMY7___getit5___KEY' changed from 1 to 6 in /tmp/ccKeUSiR.ltrans0.ltrans.o
// /usr/bin/ld: warning: incremental linking of LTO and non-LTO objects; using -flinker-output=nolto-rel which will bypass whole program optimization
// cSpell:enable
use std::ffi::CString;
use std::fs::{self, File};
use std::path::{Path, PathBuf};

use gccjit::OutputKind;
use object::read::archive::ArchiveFile;
use rustc_codegen_ssa::back::lto::SerializedModule;
use rustc_codegen_ssa::back::write::{CodegenContext, FatLtoInput, SharedEmitter};
use rustc_codegen_ssa::traits::*;
use rustc_codegen_ssa::{ModuleCodegen, ModuleKind, looks_like_rust_object_file};
use rustc_data_structures::memmap::Mmap;
use rustc_data_structures::profiling::SelfProfilerRef;
use rustc_errors::{DiagCtxt, DiagCtxtHandle};
use rustc_log::tracing::info;
use rustc_session::config::Lto;
use tempfile::{TempDir, tempdir};

use crate::back::write::save_temp_bitcode;
use crate::errors::LtoBitcodeFromRlib;
use crate::{GccCodegenBackend, GccContext, LtoMode, to_gcc_opt_level};

struct LtoData {
    // TODO(antoyo): use symbols_below_threshold.
    //symbols_below_threshold: Vec<String>,
    upstream_modules: Vec<(SerializedModule<ModuleBuffer>, CString)>,
    tmp_path: TempDir,
}

fn prepare_lto(
    cgcx: &CodegenContext,
    each_linked_rlib_for_lto: &[PathBuf],
    dcx: DiagCtxtHandle<'_>,
) -> LtoData {
    let tmp_path = match tempdir() {
        Ok(tmp_path) => tmp_path,
        Err(error) => {
            dcx.fatal(format!("Cannot create temporary directory: {}", error));
        }
    };

    // If we're performing LTO for the entire crate graph, then for each of our
    // upstream dependencies, find the corresponding rlib and load the bitcode
    // from the archive.
    //
    // We save off all the bytecode and GCC module file path for later processing
    // with either fat or thin LTO
    let mut upstream_modules = Vec::new();
    if cgcx.lto != Lto::ThinLocal {
        for path in each_linked_rlib_for_lto {
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
                        dcx.emit_fatal(e);
                    }
                }
            }
        }
    }

    LtoData { upstream_modules, tmp_path }
}

fn save_as_file(obj: &[u8], path: &Path) -> Result<(), LtoBitcodeFromRlib> {
    fs::write(path, obj).map_err(|error| LtoBitcodeFromRlib {
        gcc_err: format!("write object file to temp dir: {}", error),
    })
}

/// Performs fat LTO by merging all modules into a single one and returning it
/// for further optimization.
pub(crate) fn run_fat(
    cgcx: &CodegenContext,
    prof: &SelfProfilerRef,
    shared_emitter: &SharedEmitter,
    each_linked_rlib_for_lto: &[PathBuf],
    modules: Vec<FatLtoInput<GccCodegenBackend>>,
) -> ModuleCodegen<GccContext> {
    let dcx = DiagCtxt::new(Box::new(shared_emitter.clone()));
    let dcx = dcx.handle();
    let lto_data = prepare_lto(cgcx, each_linked_rlib_for_lto, dcx);
    /*let symbols_below_threshold =
    lto_data.symbols_below_threshold.iter().map(|c| c.as_ptr()).collect::<Vec<_>>();*/
    fat_lto(
        cgcx,
        prof,
        dcx,
        modules,
        lto_data.upstream_modules,
        lto_data.tmp_path,
        //&lto_data.symbols_below_threshold,
    )
}

fn fat_lto(
    cgcx: &CodegenContext,
    prof: &SelfProfilerRef,
    _dcx: DiagCtxtHandle<'_>,
    modules: Vec<FatLtoInput<GccCodegenBackend>>,
    mut serialized_modules: Vec<(SerializedModule<ModuleBuffer>, CString)>,
    tmp_path: TempDir,
    //symbols_below_threshold: &[String],
) -> ModuleCodegen<GccContext> {
    let _timer = prof.generic_activity("GCC_fat_lto_build_monolithic_module");
    info!("going for a fat lto");

    // Sort out all our lists of incoming modules into two lists.
    //
    // * `serialized_modules` (also and argument to this function) contains all
    //   modules that are serialized in-memory.
    // * `in_memory` contains modules which are already parsed and in-memory,
    //   such as from multi-CGU builds.
    let mut in_memory = Vec::new();
    for module in modules {
        match module {
            FatLtoInput::InMemory(m) => in_memory.push(m),
            FatLtoInput::Serialized { name, buffer } => {
                info!("pushing serialized module {:?}", name);
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
            let config = &cgcx.module_config;
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
            let _timer = prof
                .generic_activity_with_arg_recorder("GCC_fat_lto_link_module", |recorder| {
                    recorder.record_arg(format!("{:?}", name))
                });
            info!("linking {:?}", name);
            match bc_decoded {
                SerializedModule::Local(ref module_buffer) => {
                    module.module_llvm.lto_mode = LtoMode::Fat;
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

    module
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
