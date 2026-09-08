//! We take in a `device.bin` (from a previous device pass) and our current fat-lto LLVM (host)
//! module.  We clone the host module, embed the device code in it, and write it out as a
//! `host.o` object file. The `clang-linker-wrapper` then produces the final binary.

use std::path::PathBuf;

use rustc_codegen_ssa::ModuleCodegen;
use rustc_codegen_ssa::back::write::CodegenContext;
use rustc_data_structures::profiling::SelfProfilerRef;
use rustc_errors::DiagCtxtHandle;
use rustc_fs_util::path_to_c_string;
use rustc_session::config::{self, OutputType};

use crate::back::write::write_output_file;
use crate::{ModuleLlvm, llvm};

/// Embed the device image into the host module and write the result out as `host.o`.
///
/// Does nothing unless this is a host compilation with `-Zoffload=Host=<path>`.
pub(crate) fn finalize_host_module(
    cgcx: &CodegenContext,
    prof: &SelfProfilerRef,
    dcx: DiagCtxtHandle<'_>,
    module: &ModuleCodegen<ModuleLlvm>,
) {
    if cgcx.target_is_like_gpu {
        return;
    }

    let config = &cgcx.module_config;
    let Some(device_path) = config
        .offload
        .iter()
        .find_map(|o| if let config::Offload::Host(path) = o { Some(path) } else { None })
    else {
        return;
    };

    // This assumes that we previously compiled our kernels for a gpu target, which created a
    // `device.bin` artifact. The user is supposed to provide us with a path to this artifact, we
    // don't need any other artifacts from the previous run. We will embed this artifact into our
    // LLVM-IR host module, to create a `host.o` ObjectFile, which we will write to disk.
    // The last, not yet automated step uses the `clang-linker-wrapper` to process `host.o`.
    let device_pathbuf = PathBuf::from(device_path);
    if device_pathbuf.is_relative() {
        dcx.emit_err(crate::diagnostics::OffloadWithoutAbsPath);
    } else if device_pathbuf.file_name().and_then(|n| n.to_str()).is_some_and(|n| n != "device.bin")
    {
        dcx.emit_err(crate::diagnostics::OffloadWrongFileName);
    } else if !device_pathbuf.exists() {
        dcx.emit_err(crate::diagnostics::OffloadNonexistingPath);
    }
    let host_path = cgcx.output_filenames.path(OutputType::Object);
    let host_dir = host_path.parent().unwrap();
    let out_obj = host_dir.join("host.o");
    let device_bin_c = path_to_c_string(device_pathbuf.as_path());

    // Finalize host: lib.bc + device.bin -> host.o (host TM)
    // We create a full clone of our LLVM host module, since we will embed the device IR
    // into it, and this might break caching or incremental compilation otherwise.
    let llmod2 = llvm::LLVMCloneModule(module.module_llvm.llmod());
    let ok = unsafe {
        llvm::RustOffloadWrapper::get_instance()
            .llvm_rust_offload_embed_buffer_in_module(llmod2, device_bin_c.as_c_str())
    };
    if !ok {
        dcx.emit_err(crate::diagnostics::OffloadEmbedFailed);
    }
    write_output_file(
        dcx,
        module.module_llvm.tm.raw(),
        config.no_builtins,
        llmod2,
        &out_obj,
        None,
        llvm::FileType::ObjectFile,
        prof,
        true,
    );
    // We ignore cgcx.save_temps here and unconditionally always keep our `device.bin` artifact.
    // Otherwise, recompiling the host code would fail since we deleted that device artifact
    // in the previous host compilation, which would be confusing at best.
}
