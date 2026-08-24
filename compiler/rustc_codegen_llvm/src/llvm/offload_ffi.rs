use std::ffi::{CStr, c_char};
use std::sync::OnceLock;

use super::ffi::{Module, TargetMachine, Value};

type LLVMRustBundleImagesFn = unsafe extern "C" fn(&Module, &TargetMachine, *const c_char) -> bool;
type LLVMRustOffloadEmbedBufferInModuleFn = unsafe extern "C" fn(&Module, *const c_char) -> bool;
type LLVMRustOffloadMapperFn = unsafe extern "C" fn(&Value, &Value, *const &Value);

use rustc_session::config::host_tuple;
use rustc_session::filesearch;

use crate::llvm::LLVMRustVersionMajor;

pub(crate) struct RustOffloadWrapper {
    LLVMRustBundleImages: LLVMRustBundleImagesFn,
    LLVMRustOffloadEmbedBufferInModule: LLVMRustOffloadEmbedBufferInModuleFn,
    LLVMRustOffloadMapper: LLVMRustOffloadMapperFn,
    // Keep the dynamic library loaded while the function pointers are used.
    _lib: libloading::Library,
}

#[derive(Debug)]
pub(crate) enum RustOffloadLibraryError {
    NotFound { err: String },
    LoadFailed { err: String },
}

impl From<libloading::Error> for RustOffloadLibraryError {
    fn from(err: libloading::Error) -> Self {
        Self::LoadFailed { err: format!("{err:?}") }
    }
}

static OFFLOAD_INSTANCE: OnceLock<RustOffloadWrapper> = OnceLock::new();

impl RustOffloadWrapper {
    pub(crate) fn get_or_init(
        sysroot: &rustc_session::config::Sysroot,
    ) -> Result<&'static RustOffloadWrapper, RustOffloadLibraryError> {
        OFFLOAD_INSTANCE.get_or_try_init(|| {
            let w = Self::call_dynamic(sysroot)?;
            Ok(w)
        })
    }

    pub(crate) fn get_instance() -> &'static RustOffloadWrapper {
        OFFLOAD_INSTANCE
            .get()
            .expect("RustOffloadWrapper not initialized. Call get_or_init with sysroot first.")
    }

    pub(crate) unsafe fn llvm_rust_bundle_images(
        &self,
        m: &Module,
        tm: &TargetMachine,
        c: &CStr,
    ) -> bool {
        unsafe { (self.LLVMRustBundleImages)(m, tm, c.as_ptr()) }
    }

    pub(crate) unsafe fn llvm_rust_offload_embed_buffer_in_module(
        &self,
        m: &Module,
        i: &CStr,
    ) -> bool {
        unsafe { (self.LLVMRustOffloadEmbedBufferInModule)(m, i.as_ptr()) }
    }

    pub(crate) unsafe fn llvm_rust_offload_wrapper(&self, v1: &Value, v2: &Value, vs: &[&Value]) {
        unsafe { (self.LLVMRustOffloadMapper)(v1, v2, vs.as_ptr()) }
    }

    fn call_dynamic(
        sysroot: &rustc_session::config::Sysroot,
    ) -> Result<Self, RustOffloadLibraryError> {
        let rust_offload_path = Self::get_rust_offload_path(sysroot)?;
        let lib = unsafe { libloading::Library::new(rust_offload_path)? };

        let llvm_rust_bundle_images =
            *unsafe { lib.get::<LLVMRustBundleImagesFn>(b"LLVMRustBundleImages\0")? };
        let llvm_rust_offload_embed_buffer_in_module = *unsafe {
            lib.get::<LLVMRustOffloadEmbedBufferInModuleFn>(
                b"LLVMRustOffloadEmbedBufferInModule\0",
            )?
        };
        let llvm_rust_offload_wrapper =
            *unsafe { lib.get::<LLVMRustOffloadMapperFn>(b"LLVMRustOffloadMapper\0")? };

        Ok(Self {
            LLVMRustBundleImages: llvm_rust_bundle_images,
            LLVMRustOffloadEmbedBufferInModule: llvm_rust_offload_embed_buffer_in_module,
            LLVMRustOffloadMapper: llvm_rust_offload_wrapper,
            _lib: lib,
        })
    }

    fn get_rust_offload_path(
        sysroot: &rustc_session::config::Sysroot,
    ) -> Result<String, RustOffloadLibraryError> {
        let llvm_version_major = unsafe { LLVMRustVersionMajor() };

        let path_buf = sysroot
            .all_paths()
            .find_map(|p| {
                let candidate = filesearch::make_target_lib_path(p, host_tuple())
                    .join(format!("libRustOffload-{}", llvm_version_major))
                    .with_extension(std::env::consts::DLL_EXTENSION);

                candidate.exists().then_some(candidate)
            })
            .ok_or_else(|| {
                let candidates = sysroot
                    .all_paths()
                    .map(|p| p.join("lib").display().to_string())
                    .collect::<Vec<String>>()
                    .join("\n* ");
                RustOffloadLibraryError::NotFound {
                    err: format!(
                        "failed to find a `libRustOffload-{llvm_version_major}` \
                    in the sysroot candidates:\n* {candidates}"
                    ),
                }
            })?;

        Ok(path_buf
            .to_str()
            .ok_or_else(|| RustOffloadLibraryError::LoadFailed {
                err: format!("invalid UTF-8 in path: {}", path_buf.display()),
            })?
            .to_string())
    }
}
