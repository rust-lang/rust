use std::ffi::{CStr, c_char};
use std::path::PathBuf;
use std::sync::OnceLock;

use super::ffi::{Module, Value};

type LLVMRustOffloadMapperFn = unsafe extern "C" fn(&Value, &Value, *const &Value);
type LLVMRustOffloadWrapImagesFn = unsafe extern "C" fn(&Module, *const c_char) -> bool;

use rustc_session::config::host_tuple;
use rustc_session::filesearch;

use crate::llvm;

pub(crate) struct RustOffloadWrapper {
    LLVMRustOffloadMapper: LLVMRustOffloadMapperFn,
    LLVMRustOffloadWrapImages: LLVMRustOffloadWrapImagesFn,
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

    pub(crate) unsafe fn llvm_rust_offload_wrapper(&self, v1: &Value, v2: &Value, vs: &[&Value]) {
        unsafe { (self.LLVMRustOffloadMapper)(v1, v2, vs.as_ptr()) }
    }

    pub(crate) unsafe fn llvm_rust_offload_wrap_images(
        &self,
        host_m: &Module,
        device_image_path: &CStr,
    ) -> bool {
        unsafe { (self.LLVMRustOffloadWrapImages)(host_m, device_image_path.as_ptr()) }
    }

    fn call_dynamic(
        sysroot: &rustc_session::config::Sysroot,
    ) -> Result<Self, RustOffloadLibraryError> {
        let rust_offload_path = Self::get_offload_path(sysroot)?;
        let lib = unsafe { libloading::Library::new(rust_offload_path)? };

        let llvm_rust_offload_wrapper =
            *unsafe { lib.get::<LLVMRustOffloadMapperFn>(b"LLVMRustOffloadMapper\0")? };
        let llvm_rust_offload_wrap_images =
            *unsafe { lib.get::<LLVMRustOffloadWrapImagesFn>(b"LLVMRustOffloadWrapImages\0")? };

        Ok(Self {
            LLVMRustOffloadMapper: llvm_rust_offload_wrapper,
            LLVMRustOffloadWrapImages: llvm_rust_offload_wrap_images,
            _lib: lib,
        })
    }

    fn get_offload_path(
        sysroot: &rustc_session::config::Sysroot,
    ) -> Result<PathBuf, RustOffloadLibraryError> {
        let llvm_version_major = llvm::LLVMRustVersionMajor();
        let mut searched = Vec::new();

        for root in sysroot.all_paths() {
            let rust_offload_path = filesearch::make_target_lib_path(root, host_tuple())
                .join(format!("libRustOffload-{llvm_version_major}"))
                .with_extension(std::env::consts::DLL_EXTENSION);

            if !rust_offload_path.is_file() {
                searched.push(rust_offload_path);
                continue;
            }

            return Ok(rust_offload_path);
        }

        Err(RustOffloadLibraryError::NotFound {
            err: format!(
                "could not find libRustOffload-{llvm_version_major} in the sysroot candidates:\n* {}",
                searched.iter().map(|p| p.display().to_string()).collect::<Vec<_>>().join("\n* ")
            ),
        })
    }
}
