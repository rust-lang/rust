use std::ffi::{CStr, c_char};
use std::path::PathBuf;
use std::sync::OnceLock;

use super::ffi::{Module, TargetMachine, Value};

type LLVMRustBundleImagesFn = unsafe extern "C" fn(&Module, &TargetMachine, *const c_char) -> bool;
type LLVMRustOffloadEmbedBufferInModuleFn = unsafe extern "C" fn(&Module, *const c_char) -> bool;
type LLVMRustOffloadMapperFn = unsafe extern "C" fn(&Value, &Value, *const &Value);
type LLVMRustOffloadWrapImagesFn =
    unsafe extern "C" fn(&Module, *const c_char, *const c_char) -> bool;

use rustc_fs_util::path_to_c_string;
use rustc_session::config::host_tuple;
use rustc_session::filesearch;

use crate::llvm::LLVMRustVersionMajor;

pub(crate) struct RustOffloadWrapper {
    LLVMRustBundleImages: LLVMRustBundleImagesFn,
    LLVMRustOffloadEmbedBufferInModule: LLVMRustOffloadEmbedBufferInModuleFn,
    LLVMRustOffloadMapper: LLVMRustOffloadMapperFn,
    LLVMRustOffloadWrapImages: LLVMRustOffloadWrapImagesFn,
    lld_path: Option<PathBuf>,
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

    pub(crate) unsafe fn llvm_rust_offload_wrap_images(
        &self,
        host_m: &Module,
        device_bin_path: &CStr,
    ) -> bool {
        let lld_c = self.lld_path.as_deref().map(path_to_c_string).unwrap_or_default();
        unsafe {
            (self.LLVMRustOffloadWrapImages)(host_m, lld_c.as_ptr(), device_bin_path.as_ptr())
        }
    }

    fn call_dynamic(
        sysroot: &rustc_session::config::Sysroot,
    ) -> Result<Self, RustOffloadLibraryError> {
        let (rust_offload_path, lld_path) = Self::get_offload_and_lld_paths(sysroot)?;
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
        let llvm_rust_offload_wrap_images =
            *unsafe { lib.get::<LLVMRustOffloadWrapImagesFn>(b"LLVMRustOffloadWrapImages\0")? };

        Ok(Self {
            LLVMRustBundleImages: llvm_rust_bundle_images,
            LLVMRustOffloadEmbedBufferInModule: llvm_rust_offload_embed_buffer_in_module,
            LLVMRustOffloadMapper: llvm_rust_offload_wrapper,
            LLVMRustOffloadWrapImages: llvm_rust_offload_wrap_images,
            lld_path,
            _lib: lib,
        })
    }

    fn get_offload_and_lld_paths(
        sysroot: &rustc_session::config::Sysroot,
    ) -> Result<(PathBuf, Option<PathBuf>), RustOffloadLibraryError> {
        let llvm_version_major = unsafe { LLVMRustVersionMajor() };
        let mut searched = Vec::new();

        for root in sysroot.all_paths() {
            let rust_offload_path = filesearch::make_target_lib_path(root, host_tuple())
                .join(format!("libRustOffload-{llvm_version_major}"))
                .with_extension(std::env::consts::DLL_EXTENSION);

            if !rust_offload_path.is_file() {
                searched.push(rust_offload_path);
                continue;
            }

            let lld_path = filesearch::make_target_bin_path(root, host_tuple())
                .join(format!("rust-lld{}", std::env::consts::EXE_SUFFIX));
            let lld_path = lld_path.is_file().then_some(lld_path);

            return Ok((rust_offload_path, lld_path));
        }

        Err(RustOffloadLibraryError::NotFound {
            err: format!(
                "could not find libRustOffload-{llvm_version_major} in the sysroot candidates:\n* {}",
                searched.iter().map(|p| p.display().to_string()).collect::<Vec<_>>().join("\n* ")
            ),
        })
    }
}
