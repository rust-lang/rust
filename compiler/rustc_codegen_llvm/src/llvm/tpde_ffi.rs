use std::path::PathBuf;
use std::sync::{Mutex, MutexGuard, OnceLock};

use rustc_session::config::{Sysroot, host_tuple};
use rustc_session::filesearch;

#[repr(C)]
#[repr(align(8))]
pub(crate) struct LLVMPassPluginLibraryInfo {
    _unused: [u8; 40],
}

type LLVMGetPassPluginInfoFn = unsafe extern "C" fn() -> LLVMPassPluginLibraryInfo;

pub(crate) struct TpdeWrapper {
    llvmGetPassPluginInfo: LLVMGetPassPluginInfoFn,
    // Keep the dynamic library loaded while the function pointers are used.
    _lib: libloading::Library,
}

static TPDE_INSTANCE: OnceLock<Mutex<TpdeWrapper>> = OnceLock::new();

#[derive(Debug)]
pub(crate) enum TpdeLibraryError {
    NotFound { err: String },
    LoadFailed { err: String },
}

impl From<libloading::Error> for TpdeLibraryError {
    fn from(err: libloading::Error) -> Self {
        Self::LoadFailed { err: format!("{err:?}") }
    }
}

impl TpdeWrapper {
    /// Initialize TpdeWrapper with the given sysroot if not already initialized.
    /// Safe to call multiple times - subsequent calls are no-ops due to OnceLock.
    pub(crate) fn get_or_init(
        sysroot: &rustc_session::config::Sysroot,
    ) -> Result<MutexGuard<'static, Self>, TpdeLibraryError> {
        let mtx: &'static Mutex<TpdeWrapper> = TPDE_INSTANCE.get_or_try_init(|| {
            let w = Self::call_dynamic(sysroot)?;
            Ok::<_, TpdeLibraryError>(Mutex::new(w))
        })?;

        Ok(mtx.lock().unwrap())
    }

    fn call_dynamic(sysroot: &rustc_session::config::Sysroot) -> Result<Self, TpdeLibraryError> {
        let tpde_path = Self::get_tpde_path(sysroot)?;
        let lib = unsafe { libloading::Library::new(tpde_path)? };
        let llvm_get_pass_plugin_info =
            *unsafe { lib.get::<LLVMGetPassPluginInfoFn>(b"llvmGetPassPluginInfo\0")? };
        Ok(Self { llvmGetPassPluginInfo: llvm_get_pass_plugin_info, _lib: lib })
    }

    fn get_tpde_path(sysroot: &Sysroot) -> Result<PathBuf, TpdeLibraryError> {
        let path_buf = sysroot
            .all_paths()
            .map(|sysroot_path| {
                filesearch::make_target_lib_path(sysroot_path, host_tuple())
                    .join("lib")
                    .with_file_name("tpde-plugin")
                    .with_extension(std::env::consts::DLL_EXTENSION)
            })
            .find(|f| f.exists())
            .ok_or_else(|| {
                let candidates = sysroot
                    .all_paths()
                    .map(|p| p.join("lib").display().to_string())
                    .collect::<Vec<String>>()
                    .join("\n* ");
                TpdeLibraryError::NotFound {
                    err: format!(
                        "failed to find a `tpde-plugin` library \
                    in the sysroot candidates:\n* {candidates}"
                    ),
                }
            })?;

        Ok(path_buf)
    }
}
