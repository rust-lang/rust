use std::path::PathBuf;
use std::sync::{Mutex, MutexGuard, OnceLock};

use rustc_session::config::{self, Sysroot, host_tuple};
use rustc_session::filesearch;

// Matches v2 of the LLVM plugin ABI
#[repr(C)]
#[repr(align(8))]
pub(crate) struct LLVMPassPluginLibraryInfo {
    _unused: [u8; 40],
}

type LLVMGetPassPluginInfoFn = unsafe extern "C" fn() -> LLVMPassPluginLibraryInfo;

/// LLVMRustTpdeMode
#[repr(C)]
#[derive(Clone)]
pub(crate) enum LLVMRustTpdeMode {
    None,
    Only,
    Try,
}

/// LLVMRustTpdeOptions
#[repr(C)]
#[derive(Clone)]
pub(crate) struct LLVMRustTpdeOptions {
    mode: LLVMRustTpdeMode,
    // `None` if mode is `LLVMRustTpdeMode::None`
    plugin: Option<LLVMGetPassPluginInfoFn>,
}

pub(crate) struct TpdeWrapper {
    options: LLVMRustTpdeOptions,
    // Keep the dynamic library loaded while the function pointers are used.
    _lib: Option<libloading::Library>,
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
    /// Initialize TpdeWrapper with the given sysroot and mode if not already initialized.
    /// Safe to call multiple times - subsequent calls are no-ops due to OnceLock.
    pub(crate) fn get_or_init(
        sysroot: &rustc_session::config::Sysroot,
        mode: &Option<config::Tpde>,
    ) -> Result<MutexGuard<'static, Self>, TpdeLibraryError> {
        let mtx: &'static Mutex<TpdeWrapper> = TPDE_INSTANCE.get_or_try_init(|| {
            let w = match mode {
                Some(mode) => Self::call_dynamic(sysroot, mode)?,
                None => Self {
                    options: LLVMRustTpdeOptions { mode: LLVMRustTpdeMode::None, plugin: None },
                    _lib: None,
                },
            };
            Ok::<_, TpdeLibraryError>(Mutex::new(w))
        })?;

        Ok(mtx.lock().unwrap())
    }

    /// Get the TpdeWrapper instance. Panics if not initialized.
    pub(crate) fn get_instance() -> MutexGuard<'static, Self> {
        TPDE_INSTANCE
            .get()
            .expect("TpdeWrapper not initialized. Call get_or_init with sysroot first.")
            .lock()
            .unwrap()
    }

    pub(crate) fn options(&self) -> &LLVMRustTpdeOptions {
        &self.options
    }

    fn call_dynamic(
        sysroot: &rustc_session::config::Sysroot,
        mode: &config::Tpde,
    ) -> Result<Self, TpdeLibraryError> {
        let mode = match mode {
            config::Tpde::Only => LLVMRustTpdeMode::Only,
            config::Tpde::Try => LLVMRustTpdeMode::Try,
        };
        let tpde_path = Self::get_tpde_path(sysroot)?;
        let lib = unsafe { libloading::Library::new(tpde_path)? };
        let llvm_get_pass_plugin_info =
            *unsafe { lib.get::<LLVMGetPassPluginInfoFn>(b"llvmGetPassPluginInfo\0")? };
        let options = LLVMRustTpdeOptions { mode, plugin: Some(llvm_get_pass_plugin_info) };
        Ok(Self { options, _lib: Some(lib) })
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
                        "failed to find the `tpde-plugin` shared library \
                    in the sysroot candidates:\n* {candidates}"
                    ),
                }
            })?;

        Ok(path_buf)
    }
}
