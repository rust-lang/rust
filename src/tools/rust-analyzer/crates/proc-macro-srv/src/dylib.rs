//! Handles dynamic library loading for proc macro

mod version;

use proc_macro::bridge;
use std::{fmt, fs::File, io};

use libloading::Library;
use memmap2::Mmap;
use object::Object;
use paths::{AbsPath, Utf8Path, Utf8PathBuf};
use proc_macro_api::ProcMacroKind;

use crate::ProcMacroSrvSpan;

const NEW_REGISTRAR_SYMBOL: &str = "_rustc_proc_macro_decls_";

fn invalid_data_err(e: impl Into<Box<dyn std::error::Error + Send + Sync>>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, e)
}

fn is_derive_registrar_symbol(symbol: &str) -> bool {
    symbol.contains(NEW_REGISTRAR_SYMBOL)
}

fn find_registrar_symbol(file: &Utf8Path) -> io::Result<Option<String>> {
    let file = File::open(file)?;
    let buffer = unsafe { Mmap::map(&file)? };

    Ok(object::File::parse(&*buffer)
        .map_err(invalid_data_err)?
        .exports()
        .map_err(invalid_data_err)?
        .into_iter()
        .map(|export| export.name())
        .filter_map(|sym| String::from_utf8(sym.into()).ok())
        .find(|sym| is_derive_registrar_symbol(sym))
        .map(|sym| {
            // From MacOS docs:
            // https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man3/dlsym.3.html
            // Unlike other dyld API's, the symbol name passed to dlsym() must NOT be
            // prepended with an underscore.
            if cfg!(target_os = "macos") && sym.starts_with('_') {
                sym[1..].to_owned()
            } else {
                sym
            }
        }))
}

/// Loads dynamic library in platform dependent manner.
///
/// For unix, you have to use RTLD_DEEPBIND flag to escape problems described
/// [here](https://github.com/fedochet/rust-proc-macro-panic-inside-panic-expample)
/// and [here](https://github.com/rust-lang/rust/issues/60593).
///
/// Usage of RTLD_DEEPBIND
/// [here](https://github.com/fedochet/rust-proc-macro-panic-inside-panic-expample/issues/1)
///
/// It seems that on Windows that behaviour is default, so we do nothing in that case.
#[cfg(windows)]
fn load_library(file: &Utf8Path) -> Result<Library, libloading::Error> {
    unsafe { Library::new(file) }
}

#[cfg(unix)]
fn load_library(file: &Utf8Path) -> Result<Library, libloading::Error> {
    use libloading::os::unix::Library as UnixLibrary;
    use std::os::raw::c_int;

    const RTLD_NOW: c_int = 0x00002;
    const RTLD_DEEPBIND: c_int = 0x00008;

    unsafe { UnixLibrary::open(Some(file), RTLD_NOW | RTLD_DEEPBIND).map(|lib| lib.into()) }
}

#[derive(Debug)]
pub enum LoadProcMacroDylibError {
    Io(io::Error),
    LibLoading(libloading::Error),
    AbiMismatch(String),
}

impl fmt::Display for LoadProcMacroDylibError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => e.fmt(f),
            Self::AbiMismatch(v) => {
                use crate::RUSTC_VERSION_STRING;
                write!(f, "mismatched ABI expected: `{RUSTC_VERSION_STRING}`, got `{v}`")
            }
            Self::LibLoading(e) => e.fmt(f),
        }
    }
}

impl From<io::Error> for LoadProcMacroDylibError {
    fn from(e: io::Error) -> Self {
        LoadProcMacroDylibError::Io(e)
    }
}

impl From<libloading::Error> for LoadProcMacroDylibError {
    fn from(e: libloading::Error) -> Self {
        LoadProcMacroDylibError::LibLoading(e)
    }
}

struct ProcMacroLibraryLibloading {
    // Hold on to the library so it doesn't unload
    _lib: Library,
    proc_macros: crate::proc_macros::ProcMacros,
}

impl ProcMacroLibraryLibloading {
    fn open(file: &Utf8Path) -> Result<Self, LoadProcMacroDylibError> {
        let symbol_name = find_registrar_symbol(file)?.ok_or_else(|| {
            invalid_data_err(format!("Cannot find registrar symbol in file {file}"))
        })?;

        let abs_file: &AbsPath = file
            .try_into()
            .map_err(|_| invalid_data_err(format!("expected an absolute path, got {file}")))?;
        let version_info = version::read_dylib_info(abs_file)?;

        let lib = load_library(file).map_err(invalid_data_err)?;
        let proc_macros = crate::proc_macros::ProcMacros::from_lib(
            &lib,
            symbol_name,
            &version_info.version_string,
        )?;
        Ok(ProcMacroLibraryLibloading { _lib: lib, proc_macros })
    }
}

pub(crate) struct Expander {
    inner: ProcMacroLibraryLibloading,
    path: Utf8PathBuf,
}

impl Drop for Expander {
    fn drop(&mut self) {
        #[cfg(windows)]
        std::fs::remove_file(&self.path).ok();
        _ = self.path;
    }
}

impl Expander {
    pub(crate) fn new(lib: &Utf8Path) -> Result<Expander, LoadProcMacroDylibError> {
        // Some libraries for dynamic loading require canonicalized path even when it is
        // already absolute
        let lib = lib.canonicalize_utf8()?;

        let path = ensure_file_with_lock_free_access(&lib)?;

        let library = ProcMacroLibraryLibloading::open(path.as_ref())?;

        Ok(Expander { inner: library, path })
    }

    pub(crate) fn expand<S: ProcMacroSrvSpan>(
        &self,
        macro_name: &str,
        macro_body: tt::Subtree<S>,
        attributes: Option<tt::Subtree<S>>,
        def_site: S,
        call_site: S,
        mixed_site: S,
    ) -> Result<tt::Subtree<S>, String>
    where
        <S::Server as bridge::server::Types>::TokenStream: Default,
    {
        let result = self
            .inner
            .proc_macros
            .expand(macro_name, macro_body, attributes, def_site, call_site, mixed_site);
        result.map_err(|e| e.into_string().unwrap_or_default())
    }

    pub(crate) fn list_macros(&self) -> Vec<(String, ProcMacroKind)> {
        self.inner.proc_macros.list_macros()
    }
}

/// Copy the dylib to temp directory to prevent locking in Windows
#[cfg(windows)]
fn ensure_file_with_lock_free_access(path: &Utf8Path) -> io::Result<Utf8PathBuf> {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};

    if std::env::var("RA_DONT_COPY_PROC_MACRO_DLL").is_ok() {
        return Ok(path.to_path_buf());
    }

    let mut to = Utf8PathBuf::from_path_buf(std::env::temp_dir()).unwrap();

    let file_name = path.file_name().ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidInput, format!("File path is invalid: {path}"))
    })?;

    // Generate a unique number by abusing `HashMap`'s hasher.
    // Maybe this will also "inspire" a libs team member to finally put `rand` in libstd.
    let t = RandomState::new().build_hasher().finish();

    let mut unique_name = t.to_string();
    unique_name.push_str(file_name);

    to.push(unique_name);
    std::fs::copy(path, &to)?;
    Ok(to)
}

#[cfg(unix)]
fn ensure_file_with_lock_free_access(path: &Utf8Path) -> io::Result<Utf8PathBuf> {
    Ok(path.to_owned())
}
