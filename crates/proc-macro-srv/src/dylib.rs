//! Handles dynamic library loading for proc macro

use std::{
    fmt,
    fs::File,
    io,
    path::{Path, PathBuf},
};

use libloading::Library;
use memmap2::Mmap;
use object::Object;
use paths::AbsPath;
use proc_macro_api::{read_dylib_info, ProcMacroKind};

use crate::tt;

use super::abis::Abi;

const NEW_REGISTRAR_SYMBOL: &str = "_rustc_proc_macro_decls_";

fn invalid_data_err(e: impl Into<Box<dyn std::error::Error + Send + Sync>>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, e)
}

fn is_derive_registrar_symbol(symbol: &str) -> bool {
    symbol.contains(NEW_REGISTRAR_SYMBOL)
}

fn find_registrar_symbol(file: &Path) -> io::Result<Option<String>> {
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
fn load_library(file: &Path) -> Result<Library, libloading::Error> {
    unsafe { Library::new(file) }
}

#[cfg(unix)]
fn load_library(file: &Path) -> Result<Library, libloading::Error> {
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
    UnsupportedABI(String),
}

impl fmt::Display for LoadProcMacroDylibError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => e.fmt(f),
            Self::UnsupportedABI(v) => write!(f, "unsupported ABI `{v}`"),
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
    abi: Abi,
}

impl ProcMacroLibraryLibloading {
    fn open(file: &Path) -> Result<Self, LoadProcMacroDylibError> {
        let symbol_name = find_registrar_symbol(file)?.ok_or_else(|| {
            invalid_data_err(format!("Cannot find registrar symbol in file {}", file.display()))
        })?;

        let abs_file: &AbsPath = file.try_into().map_err(|_| {
            invalid_data_err(format!("expected an absolute path, got {}", file.display()))
        })?;
        let version_info = read_dylib_info(abs_file)?;

        let lib = load_library(file).map_err(invalid_data_err)?;
        let abi = Abi::from_lib(&lib, symbol_name, version_info)?;
        Ok(ProcMacroLibraryLibloading { _lib: lib, abi })
    }
}

pub struct Expander {
    inner: ProcMacroLibraryLibloading,
}

impl Expander {
    pub fn new(lib: &Path) -> Result<Expander, LoadProcMacroDylibError> {
        // Some libraries for dynamic loading require canonicalized path even when it is
        // already absolute
        let lib = lib.canonicalize()?;

        let lib = ensure_file_with_lock_free_access(&lib)?;

        let library = ProcMacroLibraryLibloading::open(lib.as_ref())?;

        Ok(Expander { inner: library })
    }

    pub fn expand(
        &self,
        macro_name: &str,
        macro_body: &tt::Subtree,
        attributes: Option<&tt::Subtree>,
    ) -> Result<tt::Subtree, String> {
        let result = self.inner.abi.expand(macro_name, macro_body, attributes);
        result.map_err(|e| e.as_str().unwrap_or_else(|| "<unknown error>".to_string()))
    }

    pub fn list_macros(&self) -> Vec<(String, ProcMacroKind)> {
        self.inner.abi.list_macros()
    }
}

/// Copy the dylib to temp directory to prevent locking in Windows
#[cfg(windows)]
fn ensure_file_with_lock_free_access(path: &Path) -> io::Result<PathBuf> {
    use std::collections::hash_map::RandomState;
    use std::ffi::OsString;
    use std::hash::{BuildHasher, Hasher};

    if std::env::var("RA_DONT_COPY_PROC_MACRO_DLL").is_ok() {
        return Ok(path.to_path_buf());
    }

    let mut to = std::env::temp_dir();

    let file_name = path.file_name().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("File path is invalid: {}", path.display()),
        )
    })?;

    // Generate a unique number by abusing `HashMap`'s hasher.
    // Maybe this will also "inspire" a libs team member to finally put `rand` in libstd.
    let t = RandomState::new().build_hasher().finish();

    let mut unique_name = OsString::from(t.to_string());
    unique_name.push(file_name);

    to.push(unique_name);
    std::fs::copy(path, &to).unwrap();
    Ok(to)
}

#[cfg(unix)]
fn ensure_file_with_lock_free_access(path: &Path) -> io::Result<PathBuf> {
    Ok(path.to_path_buf())
}
