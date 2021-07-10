//! Handles dynamic library loading for proc macro

use std::{
    convert::{TryFrom, TryInto},
    fmt,
    fs::File,
    io,
    path::{Path, PathBuf},
};

use libloading::Library;
use memmap2::Mmap;
use object::Object;
use proc_macro_api::{read_dylib_info, ProcMacroKind, RustCInfo};

use crate::{
    proc_macro::bridge::{self as stable_bridge, client::ProcMacro as StableProcMacro},
    rustc_server::TokenStream as StableTokenStream,
};
use crate::{
    proc_macro_nightly::bridge::{self as nightly_bridge, client::ProcMacro as NightlyProcMacro},
    rustc_server_nightly::TokenStream as NightlyTokenStream,
};

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

enum ProcMacroABI {
    Stable,
    Nightly,
}

impl TryFrom<RustCInfo> for ProcMacroABI {
    type Error = LoadProcMacroDylibError;

    fn try_from(info: RustCInfo) -> Result<Self, Self::Error> {
        if info.version.0 < 1 {
            Err(LoadProcMacroDylibError::UnsupportedABI)
        } else if info.version.1 < 47 {
            Err(LoadProcMacroDylibError::UnsupportedABI)
        } else if info.version.1 < 54 {
            Ok(ProcMacroABI::Stable)
        } else {
            Ok(ProcMacroABI::Nightly)
        }
    }
}

#[derive(Debug)]
pub enum LoadProcMacroDylibError {
    Io(io::Error),
    UnsupportedABI,
}

impl fmt::Display for LoadProcMacroDylibError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => e.fmt(f),
            Self::UnsupportedABI => write!(f, "unsupported ABI version"),
        }
    }
}

impl From<io::Error> for LoadProcMacroDylibError {
    fn from(e: io::Error) -> Self {
        LoadProcMacroDylibError::Io(e)
    }
}

enum ProcMacroLibraryLibloading {
    StableProcMacroLibrary {
        _lib: Library,
        exported_macros: Vec<crate::proc_macro::bridge::client::ProcMacro>,
    },
    NightlyProcMacroLibrary {
        _lib: Library,
        exported_macros: Vec<crate::proc_macro_nightly::bridge::client::ProcMacro>,
    },
}

impl ProcMacroLibraryLibloading {
    fn open(file: &Path) -> Result<Self, LoadProcMacroDylibError> {
        let symbol_name = find_registrar_symbol(file)?.ok_or_else(|| {
            invalid_data_err(format!("Cannot find registrar symbol in file {}", file.display()))
        })?;

        let version_info = read_dylib_info(file)?;
        let macro_abi: ProcMacroABI = version_info.try_into()?;

        let lib = load_library(file).map_err(invalid_data_err)?;
        match macro_abi {
            ProcMacroABI::Stable => {
                let macros: libloading::Symbol<&&[crate::proc_macro::bridge::client::ProcMacro]> =
                    unsafe { lib.get(symbol_name.as_bytes()) }.map_err(invalid_data_err)?;
                let macros_vec = macros.to_vec();
                Ok(ProcMacroLibraryLibloading::StableProcMacroLibrary {
                    _lib: lib,
                    exported_macros: macros_vec,
                })
            }
            ProcMacroABI::Nightly => {
                let macros: libloading::Symbol<
                    &&[crate::proc_macro_nightly::bridge::client::ProcMacro],
                > = unsafe { lib.get(symbol_name.as_bytes()) }.map_err(invalid_data_err)?;
                let macros_vec = macros.to_vec();
                Ok(ProcMacroLibraryLibloading::NightlyProcMacroLibrary {
                    _lib: lib,
                    exported_macros: macros_vec,
                })
            }
        }
    }
}

#[derive(Debug)]
pub enum PanicMessage {
    Stable(stable_bridge::PanicMessage),
    Nightly(nightly_bridge::PanicMessage),
}

impl From<stable_bridge::PanicMessage> for PanicMessage {
    fn from(p: stable_bridge::PanicMessage) -> Self {
        PanicMessage::Stable(p)
    }
}

impl From<nightly_bridge::PanicMessage> for PanicMessage {
    fn from(p: nightly_bridge::PanicMessage) -> Self {
        PanicMessage::Nightly(p)
    }
}

impl PanicMessage {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::Stable(p) => p.as_str(),
            Self::Nightly(p) => p.as_str(),
        }
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

        let library = ProcMacroLibraryLibloading::open(&lib)?;

        Ok(Expander { inner: library })
    }

    pub fn expand(
        &self,
        macro_name: &str,
        macro_body: &tt::Subtree,
        attributes: Option<&tt::Subtree>,
    ) -> Result<tt::Subtree, PanicMessage> {
        match &self.inner {
            ProcMacroLibraryLibloading::StableProcMacroLibrary { exported_macros, .. } => {
                expand_stable(macro_name, macro_body, attributes, &exported_macros[..])
                    .map_err(PanicMessage::from)
            }
            ProcMacroLibraryLibloading::NightlyProcMacroLibrary { exported_macros, .. } => {
                expand_nightly(macro_name, macro_body, attributes, &exported_macros[..])
                    .map_err(PanicMessage::from)
            }
        }
    }

    pub fn list_macros(&self) -> Vec<(String, ProcMacroKind)> {
        match &self.inner {
            ProcMacroLibraryLibloading::StableProcMacroLibrary { exported_macros, .. } => {
                list_macros_stable(&exported_macros[..])
            }
            ProcMacroLibraryLibloading::NightlyProcMacroLibrary { exported_macros, .. } => {
                list_macros_nightly(&exported_macros[..])
            }
        }
    }
}

/// Copy the dylib to temp directory to prevent locking in Windows
#[cfg(windows)]
fn ensure_file_with_lock_free_access(path: &Path) -> io::Result<PathBuf> {
    use std::collections::hash_map::RandomState;
    use std::ffi::OsString;
    use std::hash::{BuildHasher, Hasher};

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

fn expand_nightly(
    macro_name: &str,
    macro_body: &tt::Subtree,
    attributes: Option<&tt::Subtree>,
    macros: &[NightlyProcMacro],
) -> Result<tt::Subtree, crate::proc_macro_nightly::bridge::PanicMessage> {
    let parsed_body = NightlyTokenStream::with_subtree(macro_body.clone());

    let parsed_attributes = attributes
        .map_or(crate::rustc_server_nightly::TokenStream::new(), |attr| {
            NightlyTokenStream::with_subtree(attr.clone())
        });

    for proc_macro in macros {
        match proc_macro {
            crate::proc_macro_nightly::bridge::client::ProcMacro::CustomDerive {
                trait_name,
                client,
                ..
            } if *trait_name == macro_name => {
                let res = client.run(
                    &crate::proc_macro_nightly::bridge::server::SameThread,
                    crate::rustc_server_nightly::Rustc::default(),
                    parsed_body,
                    false,
                );
                return res.map(|it| it.into_subtree());
            }
            crate::proc_macro_nightly::bridge::client::ProcMacro::Bang { name, client }
                if *name == macro_name =>
            {
                let res = client.run(
                    &crate::proc_macro_nightly::bridge::server::SameThread,
                    crate::rustc_server_nightly::Rustc::default(),
                    parsed_body,
                    false,
                );
                return res.map(|it| it.into_subtree());
            }
            crate::proc_macro_nightly::bridge::client::ProcMacro::Attr { name, client }
                if *name == macro_name =>
            {
                let res = client.run(
                    &crate::proc_macro_nightly::bridge::server::SameThread,
                    crate::rustc_server_nightly::Rustc::default(),
                    parsed_attributes,
                    parsed_body,
                    false,
                );
                return res.map(|it| it.into_subtree());
            }
            _ => continue,
        }
    }

    Err(crate::proc_macro_nightly::bridge::PanicMessage::String("Nothing to expand".to_string()))
}

fn expand_stable(
    macro_name: &str,
    macro_body: &tt::Subtree,
    attributes: Option<&tt::Subtree>,
    macros: &[StableProcMacro],
) -> Result<tt::Subtree, crate::proc_macro::bridge::PanicMessage> {
    let parsed_body = StableTokenStream::with_subtree(macro_body.clone());

    let parsed_attributes = attributes.map_or(crate::rustc_server::TokenStream::new(), |attr| {
        StableTokenStream::with_subtree(attr.clone())
    });

    for proc_macro in macros {
        match proc_macro {
            crate::proc_macro::bridge::client::ProcMacro::CustomDerive {
                trait_name,
                client,
                ..
            } if *trait_name == macro_name => {
                let res = client.run(
                    &crate::proc_macro::bridge::server::SameThread,
                    crate::rustc_server::Rustc::default(),
                    parsed_body,
                    false,
                );
                return res.map(|it| it.into_subtree());
            }
            crate::proc_macro::bridge::client::ProcMacro::Bang { name, client }
                if *name == macro_name =>
            {
                let res = client.run(
                    &crate::proc_macro::bridge::server::SameThread,
                    crate::rustc_server::Rustc::default(),
                    parsed_body,
                    false,
                );
                return res.map(|it| it.into_subtree());
            }
            crate::proc_macro::bridge::client::ProcMacro::Attr { name, client }
                if *name == macro_name =>
            {
                let res = client.run(
                    &crate::proc_macro::bridge::server::SameThread,
                    crate::rustc_server::Rustc::default(),
                    parsed_attributes,
                    parsed_body,
                    false,
                );
                return res.map(|it| it.into_subtree());
            }
            _ => continue,
        }
    }

    Err(crate::proc_macro::bridge::PanicMessage::String("Nothing to expand".to_string()))
}

pub fn list_macros_stable(macros: &[StableProcMacro]) -> Vec<(String, ProcMacroKind)> {
    macros
        .iter()
        .map(|proc_macro| match proc_macro {
            crate::proc_macro::bridge::client::ProcMacro::CustomDerive { trait_name, .. } => {
                (trait_name.to_string(), ProcMacroKind::CustomDerive)
            }
            crate::proc_macro::bridge::client::ProcMacro::Bang { name, .. } => {
                (name.to_string(), ProcMacroKind::FuncLike)
            }
            crate::proc_macro::bridge::client::ProcMacro::Attr { name, .. } => {
                (name.to_string(), ProcMacroKind::Attr)
            }
        })
        .collect()
}

pub fn list_macros_nightly(macros: &[NightlyProcMacro]) -> Vec<(String, ProcMacroKind)> {
    macros
        .iter()
        .map(|proc_macro| match proc_macro {
            crate::proc_macro_nightly::bridge::client::ProcMacro::CustomDerive {
                trait_name,
                ..
            } => (trait_name.to_string(), ProcMacroKind::CustomDerive),
            crate::proc_macro_nightly::bridge::client::ProcMacro::Bang { name, .. } => {
                (name.to_string(), ProcMacroKind::FuncLike)
            }
            crate::proc_macro_nightly::bridge::client::ProcMacro::Attr { name, .. } => {
                (name.to_string(), ProcMacroKind::Attr)
            }
        })
        .collect()
}
