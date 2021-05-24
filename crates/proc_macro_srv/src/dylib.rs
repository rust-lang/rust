//! Handles dynamic library loading for proc macro

use std::{
    fs::File,
    io,
    path::{Path, PathBuf},
};

use libloading::Library;
use memmap2::Mmap;
use object::Object;
use proc_macro_api::ProcMacroKind;

use crate::{proc_macro::bridge, rustc_server::TokenStream};

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

struct ProcMacroLibraryLibloading {
    // Hold the dylib to prevent it from unloading
    _lib: Library,
    exported_macros: Vec<bridge::client::ProcMacro>,
}

impl ProcMacroLibraryLibloading {
    fn open(file: &Path) -> io::Result<Self> {
        let symbol_name = find_registrar_symbol(file)?.ok_or_else(|| {
            invalid_data_err(format!("Cannot find registrar symbol in file {}", file.display()))
        })?;

        let lib = load_library(file).map_err(invalid_data_err)?;
        let exported_macros = {
            let macros: libloading::Symbol<&&[bridge::client::ProcMacro]> =
                unsafe { lib.get(symbol_name.as_bytes()) }.map_err(invalid_data_err)?;
            macros.to_vec()
        };

        Ok(ProcMacroLibraryLibloading { _lib: lib, exported_macros })
    }
}

pub struct Expander {
    inner: ProcMacroLibraryLibloading,
}

impl Expander {
    pub fn new(lib: &Path) -> io::Result<Expander> {
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
    ) -> Result<tt::Subtree, bridge::PanicMessage> {
        let parsed_body = TokenStream::with_subtree(macro_body.clone());

        let parsed_attributes = attributes
            .map_or(crate::rustc_server::TokenStream::new(), |attr| {
                TokenStream::with_subtree(attr.clone())
            });

        for proc_macro in &self.inner.exported_macros {
            match proc_macro {
                bridge::client::ProcMacro::CustomDerive { trait_name, client, .. }
                    if *trait_name == macro_name =>
                {
                    let res = client.run(
                        &crate::proc_macro::bridge::server::SameThread,
                        crate::rustc_server::Rustc::default(),
                        parsed_body,
                        false,
                    );
                    return res.map(|it| it.into_subtree());
                }
                bridge::client::ProcMacro::Bang { name, client } if *name == macro_name => {
                    let res = client.run(
                        &crate::proc_macro::bridge::server::SameThread,
                        crate::rustc_server::Rustc::default(),
                        parsed_body,
                        false,
                    );
                    return res.map(|it| it.into_subtree());
                }
                bridge::client::ProcMacro::Attr { name, client } if *name == macro_name => {
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

        Err(bridge::PanicMessage::String("Nothing to expand".to_string()))
    }

    pub fn list_macros(&self) -> Vec<(String, ProcMacroKind)> {
        self.inner
            .exported_macros
            .iter()
            .map(|proc_macro| match proc_macro {
                bridge::client::ProcMacro::CustomDerive { trait_name, .. } => {
                    (trait_name.to_string(), ProcMacroKind::CustomDerive)
                }
                bridge::client::ProcMacro::Bang { name, .. } => {
                    (name.to_string(), ProcMacroKind::FuncLike)
                }
                bridge::client::ProcMacro::Attr { name, .. } => {
                    (name.to_string(), ProcMacroKind::Attr)
                }
            })
            .collect()
    }
}

/// Copy the dylib to temp directory to prevent locking in Windows
#[cfg(windows)]
fn ensure_file_with_lock_free_access(path: &Path) -> io::Result<PathBuf> {
    use std::{ffi::OsString, time::SystemTime};

    let mut to = std::env::temp_dir();

    let file_name = path.file_name().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("File path is invalid: {}", path.display()),
        )
    })?;

    // generate a time deps unique number
    let t = SystemTime::now().duration_since(std::time::UNIX_EPOCH).expect("Time went backwards");

    let mut unique_name = OsString::from(t.as_millis().to_string());
    unique_name.push(file_name);

    to.push(unique_name);
    std::fs::copy(path, &to).unwrap();
    Ok(to)
}

#[cfg(unix)]
fn ensure_file_with_lock_free_access(path: &Path) -> io::Result<PathBuf> {
    Ok(path.to_path_buf())
}
