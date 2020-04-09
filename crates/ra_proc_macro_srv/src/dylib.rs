//! Handles dynamic library loading for proc macro

use crate::{proc_macro::bridge, rustc_server::TokenStream};
use std::fs::File;
use std::io::Read;
use std::path::Path;

use goblin::{mach::Mach, Object};
use libloading::Library;
use ra_proc_macro::ProcMacroKind;

static NEW_REGISTRAR_SYMBOL: &str = "__rustc_proc_macro_decls_";
static _OLD_REGISTRAR_SYMBOL: &str = "__rustc_derive_registrar_";

fn read_bytes(file: &Path) -> Option<Vec<u8>> {
    let mut fd = File::open(file).ok()?;
    let mut buffer = Vec::new();
    fd.read_to_end(&mut buffer).ok()?;

    Some(buffer)
}

fn get_symbols_from_lib(file: &Path) -> Option<Vec<String>> {
    let buffer = read_bytes(file)?;
    let object = Object::parse(&buffer).ok()?;

    return match object {
        Object::Elf(elf) => {
            let symbols = elf.dynstrtab.to_vec().ok()?;
            let names = symbols.iter().map(|s| s.to_string()).collect();

            Some(names)
        }

        Object::PE(pe) => {
            let symbol_names =
                pe.exports.iter().flat_map(|s| s.name).map(|n| n.to_string()).collect();
            Some(symbol_names)
        }

        Object::Mach(mach) => match mach {
            Mach::Binary(binary) => {
                let exports = binary.exports().ok()?;
                let names = exports.iter().map(|s| s.name.clone()).collect();

                Some(names)
            }

            Mach::Fat(_) => None,
        },

        Object::Archive(_) | Object::Unknown(_) => None,
    };
}

fn is_derive_registrar_symbol(symbol: &str) -> bool {
    symbol.contains(NEW_REGISTRAR_SYMBOL)
}

fn find_registrar_symbol(file: &Path) -> Option<String> {
    let symbols = get_symbols_from_lib(file)?;

    symbols.iter().find(|s| is_derive_registrar_symbol(s)).map(|s| s.to_string())
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
    Library::new(file)
}

#[cfg(unix)]
fn load_library(file: &Path) -> Result<Library, libloading::Error> {
    use libloading::os::unix::Library as UnixLibrary;
    use std::os::raw::c_int;

    const RTLD_NOW: c_int = 0x00002;
    const RTLD_DEEPBIND: c_int = 0x00008;

    UnixLibrary::open(Some(file), RTLD_NOW | RTLD_DEEPBIND).map(|lib| lib.into())
}

struct ProcMacroLibraryLibloading {
    // Hold the dylib to prevent it for unloadeding
    _lib: Library,
    exported_macros: Vec<bridge::client::ProcMacro>,
}

impl ProcMacroLibraryLibloading {
    fn open(file: &Path) -> Result<Self, String> {
        let symbol_name = find_registrar_symbol(file)
            .ok_or(format!("Cannot find registrar symbol in file {:?}", file))?;

        let lib = load_library(file).map_err(|e| e.to_string())?;

        let exported_macros = {
            let macros: libloading::Symbol<&&[bridge::client::ProcMacro]> =
                unsafe { lib.get(symbol_name.as_bytes()) }.map_err(|e| e.to_string())?;

            macros.to_vec()
        };

        Ok(ProcMacroLibraryLibloading { _lib: lib, exported_macros })
    }
}

type ProcMacroLibraryImpl = ProcMacroLibraryLibloading;

pub struct Expander {
    libs: Vec<ProcMacroLibraryImpl>,
}

impl Expander {
    pub fn new<P: AsRef<Path>>(lib: &P) -> Result<Expander, String> {
        let mut libs = vec![];

        /* Some libraries for dynamic loading require canonicalized path (even when it is
        already absolute
        */
        let lib =
            lib.as_ref().canonicalize().expect(&format!("Cannot canonicalize {:?}", lib.as_ref()));

        let library = ProcMacroLibraryImpl::open(&lib)?;
        libs.push(library);

        Ok(Expander { libs })
    }

    pub fn expand(
        &self,
        macro_name: &str,
        macro_body: &ra_tt::Subtree,
        attributes: Option<&ra_tt::Subtree>,
    ) -> Result<ra_tt::Subtree, bridge::PanicMessage> {
        let parsed_body = TokenStream::with_subtree(macro_body.clone());

        let parsed_attributes = attributes
            .map_or(crate::rustc_server::TokenStream::new(), |attr| {
                TokenStream::with_subtree(attr.clone())
            });

        for lib in &self.libs {
            for proc_macro in &lib.exported_macros {
                match proc_macro {
                    bridge::client::ProcMacro::CustomDerive { trait_name, client, .. }
                        if *trait_name == macro_name =>
                    {
                        let res = client.run(
                            &crate::proc_macro::bridge::server::SameThread,
                            crate::rustc_server::Rustc::default(),
                            parsed_body,
                        );

                        return res.map(|it| it.subtree);
                    }

                    bridge::client::ProcMacro::Bang { name, client } if *name == macro_name => {
                        let res = client.run(
                            &crate::proc_macro::bridge::server::SameThread,
                            crate::rustc_server::Rustc::default(),
                            parsed_body,
                        );

                        return res.map(|it| it.subtree);
                    }

                    bridge::client::ProcMacro::Attr { name, client } if *name == macro_name => {
                        let res = client.run(
                            &crate::proc_macro::bridge::server::SameThread,
                            crate::rustc_server::Rustc::default(),
                            parsed_attributes,
                            parsed_body,
                        );

                        return res.map(|it| it.subtree);
                    }

                    _ => {
                        continue;
                    }
                }
            }
        }

        Err(bridge::PanicMessage::String("Nothing to expand".to_string()))
    }

    pub fn list_macros(&self) -> Result<Vec<(String, ProcMacroKind)>, bridge::PanicMessage> {
        let mut result = vec![];

        for lib in &self.libs {
            for proc_macro in &lib.exported_macros {
                let res = match proc_macro {
                    bridge::client::ProcMacro::CustomDerive { trait_name, .. } => {
                        (trait_name.to_string(), ProcMacroKind::CustomDerive)
                    }
                    bridge::client::ProcMacro::Bang { name, .. } => {
                        (name.to_string(), ProcMacroKind::FuncLike)
                    }
                    bridge::client::ProcMacro::Attr { name, .. } => {
                        (name.to_string(), ProcMacroKind::Attr)
                    }
                };
                result.push(res);
            }
        }

        Ok(result)
    }
}
