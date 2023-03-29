//! Proc macro ABI

use libloading::Library;
use proc_macro_api::{ProcMacroKind, RustCInfo};

use crate::{dylib::LoadProcMacroDylibError, server::SYMBOL_INTERNER, tt};

pub(crate) struct ProcMacros {
    exported_macros: Vec<proc_macro::bridge::client::ProcMacro>,
}

impl From<proc_macro::bridge::PanicMessage> for crate::PanicMessage {
    fn from(p: proc_macro::bridge::PanicMessage) -> Self {
        Self { message: p.as_str().map(|s| s.to_string()) }
    }
}

impl ProcMacros {
    /// Load a new ABI.
    ///
    /// # Arguments
    ///
    /// *`lib` - The dynamic library containing the macro implementations
    /// *`symbol_name` - The symbol name the macros can be found attributes
    /// *`info` - RustCInfo about the compiler that was used to compile the
    ///           macro crate. This is the information we use to figure out
    ///           which ABI to return
    pub(crate) fn from_lib(
        lib: &Library,
        symbol_name: String,
        info: RustCInfo,
    ) -> Result<ProcMacros, LoadProcMacroDylibError> {
        if info.version_string == crate::RUSTC_VERSION_STRING {
            let macros = unsafe {
                lib.get::<&&[proc_macro::bridge::client::ProcMacro]>(symbol_name.as_bytes())
            }?;

            return Ok(Self { exported_macros: macros.to_vec() });
        }
        Err(LoadProcMacroDylibError::AbiMismatch(info.version_string))
    }

    pub(crate) fn expand(
        &self,
        macro_name: &str,
        macro_body: &tt::Subtree,
        attributes: Option<&tt::Subtree>,
    ) -> Result<tt::Subtree, crate::PanicMessage> {
        let parsed_body = crate::server::TokenStream::with_subtree(macro_body.clone());

        let parsed_attributes = attributes.map_or(crate::server::TokenStream::new(), |attr| {
            crate::server::TokenStream::with_subtree(attr.clone())
        });

        for proc_macro in &self.exported_macros {
            match proc_macro {
                proc_macro::bridge::client::ProcMacro::CustomDerive {
                    trait_name, client, ..
                } if *trait_name == macro_name => {
                    let res = client.run(
                        &proc_macro::bridge::server::SameThread,
                        crate::server::RustAnalyzer { interner: &SYMBOL_INTERNER },
                        parsed_body,
                        true,
                    );
                    return res.map(|it| it.into_subtree()).map_err(crate::PanicMessage::from);
                }
                proc_macro::bridge::client::ProcMacro::Bang { name, client }
                    if *name == macro_name =>
                {
                    let res = client.run(
                        &proc_macro::bridge::server::SameThread,
                        crate::server::RustAnalyzer { interner: &SYMBOL_INTERNER },
                        parsed_body,
                        true,
                    );
                    return res.map(|it| it.into_subtree()).map_err(crate::PanicMessage::from);
                }
                proc_macro::bridge::client::ProcMacro::Attr { name, client }
                    if *name == macro_name =>
                {
                    let res = client.run(
                        &proc_macro::bridge::server::SameThread,
                        crate::server::RustAnalyzer { interner: &SYMBOL_INTERNER },
                        parsed_attributes,
                        parsed_body,
                        true,
                    );
                    return res.map(|it| it.into_subtree()).map_err(crate::PanicMessage::from);
                }
                _ => continue,
            }
        }

        Err(proc_macro::bridge::PanicMessage::String("Nothing to expand".to_string()).into())
    }

    pub(crate) fn list_macros(&self) -> Vec<(String, ProcMacroKind)> {
        self.exported_macros
            .iter()
            .map(|proc_macro| match proc_macro {
                proc_macro::bridge::client::ProcMacro::CustomDerive { trait_name, .. } => {
                    (trait_name.to_string(), ProcMacroKind::CustomDerive)
                }
                proc_macro::bridge::client::ProcMacro::Bang { name, .. } => {
                    (name.to_string(), ProcMacroKind::FuncLike)
                }
                proc_macro::bridge::client::ProcMacro::Attr { name, .. } => {
                    (name.to_string(), ProcMacroKind::Attr)
                }
            })
            .collect()
    }
}

#[test]
fn test_version_check() {
    let path = paths::AbsPathBuf::assert(crate::proc_macro_test_dylib_path());
    let info = proc_macro_api::read_dylib_info(&path).unwrap();
    assert_eq!(
        info.version_string,
        crate::RUSTC_VERSION_STRING,
        "sysroot ABI mismatch: dylib rustc version (read from .rustc section): {:?} != proc-macro-srv version (read from 'rustc --version'): {:?}",
        info.version_string,
        crate::RUSTC_VERSION_STRING,
    );
}
