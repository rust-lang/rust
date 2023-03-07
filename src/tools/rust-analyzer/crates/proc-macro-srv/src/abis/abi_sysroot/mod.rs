//! Proc macro ABI

extern crate proc_macro;

#[allow(dead_code)]
#[doc(hidden)]
mod ra_server;

use libloading::Library;
use proc_macro_api::ProcMacroKind;

use super::{tt, PanicMessage};

pub use ra_server::TokenStream;

pub(crate) struct Abi {
    exported_macros: Vec<proc_macro::bridge::client::ProcMacro>,
}

impl From<proc_macro::bridge::PanicMessage> for PanicMessage {
    fn from(p: proc_macro::bridge::PanicMessage) -> Self {
        Self { message: p.as_str().map(|s| s.to_string()) }
    }
}

impl Abi {
    pub unsafe fn from_lib(lib: &Library, symbol_name: String) -> Result<Abi, libloading::Error> {
        let macros: libloading::Symbol<'_, &&[proc_macro::bridge::client::ProcMacro]> =
            lib.get(symbol_name.as_bytes())?;
        Ok(Self { exported_macros: macros.to_vec() })
    }

    pub fn expand(
        &self,
        macro_name: &str,
        macro_body: &tt::Subtree,
        attributes: Option<&tt::Subtree>,
    ) -> Result<tt::Subtree, PanicMessage> {
        let parsed_body = ra_server::TokenStream::with_subtree(macro_body.clone());

        let parsed_attributes = attributes.map_or(ra_server::TokenStream::new(), |attr| {
            ra_server::TokenStream::with_subtree(attr.clone())
        });

        for proc_macro in &self.exported_macros {
            match proc_macro {
                proc_macro::bridge::client::ProcMacro::CustomDerive {
                    trait_name, client, ..
                } if *trait_name == macro_name => {
                    let res = client.run(
                        &proc_macro::bridge::server::SameThread,
                        ra_server::RustAnalyzer::default(),
                        parsed_body,
                        true,
                    );
                    return res.map(|it| it.into_subtree()).map_err(PanicMessage::from);
                }
                proc_macro::bridge::client::ProcMacro::Bang { name, client }
                    if *name == macro_name =>
                {
                    let res = client.run(
                        &proc_macro::bridge::server::SameThread,
                        ra_server::RustAnalyzer::default(),
                        parsed_body,
                        true,
                    );
                    return res.map(|it| it.into_subtree()).map_err(PanicMessage::from);
                }
                proc_macro::bridge::client::ProcMacro::Attr { name, client }
                    if *name == macro_name =>
                {
                    let res = client.run(
                        &proc_macro::bridge::server::SameThread,
                        ra_server::RustAnalyzer::default(),
                        parsed_attributes,
                        parsed_body,
                        true,
                    );
                    return res.map(|it| it.into_subtree()).map_err(PanicMessage::from);
                }
                _ => continue,
            }
        }

        Err(proc_macro::bridge::PanicMessage::String("Nothing to expand".to_string()).into())
    }

    pub fn list_macros(&self) -> Vec<(String, ProcMacroKind)> {
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
