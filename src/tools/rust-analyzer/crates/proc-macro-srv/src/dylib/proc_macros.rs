//! Proc macro ABI
use crate::{
    ProcMacroClientHandle, ProcMacroKind, ProcMacroSrvSpan, TrackedEnv, token_stream::TokenStream,
};
use rustc_expand::wasm_proc_macro::RustcProcMacro;
use rustc_proc_macro::bridge;

impl From<bridge::PanicMessage> for crate::PanicMessage {
    fn from(p: bridge::PanicMessage) -> Self {
        Self { message: p.into_string() }
    }
}

pub(crate) struct ProcMacros(Vec<(RustcProcMacro, rustc_metadata::ProcMacroKind)>);

impl ProcMacros {
    pub(super) fn new(macros: Vec<(RustcProcMacro, rustc_metadata::ProcMacroKind)>) -> Self {
        ProcMacros(macros)
    }

    pub(crate) fn expand<'a, S: ProcMacroSrvSpan>(
        &self,
        macro_name: &str,
        macro_body: TokenStream<S>,
        attribute: Option<TokenStream<S>>,
        def_site: S,
        call_site: S,
        mixed_site: S,
        tracked_env: &'a mut TrackedEnv,
        callback: Option<ProcMacroClientHandle<'a>>,
    ) -> Result<TokenStream<S>, crate::PanicMessage> {
        let parsed_attributes = attribute.unwrap_or_default();

        for (client, kind) in &self.0 {
            let RustcProcMacro::Dylib { client } = client else {
                panic!("not yet implemented support for wasm proc macro, should reuse rustc impl");
            };
            match kind {
                rustc_metadata::ProcMacroKind::CustomDerive { trait_name, .. }
                    if trait_name.as_str() == macro_name =>
                {
                    let res = client.run1(
                        &bridge::server::SAME_THREAD,
                        S::make_server(call_site, def_site, mixed_site, tracked_env, callback),
                        macro_body,
                        cfg!(debug_assertions),
                    );
                    return res.map_err(crate::PanicMessage::from);
                }
                rustc_metadata::ProcMacroKind::Bang { name } if name.as_str() == macro_name => {
                    let res = client.run1(
                        &bridge::server::SAME_THREAD,
                        S::make_server(call_site, def_site, mixed_site, tracked_env, callback),
                        macro_body,
                        cfg!(debug_assertions),
                    );
                    return res.map_err(crate::PanicMessage::from);
                }
                rustc_metadata::ProcMacroKind::Attr { name } if name.as_str() == macro_name => {
                    let res = client.run2(
                        &bridge::server::SAME_THREAD,
                        S::make_server(call_site, def_site, mixed_site, tracked_env, callback),
                        parsed_attributes,
                        macro_body,
                        cfg!(debug_assertions),
                    );
                    return res.map_err(crate::PanicMessage::from);
                }
                _ => continue,
            }
        }

        Err(bridge::PanicMessage::String(format!("proc-macro `{macro_name}` is missing")).into())
    }

    pub(crate) fn list_macros(&self) -> impl Iterator<Item = (&str, ProcMacroKind)> {
        self.0.iter().map(|(_client, kind)| match kind {
            rustc_metadata::ProcMacroKind::CustomDerive { trait_name, .. } => {
                (trait_name.as_str(), ProcMacroKind::CustomDerive)
            }
            rustc_metadata::ProcMacroKind::Bang { name, .. } => {
                (name.as_str(), ProcMacroKind::Bang)
            }
            rustc_metadata::ProcMacroKind::Attr { name, .. } => {
                (name.as_str(), ProcMacroKind::Attr)
            }
        })
    }
}
