//! Proc macro ABI

use proc_macro::bridge;

use crate::{ProcMacroKind, ProcMacroSrvSpan, server_impl::TopSubtree};

#[repr(transparent)]
pub(crate) struct ProcMacros([bridge::client::ProcMacro]);

impl From<bridge::PanicMessage> for crate::PanicMessage {
    fn from(p: bridge::PanicMessage) -> Self {
        Self { message: p.as_str().map(|s| s.to_owned()) }
    }
}

impl ProcMacros {
    pub(crate) fn expand<S: ProcMacroSrvSpan>(
        &self,
        macro_name: &str,
        macro_body: TopSubtree<S>,
        attributes: Option<TopSubtree<S>>,
        def_site: S,
        call_site: S,
        mixed_site: S,
    ) -> Result<TopSubtree<S>, crate::PanicMessage> {
        let parsed_body = crate::server_impl::TokenStream::with_subtree(macro_body);

        let parsed_attributes = attributes
            .map_or_else(crate::server_impl::TokenStream::default, |attr| {
                crate::server_impl::TokenStream::with_subtree(attr)
            });

        for proc_macro in &self.0 {
            match proc_macro {
                bridge::client::ProcMacro::CustomDerive { trait_name, client, .. }
                    if *trait_name == macro_name =>
                {
                    let res = client.run(
                        &bridge::server::SameThread,
                        S::make_server(call_site, def_site, mixed_site),
                        parsed_body,
                        cfg!(debug_assertions),
                    );
                    return res
                        .map(|it| it.into_subtree(call_site))
                        .map_err(crate::PanicMessage::from);
                }
                bridge::client::ProcMacro::Bang { name, client } if *name == macro_name => {
                    let res = client.run(
                        &bridge::server::SameThread,
                        S::make_server(call_site, def_site, mixed_site),
                        parsed_body,
                        cfg!(debug_assertions),
                    );
                    return res
                        .map(|it| it.into_subtree(call_site))
                        .map_err(crate::PanicMessage::from);
                }
                bridge::client::ProcMacro::Attr { name, client } if *name == macro_name => {
                    let res = client.run(
                        &bridge::server::SameThread,
                        S::make_server(call_site, def_site, mixed_site),
                        parsed_attributes,
                        parsed_body,
                        cfg!(debug_assertions),
                    );
                    return res
                        .map(|it| it.into_subtree(call_site))
                        .map_err(crate::PanicMessage::from);
                }
                _ => continue,
            }
        }

        Err(bridge::PanicMessage::String(format!("proc-macro `{macro_name}` is missing")).into())
    }

    pub(crate) fn list_macros(&self) -> impl Iterator<Item = (&str, ProcMacroKind)> {
        self.0.iter().map(|proc_macro| match *proc_macro {
            bridge::client::ProcMacro::CustomDerive { trait_name, .. } => {
                (trait_name, ProcMacroKind::CustomDerive)
            }
            bridge::client::ProcMacro::Bang { name, .. } => (name, ProcMacroKind::Bang),
            bridge::client::ProcMacro::Attr { name, .. } => (name, ProcMacroKind::Attr),
        })
    }
}
