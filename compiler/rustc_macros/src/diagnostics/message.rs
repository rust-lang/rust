use proc_macro2::TokenStream;
use quote::quote;
use syn::Path;

pub(crate) enum Message {
    Slug(Path),
    Inline(String),
}

impl Message {
    pub(crate) fn slug(&self) -> Option<&Path> {
        match self {
            Message::Slug(slug) => Some(slug),
            Message::Inline(_) => None,
        }
    }

    pub(crate) fn diag_message(&self) -> TokenStream {
        match self {
            Message::Slug(slug) => {
                quote! { crate::fluent_generated::#slug }
            }
            Message::Inline(message) => {
                quote! { rustc_errors::DiagMessage::Inline(std::borrow::Cow::Borrowed(#message)) }
            }
        }
    }
}
