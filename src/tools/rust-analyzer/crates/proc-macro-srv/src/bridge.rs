use proc_macro::bridge as pm_bridge;

pub(crate) use pm_bridge::{DelimSpan, Diagnostic, ExpnGlobals, LitKind};

pub(crate) type TokenTree<S> =
    pm_bridge::TokenTree<crate::token_stream::TokenStream<S>, S, intern::Symbol>;
pub(crate) type Literal<S> = pm_bridge::Literal<S, intern::Symbol>;
pub(crate) type Group<S> = pm_bridge::Group<crate::token_stream::TokenStream<S>, S>;
pub(crate) type Punct<S> = pm_bridge::Punct<S>;
pub(crate) type Ident<S> = pm_bridge::Ident<S, intern::Symbol>;
