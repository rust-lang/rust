//! `proc_macro::bridge` newtypes.

use rustc_proc_macro::bridge as pm_bridge;

pub use pm_bridge::{DelimSpan, Diagnostic, ExpnGlobals, LitKind};

pub type TokenTree<S> =
    pm_bridge::TokenTree<crate::token_stream::TokenStream<S>, S, intern::Symbol>;
pub type Literal<S> = pm_bridge::Literal<S, intern::Symbol>;
pub type Group<S> = pm_bridge::Group<crate::token_stream::TokenStream<S>, S>;
pub type Punct<S> = pm_bridge::Punct<S>;
pub type Ident<S> = pm_bridge::Ident<S, intern::Symbol>;
