//! proc-macro server backend based on [`proc_macro_api::msg::SpanId`] as the backing span.
//! This backend is rather inflexible, used by RustRover and older rust-analyzer versions.
use std::{
    collections::{HashMap, HashSet},
    ops::{Bound, Range},
};

use intern::Symbol;
use proc_macro::bridge::server;

use crate::{
    bridge::{Diagnostic, ExpnGlobals, Literal, TokenTree},
    server_impl::literal_from_str,
};

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct SpanId(pub u32);

impl std::fmt::Debug for SpanId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

type Span = SpanId;

pub struct FreeFunctions;

pub struct SpanIdServer {
    // FIXME: Report this back to the caller to track as dependencies
    pub tracked_env_vars: HashMap<Box<str>, Option<Box<str>>>,
    // FIXME: Report this back to the caller to track as dependencies
    pub tracked_paths: HashSet<Box<str>>,
    pub call_site: Span,
    pub def_site: Span,
    pub mixed_site: Span,
}

impl server::Types for SpanIdServer {
    type FreeFunctions = FreeFunctions;
    type TokenStream = crate::token_stream::TokenStream<Span>;
    type Span = Span;
    type Symbol = Symbol;
}

impl server::FreeFunctions for SpanIdServer {
    fn injected_env_var(&mut self, _: &str) -> Option<std::string::String> {
        None
    }
    fn track_env_var(&mut self, var: &str, value: Option<&str>) {
        self.tracked_env_vars.insert(var.into(), value.map(Into::into));
    }
    fn track_path(&mut self, path: &str) {
        self.tracked_paths.insert(path.into());
    }

    fn literal_from_str(&mut self, s: &str) -> Result<Literal<Self::Span>, ()> {
        literal_from_str(s, self.call_site)
    }

    fn emit_diagnostic(&mut self, _: Diagnostic<Self::Span>) {}
}

impl server::TokenStream for SpanIdServer {
    fn is_empty(&mut self, stream: &Self::TokenStream) -> bool {
        stream.is_empty()
    }
    fn from_str(&mut self, src: &str) -> Self::TokenStream {
        Self::TokenStream::from_str(src, self.call_site).unwrap_or_else(|e| {
            Self::TokenStream::from_str(
                &format!("compile_error!(\"failed to parse str to token stream: {e}\")"),
                self.call_site,
            )
            .unwrap()
        })
    }
    fn to_string(&mut self, stream: &Self::TokenStream) -> String {
        stream.to_string()
    }
    fn from_token_tree(&mut self, tree: TokenTree<Self::Span>) -> Self::TokenStream {
        Self::TokenStream::new(vec![tree])
    }

    fn expand_expr(&mut self, self_: &Self::TokenStream) -> Result<Self::TokenStream, ()> {
        Ok(self_.clone())
    }

    fn concat_trees(
        &mut self,
        base: Option<Self::TokenStream>,
        trees: Vec<TokenTree<Self::Span>>,
    ) -> Self::TokenStream {
        match base {
            Some(mut base) => {
                for tt in trees {
                    base.push_tree(tt);
                }
                base
            }
            None => Self::TokenStream::new(trees),
        }
    }

    fn concat_streams(
        &mut self,
        base: Option<Self::TokenStream>,
        streams: Vec<Self::TokenStream>,
    ) -> Self::TokenStream {
        let mut stream = base.unwrap_or_default();
        for s in streams {
            stream.push_stream(s);
        }
        stream
    }

    fn into_trees(&mut self, stream: Self::TokenStream) -> Vec<TokenTree<Self::Span>> {
        (*stream.0).clone()
    }
}

impl server::Span for SpanIdServer {
    fn debug(&mut self, span: Self::Span) -> String {
        format!("{:?}", span.0)
    }
    fn file(&mut self, _span: Self::Span) -> String {
        String::new()
    }
    fn local_file(&mut self, _span: Self::Span) -> Option<String> {
        None
    }
    fn save_span(&mut self, _span: Self::Span) -> usize {
        0
    }
    fn recover_proc_macro_span(&mut self, _id: usize) -> Self::Span {
        self.call_site
    }
    /// Recent feature, not yet in the proc_macro
    ///
    /// See PR:
    /// https://github.com/rust-lang/rust/pull/55780
    fn source_text(&mut self, _span: Self::Span) -> Option<String> {
        None
    }

    fn parent(&mut self, _span: Self::Span) -> Option<Self::Span> {
        None
    }
    fn source(&mut self, span: Self::Span) -> Self::Span {
        span
    }
    fn byte_range(&mut self, _span: Self::Span) -> Range<usize> {
        Range { start: 0, end: 0 }
    }
    fn join(&mut self, first: Self::Span, _second: Self::Span) -> Option<Self::Span> {
        // Just return the first span again, because some macros will unwrap the result.
        Some(first)
    }
    fn subspan(
        &mut self,
        span: Self::Span,
        _start: Bound<usize>,
        _end: Bound<usize>,
    ) -> Option<Self::Span> {
        // Just return the span again, because some macros will unwrap the result.
        Some(span)
    }
    fn resolved_at(&mut self, _span: Self::Span, _at: Self::Span) -> Self::Span {
        self.call_site
    }

    fn end(&mut self, _self_: Self::Span) -> Self::Span {
        self.call_site
    }

    fn start(&mut self, _self_: Self::Span) -> Self::Span {
        self.call_site
    }

    fn line(&mut self, _span: Self::Span) -> usize {
        1
    }

    fn column(&mut self, _span: Self::Span) -> usize {
        1
    }
}

impl server::Symbol for SpanIdServer {
    fn normalize_and_validate_ident(&mut self, string: &str) -> Result<Self::Symbol, ()> {
        // FIXME: nfc-normalize and validate idents
        Ok(<Self as server::Server>::intern_symbol(string))
    }
}

impl server::Server for SpanIdServer {
    fn globals(&mut self) -> ExpnGlobals<Self::Span> {
        ExpnGlobals {
            def_site: self.def_site,
            call_site: self.call_site,
            mixed_site: self.mixed_site,
        }
    }

    fn intern_symbol(ident: &str) -> Self::Symbol {
        Symbol::intern(ident)
    }

    fn with_symbol_string(symbol: &Self::Symbol, f: impl FnOnce(&str)) {
        f(symbol.as_str())
    }
}
