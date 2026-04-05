//! proc-macro server backend based on [`proc_macro_api::msg::SpanId`] as the backing span.
//! This backend is rather inflexible, used by RustRover and older rust-analyzer versions.
use std::{
    collections::{HashMap, HashSet},
    ops::{Bound, Range},
};

use intern::Symbol;
use rustc_proc_macro::bridge::server;

use crate::{
    ProcMacroClientHandle,
    bridge::{Diagnostic, ExpnGlobals, Literal},
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

pub struct SpanIdServer<'a> {
    // FIXME: Report this back to the caller to track as dependencies
    pub tracked_env_vars: HashMap<Box<str>, Option<Box<str>>>,
    // FIXME: Report this back to the caller to track as dependencies
    pub tracked_paths: HashSet<Box<str>>,
    pub call_site: Span,
    pub def_site: Span,
    pub mixed_site: Span,
    pub callback: Option<ProcMacroClientHandle<'a>>,
}

type TokenStream = crate::token_stream::TokenStream<Span>;
type BridgeTokenStream = crate::bridge::TokenStream<Span>;

impl server::Server for SpanIdServer<'_> {
    type Span = Span;
    type Symbol = Symbol;

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

    fn injected_env_var(&mut self, _: &str) -> Option<std::string::String> {
        None
    }
    fn track_env_var(&mut self, var: &str, value: Option<&str>) {
        self.tracked_env_vars.insert(var.into(), value.map(Into::into));
    }
    fn track_path(&mut self, path: &str) {
        self.tracked_paths.insert(path.into());
    }

    fn literal_from_str(&mut self, s: &str) -> Result<Literal<Self::Span>, String> {
        literal_from_str(s, self.call_site)
            .map_err(|()| "cannot parse string into literal".to_string())
    }

    fn emit_diagnostic(&mut self, _: Diagnostic<Self::Span>) {}

    fn ts_from_str(&mut self, src: &str) -> Result<BridgeTokenStream, String> {
        TokenStream::from_str(src, self.call_site)
            .map_err(|e| format!("failed to parse str to token stream: {e}"))
            .map(TokenStream::into_bridge)
    }
    fn ts_to_string(&mut self, stream: BridgeTokenStream) -> String {
        TokenStream::from_bridge(stream).to_string()
    }

    fn ts_expand_expr(&mut self, self_: BridgeTokenStream) -> Result<BridgeTokenStream, ()> {
        Ok(self_)
    }

    fn span_debug(&mut self, span: Self::Span) -> String {
        format!("{:?}", span.0)
    }
    fn span_file(&mut self, _span: Self::Span) -> String {
        String::new()
    }
    fn span_local_file(&mut self, _span: Self::Span) -> Option<String> {
        None
    }
    fn span_save_span(&mut self, _span: Self::Span) -> usize {
        0
    }
    fn span_recover_proc_macro_span(&mut self, _id: usize) -> Self::Span {
        self.call_site
    }
    /// Recent feature, not yet in the proc_macro
    ///
    /// See PR:
    /// https://github.com/rust-lang/rust/pull/55780
    fn span_source_text(&mut self, _span: Self::Span) -> Option<String> {
        None
    }

    fn span_parent(&mut self, _span: Self::Span) -> Option<Self::Span> {
        None
    }
    fn span_source(&mut self, span: Self::Span) -> Self::Span {
        span
    }
    fn span_byte_range(&mut self, _span: Self::Span) -> Range<usize> {
        Range { start: 0, end: 0 }
    }
    fn span_join(&mut self, first: Self::Span, _second: Self::Span) -> Option<Self::Span> {
        // Just return the first span again, because some macros will unwrap the result.
        Some(first)
    }
    fn span_subspan(
        &mut self,
        span: Self::Span,
        _start: Bound<usize>,
        _end: Bound<usize>,
    ) -> Option<Self::Span> {
        // Just return the span again, because some macros will unwrap the result.
        Some(span)
    }
    fn span_resolved_at(&mut self, _span: Self::Span, _at: Self::Span) -> Self::Span {
        self.call_site
    }

    fn span_end(&mut self, _self_: Self::Span) -> Self::Span {
        self.call_site
    }

    fn span_start(&mut self, _self_: Self::Span) -> Self::Span {
        self.call_site
    }

    fn span_line(&mut self, _span: Self::Span) -> usize {
        1
    }

    fn span_column(&mut self, _span: Self::Span) -> usize {
        1
    }

    fn symbol_normalize_and_validate_ident(&mut self, string: &str) -> Result<Self::Symbol, ()> {
        // FIXME: nfc-normalize and validate idents
        Ok(<Self as server::Server>::intern_symbol(string))
    }
}
