#![allow(unused_variables)]
use std::cell::RefCell;
use std::ops::{Bound, Range};

use crate::bridge::client::Symbol;
use crate::bridge::{Diagnostic, ExpnGlobals, Literal, TokenTree, server};

pub struct NoRustc;

impl server::Span for NoRustc {
    fn debug(&mut self, span: Self::Span) -> String {
        format!("{} bytes({}..{})", span.hi - span.lo, span.lo, span.hi)
    }

    fn parent(&mut self, span: Self::Span) -> Option<Self::Span> {
        todo!()
    }

    fn source(&mut self, span: Self::Span) -> Self::Span {
        todo!()
    }

    fn byte_range(&mut self, span: Self::Span) -> Range<usize> {
        todo!()
    }

    fn start(&mut self, span: Self::Span) -> Self::Span {
        Span { lo: span.lo, hi: span.lo }
    }

    fn end(&mut self, span: Self::Span) -> Self::Span {
        Span { lo: span.hi, hi: span.hi }
    }

    fn line(&mut self, span: Self::Span) -> usize {
        todo!()
    }

    fn column(&mut self, span: Self::Span) -> usize {
        todo!()
    }

    fn file(&mut self, span: Self::Span) -> String {
        todo!()
    }

    fn local_file(&mut self, span: Self::Span) -> Option<String> {
        todo!()
    }

    fn join(&mut self, span: Self::Span, other: Self::Span) -> Option<Self::Span> {
        todo!()
    }

    fn subspan(
        &mut self,
        span: Self::Span,
        start: Bound<usize>,
        end: Bound<usize>,
    ) -> Option<Self::Span> {
        let length = span.hi as usize - span.lo as usize;

        let start = match start {
            Bound::Included(lo) => lo,
            Bound::Excluded(lo) => lo.checked_add(1)?,
            Bound::Unbounded => 0,
        };

        let end = match end {
            Bound::Included(hi) => hi.checked_add(1)?,
            Bound::Excluded(hi) => hi,
            Bound::Unbounded => length,
        };

        // Bounds check the values, preventing addition overflow and OOB spans.
        if start > u32::MAX as usize
            || end > u32::MAX as usize
            || (u32::MAX - start as u32) < span.lo
            || (u32::MAX - end as u32) < span.lo
            || start >= end
            || end > length
        {
            return None;
        }

        let new_lo = span.lo + start as u32;
        let new_hi = span.lo + end as u32;
        Some(Span { lo: new_lo, hi: new_hi })
    }

    fn resolved_at(&mut self, span: Self::Span, at: Self::Span) -> Self::Span {
        todo!()
    }

    fn source_text(&mut self, span: Self::Span) -> Option<String> {
        todo!()
    }

    fn save_span(&mut self, span: Self::Span) -> usize {
        SAVED_SPANS.with_borrow_mut(|spans| {
            let idx = spans.len();
            spans.push(span);
            idx
        })
    }

    fn recover_proc_macro_span(&mut self, id: usize) -> Self::Span {
        SAVED_SPANS.with_borrow(|spans| spans[id])
    }
}

thread_local! {
    static SAVED_SPANS: RefCell<Vec<Span>> = const { RefCell::new(Vec::new()) };
}

impl server::FreeFunctions for NoRustc {
    fn injected_env_var(&mut self, var: &str) -> Option<String> {
        todo!()
    }

    fn track_env_var(&mut self, _var: &str, _value: Option<&str>) {}

    fn track_path(&mut self, _path: &str) {}

    fn literal_from_str(&mut self, s: &str) -> Result<Literal<Self::Span, Self::Symbol>, ()> {
        todo!()
    }

    fn emit_diagnostic(&mut self, diagnostic: Diagnostic<Self::Span>) {
        panic!("cannot emit diagnostic in standalone mode");
    }
}

impl server::TokenStream for NoRustc {
    fn is_empty(&mut self, tokens: &Self::TokenStream) -> bool {
        tokens.0.is_empty()
    }

    fn expand_expr(&mut self, tokens: &Self::TokenStream) -> Result<Self::TokenStream, ()> {
        todo!()
    }

    fn from_str(&mut self, src: &str) -> Self::TokenStream {
        todo!()
    }

    fn to_string(&mut self, tokens: &Self::TokenStream) -> String {
        todo!()
    }

    fn from_token_tree(
        &mut self,
        tree: TokenTree<Self::TokenStream, Self::Span, Self::Symbol>,
    ) -> Self::TokenStream {
        TokenStream(vec![tree])
    }

    fn concat_trees(
        &mut self,
        base: Option<Self::TokenStream>,
        trees: Vec<TokenTree<Self::TokenStream, Self::Span, Self::Symbol>>,
    ) -> Self::TokenStream {
        let mut base = base.unwrap_or_else(TokenStream::new);
        base.0.extend(trees);
        base
    }

    fn concat_streams(
        &mut self,
        base: Option<Self::TokenStream>,
        streams: Vec<Self::TokenStream>,
    ) -> Self::TokenStream {
        let mut base = base.unwrap_or_else(TokenStream::new);
        for stream in streams {
            base = self.concat_trees(Some(base), stream.0);
        }
        base
    }

    fn into_trees(
        &mut self,
        tokens: Self::TokenStream,
    ) -> Vec<TokenTree<Self::TokenStream, Self::Span, Self::Symbol>> {
        tokens.0
    }
}

pub struct FreeFunctions;
#[derive(Clone, Default)]
pub struct TokenStream(Vec<TokenTree<TokenStream, Span, Symbol>>);
impl TokenStream {
    pub fn new() -> Self {
        Self(Vec::new())
    }
}

#[derive(Hash, PartialEq, Eq, Clone, Copy)]
pub struct Span {
    pub lo: u32,
    pub hi: u32,
}
impl Span {
    pub const DUMMY: Self = Self { lo: 0, hi: 0 };
}

impl server::Types for NoRustc {
    type FreeFunctions = FreeFunctions;
    type TokenStream = TokenStream;
    type Span = Span;
    type Symbol = Symbol;
}

impl server::Server for NoRustc {
    fn globals(&mut self) -> ExpnGlobals<Self::Span> {
        ExpnGlobals { def_site: Span::DUMMY, call_site: Span::DUMMY, mixed_site: Span::DUMMY }
    }

    fn intern_symbol(ident: &str) -> Self::Symbol {
        Symbol::new(ident)
    }

    fn with_symbol_string(symbol: &Self::Symbol, f: impl FnOnce(&str)) {
        symbol.with(f);
    }
}

impl server::Symbol for NoRustc {
    fn normalize_and_validate_ident(&mut self, string: &str) -> Result<Self::Symbol, ()> {
        todo!()
    }
}
