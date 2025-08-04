//! proc-macro server backend based on [`proc_macro_api::msg::SpanId`] as the backing span.
//! This backend is rather inflexible, used by RustRover and older rust-analyzer versions.
use std::ops::{Bound, Range};

use intern::Symbol;
use proc_macro::bridge::{self, server};

use crate::server_impl::{from_token_tree, literal_from_str, token_stream::TokenStreamBuilder};

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct SpanId(pub u32);

impl std::fmt::Debug for SpanId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

type Span = SpanId;
type TokenStream = crate::server_impl::TokenStream<Span>;

pub struct FreeFunctions;

pub struct SpanIdServer {
    pub call_site: Span,
    pub def_site: Span,
    pub mixed_site: Span,
}

impl server::Types for SpanIdServer {
    type FreeFunctions = FreeFunctions;
    type TokenStream = TokenStream;
    type Span = Span;
    type Symbol = Symbol;
}

impl server::FreeFunctions for SpanIdServer {
    fn injected_env_var(&mut self, _: &str) -> Option<std::string::String> {
        None
    }
    fn track_env_var(&mut self, _var: &str, _value: Option<&str>) {}
    fn track_path(&mut self, _path: &str) {}
    fn literal_from_str(
        &mut self,
        s: &str,
    ) -> Result<bridge::Literal<Self::Span, Self::Symbol>, ()> {
        literal_from_str(s, self.call_site)
    }

    fn emit_diagnostic(&mut self, _: bridge::Diagnostic<Self::Span>) {}
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
    fn from_token_tree(
        &mut self,
        tree: bridge::TokenTree<Self::TokenStream, Self::Span, Self::Symbol>,
    ) -> Self::TokenStream {
        from_token_tree(tree)
    }

    fn expand_expr(&mut self, self_: &Self::TokenStream) -> Result<Self::TokenStream, ()> {
        Ok(self_.clone())
    }

    fn concat_trees(
        &mut self,
        base: Option<Self::TokenStream>,
        trees: Vec<bridge::TokenTree<Self::TokenStream, Self::Span, Self::Symbol>>,
    ) -> Self::TokenStream {
        let mut builder = TokenStreamBuilder::default();
        if let Some(base) = base {
            builder.push(base);
        }
        for tree in trees {
            builder.push(self.from_token_tree(tree));
        }
        builder.build()
    }

    fn concat_streams(
        &mut self,
        base: Option<Self::TokenStream>,
        streams: Vec<Self::TokenStream>,
    ) -> Self::TokenStream {
        let mut builder = TokenStreamBuilder::default();
        if let Some(base) = base {
            builder.push(base);
        }
        for stream in streams {
            builder.push(stream);
        }
        builder.build()
    }

    fn into_trees(
        &mut self,
        stream: Self::TokenStream,
    ) -> Vec<bridge::TokenTree<Self::TokenStream, Self::Span, Self::Symbol>> {
        // Can't join with `SpanId`.
        stream.into_bridge(&mut |first, _second| first)
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
    fn globals(&mut self) -> bridge::ExpnGlobals<Self::Span> {
        bridge::ExpnGlobals {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ra_server_to_string() {
        let s = TokenStream {
            token_trees: vec![
                tt::TokenTree::Leaf(tt::Leaf::Ident(tt::Ident {
                    sym: Symbol::intern("struct"),
                    span: SpanId(0),
                    is_raw: tt::IdentIsRaw::No,
                })),
                tt::TokenTree::Leaf(tt::Leaf::Ident(tt::Ident {
                    sym: Symbol::intern("T"),
                    span: SpanId(0),
                    is_raw: tt::IdentIsRaw::No,
                })),
                tt::TokenTree::Subtree(tt::Subtree {
                    delimiter: tt::Delimiter {
                        open: SpanId(0),
                        close: SpanId(0),
                        kind: tt::DelimiterKind::Brace,
                    },
                    len: 0,
                }),
            ],
        };

        assert_eq!(s.to_string(), "struct T {}");
    }

    #[test]
    fn test_ra_server_from_str() {
        let subtree_paren_a = vec![
            tt::TokenTree::Subtree(tt::Subtree {
                delimiter: tt::Delimiter {
                    open: SpanId(0),
                    close: SpanId(0),
                    kind: tt::DelimiterKind::Parenthesis,
                },
                len: 1,
            }),
            tt::TokenTree::Leaf(tt::Leaf::Ident(tt::Ident {
                is_raw: tt::IdentIsRaw::No,
                sym: Symbol::intern("a"),
                span: SpanId(0),
            })),
        ];

        let t1 = TokenStream::from_str("(a)", SpanId(0)).unwrap();
        assert_eq!(t1.token_trees.len(), 2);
        assert!(t1.token_trees[0..2] == subtree_paren_a);

        let t2 = TokenStream::from_str("(a);", SpanId(0)).unwrap();
        assert_eq!(t2.token_trees.len(), 3);
        assert!(t2.token_trees[0..2] == subtree_paren_a);

        let underscore = TokenStream::from_str("_", SpanId(0)).unwrap();
        assert!(
            underscore.token_trees[0]
                == tt::TokenTree::Leaf(tt::Leaf::Ident(tt::Ident {
                    sym: Symbol::intern("_"),
                    span: SpanId(0),
                    is_raw: tt::IdentIsRaw::No,
                }))
        );
    }
}
