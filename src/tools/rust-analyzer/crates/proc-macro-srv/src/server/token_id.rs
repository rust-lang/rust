//! proc-macro server backend based on [`proc_macro_api::msg::TokenId`] as the backing span.
//! This backend is rather inflexible, used by RustRover and older rust-analyzer versions.
use std::{
    iter,
    ops::{Bound, Range},
};

use proc_macro::bridge::{self, server};

use crate::server::{
    delim_to_external, delim_to_internal, token_stream::TokenStreamBuilder, LiteralFormatter,
    Symbol, SymbolInternerRef, SYMBOL_INTERNER,
};
mod tt {
    pub use proc_macro_api::msg::TokenId;

    pub use tt::*;

    pub type Subtree = ::tt::Subtree<TokenId>;
    pub type TokenTree = ::tt::TokenTree<TokenId>;
    pub type Leaf = ::tt::Leaf<TokenId>;
    pub type Literal = ::tt::Literal<TokenId>;
    pub type Punct = ::tt::Punct<TokenId>;
    pub type Ident = ::tt::Ident<TokenId>;
}
type Group = tt::Subtree;
type TokenTree = tt::TokenTree;
#[allow(unused)]
type Punct = tt::Punct;
type Spacing = tt::Spacing;
#[allow(unused)]
type Literal = tt::Literal;
type Span = tt::TokenId;
type TokenStream = crate::server::TokenStream<Span>;

#[derive(Clone)]
pub struct SourceFile;
pub struct FreeFunctions;

pub struct TokenIdServer {
    pub(crate) interner: SymbolInternerRef,
    pub call_site: Span,
    pub def_site: Span,
    pub mixed_site: Span,
}

impl server::Types for TokenIdServer {
    type FreeFunctions = FreeFunctions;
    type TokenStream = TokenStream;
    type SourceFile = SourceFile;
    type Span = Span;
    type Symbol = Symbol;
}

impl server::FreeFunctions for TokenIdServer {
    fn injected_env_var(&mut self, _: &str) -> Option<std::string::String> {
        None
    }
    fn track_env_var(&mut self, _var: &str, _value: Option<&str>) {}
    fn track_path(&mut self, _path: &str) {}
    fn literal_from_str(
        &mut self,
        s: &str,
    ) -> Result<bridge::Literal<Self::Span, Self::Symbol>, ()> {
        use proc_macro::bridge::LitKind;
        use rustc_lexer::{LiteralKind, Token, TokenKind};

        let mut tokens = rustc_lexer::tokenize(s);
        let minus_or_lit = tokens.next().unwrap_or(Token { kind: TokenKind::Eof, len: 0 });

        let lit = if minus_or_lit.kind == TokenKind::Minus {
            let lit = tokens.next().ok_or(())?;
            if !matches!(
                lit.kind,
                TokenKind::Literal {
                    kind: LiteralKind::Int { .. } | LiteralKind::Float { .. },
                    ..
                }
            ) {
                return Err(());
            }
            lit
        } else {
            minus_or_lit
        };

        if tokens.next().is_some() {
            return Err(());
        }

        let TokenKind::Literal { kind, suffix_start } = lit.kind else { return Err(()) };

        let (kind, start_offset, end_offset) = match kind {
            LiteralKind::Int { .. } => (LitKind::Integer, 0, 0),
            LiteralKind::Float { .. } => (LitKind::Float, 0, 0),
            LiteralKind::Char { terminated } => (LitKind::Char, 1, terminated as usize),
            LiteralKind::Byte { terminated } => (LitKind::Byte, 2, terminated as usize),
            LiteralKind::Str { terminated } => (LitKind::Str, 1, terminated as usize),
            LiteralKind::ByteStr { terminated } => (LitKind::ByteStr, 2, terminated as usize),
            LiteralKind::CStr { terminated } => (LitKind::CStr, 2, terminated as usize),
            LiteralKind::RawStr { n_hashes } => (
                LitKind::StrRaw(n_hashes.unwrap_or_default()),
                2 + n_hashes.unwrap_or_default() as usize,
                1 + n_hashes.unwrap_or_default() as usize,
            ),
            LiteralKind::RawByteStr { n_hashes } => (
                LitKind::ByteStrRaw(n_hashes.unwrap_or_default()),
                3 + n_hashes.unwrap_or_default() as usize,
                1 + n_hashes.unwrap_or_default() as usize,
            ),
            LiteralKind::RawCStr { n_hashes } => (
                LitKind::CStrRaw(n_hashes.unwrap_or_default()),
                3 + n_hashes.unwrap_or_default() as usize,
                1 + n_hashes.unwrap_or_default() as usize,
            ),
        };

        let (lit, suffix) = s.split_at(suffix_start as usize);
        let lit = &lit[start_offset..lit.len() - end_offset];
        let suffix = match suffix {
            "" | "_" => None,
            suffix => Some(Symbol::intern(self.interner, suffix)),
        };

        Ok(bridge::Literal {
            kind,
            symbol: Symbol::intern(self.interner, lit),
            suffix,
            span: self.call_site,
        })
    }

    fn emit_diagnostic(&mut self, _: bridge::Diagnostic<Self::Span>) {}
}

impl server::TokenStream for TokenIdServer {
    fn is_empty(&mut self, stream: &Self::TokenStream) -> bool {
        stream.is_empty()
    }
    fn from_str(&mut self, src: &str) -> Self::TokenStream {
        Self::TokenStream::from_str(src, self.call_site).expect("cannot parse string")
    }
    fn to_string(&mut self, stream: &Self::TokenStream) -> String {
        stream.to_string()
    }
    fn from_token_tree(
        &mut self,
        tree: bridge::TokenTree<Self::TokenStream, Self::Span, Self::Symbol>,
    ) -> Self::TokenStream {
        match tree {
            bridge::TokenTree::Group(group) => {
                let group = Group {
                    delimiter: delim_to_internal(group.delimiter, group.span),
                    token_trees: match group.stream {
                        Some(stream) => stream.into_iter().collect(),
                        None => Box::new([]),
                    },
                };
                let tree = TokenTree::from(group);
                Self::TokenStream::from_iter(iter::once(tree))
            }

            bridge::TokenTree::Ident(ident) => {
                let text = ident.sym.text(self.interner);
                let text =
                    if ident.is_raw { ::tt::SmolStr::from_iter(["r#", &text]) } else { text };
                let ident: tt::Ident = tt::Ident { text, span: ident.span };
                let leaf = tt::Leaf::from(ident);
                let tree = TokenTree::from(leaf);
                Self::TokenStream::from_iter(iter::once(tree))
            }

            bridge::TokenTree::Literal(literal) => {
                let literal = LiteralFormatter(literal);
                let text = literal.with_stringify_parts(self.interner, |parts| {
                    ::tt::SmolStr::from_iter(parts.iter().copied())
                });

                let literal = tt::Literal { text, span: literal.0.span };
                let leaf = tt::Leaf::from(literal);
                let tree = TokenTree::from(leaf);
                Self::TokenStream::from_iter(iter::once(tree))
            }

            bridge::TokenTree::Punct(p) => {
                let punct = tt::Punct {
                    char: p.ch as char,
                    spacing: if p.joint { Spacing::Joint } else { Spacing::Alone },
                    span: p.span,
                };
                let leaf = tt::Leaf::from(punct);
                let tree = TokenTree::from(leaf);
                Self::TokenStream::from_iter(iter::once(tree))
            }
        }
    }

    fn expand_expr(&mut self, self_: &Self::TokenStream) -> Result<Self::TokenStream, ()> {
        Ok(self_.clone())
    }

    fn concat_trees(
        &mut self,
        base: Option<Self::TokenStream>,
        trees: Vec<bridge::TokenTree<Self::TokenStream, Self::Span, Self::Symbol>>,
    ) -> Self::TokenStream {
        let mut builder = TokenStreamBuilder::new();
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
        let mut builder = TokenStreamBuilder::new();
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
        stream
            .into_iter()
            .map(|tree| match tree {
                tt::TokenTree::Leaf(tt::Leaf::Ident(ident)) => {
                    bridge::TokenTree::Ident(bridge::Ident {
                        sym: Symbol::intern(self.interner, ident.text.trim_start_matches("r#")),
                        is_raw: ident.text.starts_with("r#"),
                        span: ident.span,
                    })
                }
                tt::TokenTree::Leaf(tt::Leaf::Literal(lit)) => {
                    bridge::TokenTree::Literal(bridge::Literal {
                        span: lit.span,
                        ..server::FreeFunctions::literal_from_str(self, &lit.text)
                            .unwrap_or_else(|_| panic!("`{}`", lit.text))
                    })
                }
                tt::TokenTree::Leaf(tt::Leaf::Punct(punct)) => {
                    bridge::TokenTree::Punct(bridge::Punct {
                        ch: punct.char as u8,
                        joint: punct.spacing == Spacing::Joint,
                        span: punct.span,
                    })
                }
                tt::TokenTree::Subtree(subtree) => bridge::TokenTree::Group(bridge::Group {
                    delimiter: delim_to_external(subtree.delimiter),
                    stream: if subtree.token_trees.is_empty() {
                        None
                    } else {
                        Some(TokenStream { token_trees: subtree.token_trees.into_vec() })
                    },
                    span: bridge::DelimSpan::from_single(subtree.delimiter.open),
                }),
            })
            .collect()
    }
}

impl server::SourceFile for TokenIdServer {
    fn eq(&mut self, _file1: &Self::SourceFile, _file2: &Self::SourceFile) -> bool {
        true
    }
    fn path(&mut self, _file: &Self::SourceFile) -> String {
        String::new()
    }
    fn is_real(&mut self, _file: &Self::SourceFile) -> bool {
        true
    }
}

impl server::Span for TokenIdServer {
    fn debug(&mut self, span: Self::Span) -> String {
        format!("{:?}", span.0)
    }
    fn source_file(&mut self, _span: Self::Span) -> Self::SourceFile {
        SourceFile {}
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
        0
    }

    fn column(&mut self, _span: Self::Span) -> usize {
        0
    }
}

impl server::Symbol for TokenIdServer {
    fn normalize_and_validate_ident(&mut self, string: &str) -> Result<Self::Symbol, ()> {
        // FIXME: nfc-normalize and validate idents
        Ok(<Self as server::Server>::intern_symbol(string))
    }
}

impl server::Server for TokenIdServer {
    fn globals(&mut self) -> bridge::ExpnGlobals<Self::Span> {
        bridge::ExpnGlobals {
            def_site: self.def_site,
            call_site: self.call_site,
            mixed_site: self.mixed_site,
        }
    }

    fn intern_symbol(ident: &str) -> Self::Symbol {
        Symbol::intern(&SYMBOL_INTERNER, &::tt::SmolStr::from(ident))
    }

    fn with_symbol_string(symbol: &Self::Symbol, f: impl FnOnce(&str)) {
        f(symbol.text(&SYMBOL_INTERNER).as_str())
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
                    text: "struct".into(),
                    span: tt::TokenId(0),
                })),
                tt::TokenTree::Leaf(tt::Leaf::Ident(tt::Ident {
                    text: "T".into(),
                    span: tt::TokenId(0),
                })),
                tt::TokenTree::Subtree(tt::Subtree {
                    delimiter: tt::Delimiter {
                        open: tt::TokenId(0),
                        close: tt::TokenId(0),
                        kind: tt::DelimiterKind::Brace,
                    },
                    token_trees: Box::new([]),
                }),
            ],
        };

        assert_eq!(s.to_string(), "struct T {}");
    }

    #[test]
    fn test_ra_server_from_str() {
        let subtree_paren_a = tt::TokenTree::Subtree(tt::Subtree {
            delimiter: tt::Delimiter {
                open: tt::TokenId(0),
                close: tt::TokenId(0),
                kind: tt::DelimiterKind::Parenthesis,
            },
            token_trees: Box::new([tt::TokenTree::Leaf(tt::Leaf::Ident(tt::Ident {
                text: "a".into(),
                span: tt::TokenId(0),
            }))]),
        });

        let t1 = TokenStream::from_str("(a)", tt::TokenId(0)).unwrap();
        assert_eq!(t1.token_trees.len(), 1);
        assert_eq!(t1.token_trees[0], subtree_paren_a);

        let t2 = TokenStream::from_str("(a);", tt::TokenId(0)).unwrap();
        assert_eq!(t2.token_trees.len(), 2);
        assert_eq!(t2.token_trees[0], subtree_paren_a);

        let underscore = TokenStream::from_str("_", tt::TokenId(0)).unwrap();
        assert_eq!(
            underscore.token_trees[0],
            tt::TokenTree::Leaf(tt::Leaf::Ident(tt::Ident {
                text: "_".into(),
                span: tt::TokenId(0),
            }))
        );
    }
}
