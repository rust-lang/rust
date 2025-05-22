//! proc-macro server backend based on rust-analyzer's internal span representation
//! This backend is used solely by rust-analyzer as it ties into rust-analyzer internals.
//!
//! It is an unfortunate result of how the proc-macro API works that we need to look into the
//! concrete representation of the spans, and as such, RustRover cannot make use of this unless they
//! change their representation to be compatible with rust-analyzer's.
use std::{
    collections::{HashMap, HashSet},
    ops::{Bound, Range},
};

use intern::Symbol;
use proc_macro::bridge::{self, server};
use span::{FIXUP_ERASED_FILE_AST_ID_MARKER, Span};
use tt::{TextRange, TextSize};

use crate::server_impl::{TopSubtree, literal_kind_to_internal, token_stream::TokenStreamBuilder};
mod tt {
    pub use tt::*;

    pub type TokenTree = ::tt::TokenTree<super::Span>;
    pub type Leaf = ::tt::Leaf<super::Span>;
    pub type Literal = ::tt::Literal<super::Span>;
    pub type Punct = ::tt::Punct<super::Span>;
    pub type Ident = ::tt::Ident<super::Span>;
}

type TokenStream = crate::server_impl::TokenStream<Span>;

pub struct FreeFunctions;

pub struct RaSpanServer {
    // FIXME: Report this back to the caller to track as dependencies
    pub tracked_env_vars: HashMap<Box<str>, Option<Box<str>>>,
    // FIXME: Report this back to the caller to track as dependencies
    pub tracked_paths: HashSet<Box<str>>,
    pub call_site: Span,
    pub def_site: Span,
    pub mixed_site: Span,
}

impl server::Types for RaSpanServer {
    type FreeFunctions = FreeFunctions;
    type TokenStream = TokenStream;
    type Span = Span;
    type Symbol = Symbol;
}

impl server::FreeFunctions for RaSpanServer {
    fn injected_env_var(&mut self, _: &str) -> Option<std::string::String> {
        None
    }

    fn track_env_var(&mut self, var: &str, value: Option<&str>) {
        self.tracked_env_vars.insert(var.into(), value.map(Into::into));
    }
    fn track_path(&mut self, path: &str) {
        self.tracked_paths.insert(path.into());
    }

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
            suffix => Some(Symbol::intern(suffix)),
        };

        Ok(bridge::Literal { kind, symbol: Symbol::intern(lit), suffix, span: self.call_site })
    }

    fn emit_diagnostic(&mut self, _: bridge::Diagnostic<Self::Span>) {
        // FIXME handle diagnostic
    }
}

impl server::TokenStream for RaSpanServer {
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
        match tree {
            bridge::TokenTree::Group(group) => {
                let group = TopSubtree::from_bridge(group);
                TokenStream { token_trees: group.0 }
            }

            bridge::TokenTree::Ident(ident) => {
                let text = ident.sym;
                let ident: tt::Ident = tt::Ident {
                    sym: text,
                    span: ident.span,
                    is_raw: if ident.is_raw { tt::IdentIsRaw::Yes } else { tt::IdentIsRaw::No },
                };
                let leaf = tt::Leaf::from(ident);
                let tree = tt::TokenTree::from(leaf);
                TokenStream { token_trees: vec![tree] }
            }

            bridge::TokenTree::Literal(literal) => {
                let token_trees =
                    if let Some((_minus, symbol)) = literal.symbol.as_str().split_once('-') {
                        let punct = tt::Punct {
                            spacing: tt::Spacing::Alone,
                            span: literal.span,
                            char: '-' as char,
                        };
                        let leaf: tt::Leaf = tt::Leaf::from(punct);
                        let minus_tree = tt::TokenTree::from(leaf);

                        let literal = tt::Literal {
                            symbol: Symbol::intern(symbol),
                            suffix: literal.suffix,
                            span: literal.span,
                            kind: literal_kind_to_internal(literal.kind),
                        };
                        let leaf: tt::Leaf = tt::Leaf::from(literal);
                        let tree = tt::TokenTree::from(leaf);
                        vec![minus_tree, tree]
                    } else {
                        let literal = tt::Literal {
                            symbol: literal.symbol,
                            suffix: literal.suffix,
                            span: literal.span,
                            kind: literal_kind_to_internal(literal.kind),
                        };

                        let leaf: tt::Leaf = tt::Leaf::from(literal);
                        let tree = tt::TokenTree::from(leaf);
                        vec![tree]
                    };
                TokenStream { token_trees }
            }

            bridge::TokenTree::Punct(p) => {
                let punct = tt::Punct {
                    char: p.ch as char,
                    spacing: if p.joint { tt::Spacing::Joint } else { tt::Spacing::Alone },
                    span: p.span,
                };
                let leaf = tt::Leaf::from(punct);
                let tree = tt::TokenTree::from(leaf);
                TokenStream { token_trees: vec![tree] }
            }
        }
    }

    fn expand_expr(&mut self, self_: &Self::TokenStream) -> Result<Self::TokenStream, ()> {
        // FIXME: requires db, more importantly this requires name resolution so we would need to
        // eagerly expand this proc-macro, but we can't know that this proc-macro is eager until we
        // expand it ...
        // This calls for some kind of marker that a proc-macro wants to access this eager API,
        // otherwise we need to treat every proc-macro eagerly / or not support this.
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
        stream.into_bridge()
    }
}

impl server::Span for RaSpanServer {
    fn debug(&mut self, span: Self::Span) -> String {
        format!("{:?}", span)
    }
    fn file(&mut self, _: Self::Span) -> String {
        // FIXME
        String::new()
    }
    fn local_file(&mut self, _: Self::Span) -> Option<String> {
        // FIXME
        None
    }
    fn save_span(&mut self, _span: Self::Span) -> usize {
        // FIXME, quote is incompatible with third-party tools
        // This is called by the quote proc-macro which is expanded when the proc-macro is compiled
        // As such, r-a will never observe this
        0
    }
    fn recover_proc_macro_span(&mut self, _id: usize) -> Self::Span {
        // FIXME, quote is incompatible with third-party tools
        // This is called by the expansion of quote!, r-a will observe this, but we don't have
        // access to the spans that were encoded
        self.call_site
    }
    /// Recent feature, not yet in the proc_macro
    ///
    /// See PR:
    /// https://github.com/rust-lang/rust/pull/55780
    fn source_text(&mut self, _span: Self::Span) -> Option<String> {
        // FIXME requires db, needs special handling wrt fixup spans
        None
    }

    fn parent(&mut self, _span: Self::Span) -> Option<Self::Span> {
        // FIXME requires db, looks up the parent call site
        None
    }
    fn source(&mut self, span: Self::Span) -> Self::Span {
        // FIXME requires db, returns the top level call site
        span
    }
    fn byte_range(&mut self, span: Self::Span) -> Range<usize> {
        // FIXME requires db to resolve the ast id, THIS IS NOT INCREMENTAL
        Range { start: span.range.start().into(), end: span.range.end().into() }
    }
    fn join(&mut self, first: Self::Span, second: Self::Span) -> Option<Self::Span> {
        // We can't modify the span range for fixup spans, those are meaningful to fixup, so just
        // prefer the non-fixup span.
        if first.anchor.ast_id == FIXUP_ERASED_FILE_AST_ID_MARKER {
            return Some(second);
        }
        if second.anchor.ast_id == FIXUP_ERASED_FILE_AST_ID_MARKER {
            return Some(first);
        }
        // FIXME: Once we can talk back to the client, implement a "long join" request for anchors
        // that differ in [AstId]s as joining those spans requires resolving the AstIds.
        if first.anchor != second.anchor {
            return None;
        }
        // Differing context, we can't merge these so prefer the one that's root
        if first.ctx != second.ctx {
            if first.ctx.is_root() {
                return Some(second);
            } else if second.ctx.is_root() {
                return Some(first);
            }
        }
        Some(Span {
            range: first.range.cover(second.range),
            anchor: second.anchor,
            ctx: second.ctx,
        })
    }
    fn subspan(
        &mut self,
        span: Self::Span,
        start: Bound<usize>,
        end: Bound<usize>,
    ) -> Option<Self::Span> {
        // We can't modify the span range for fixup spans, those are meaningful to fixup.
        if span.anchor.ast_id == FIXUP_ERASED_FILE_AST_ID_MARKER {
            return Some(span);
        }
        let length = span.range.len().into();

        let start: u32 = match start {
            Bound::Included(lo) => lo,
            Bound::Excluded(lo) => lo.checked_add(1)?,
            Bound::Unbounded => 0,
        }
        .try_into()
        .ok()?;

        let end: u32 = match end {
            Bound::Included(hi) => hi.checked_add(1)?,
            Bound::Excluded(hi) => hi,
            Bound::Unbounded => span.range.len().into(),
        }
        .try_into()
        .ok()?;

        // Bounds check the values, preventing addition overflow and OOB spans.
        let span_start = span.range.start().into();
        if (u32::MAX - start) < span_start
            || (u32::MAX - end) < span_start
            || start >= end
            || end > length
        {
            return None;
        }

        Some(Span {
            range: TextRange::new(TextSize::from(start), TextSize::from(end)) + span.range.start(),
            ..span
        })
    }

    fn resolved_at(&mut self, span: Self::Span, at: Self::Span) -> Self::Span {
        Span { ctx: at.ctx, ..span }
    }

    fn end(&mut self, span: Self::Span) -> Self::Span {
        // We can't modify the span range for fixup spans, those are meaningful to fixup.
        if span.anchor.ast_id == FIXUP_ERASED_FILE_AST_ID_MARKER {
            return span;
        }
        Span { range: TextRange::empty(span.range.end()), ..span }
    }

    fn start(&mut self, span: Self::Span) -> Self::Span {
        // We can't modify the span range for fixup spans, those are meaningful to fixup.
        if span.anchor.ast_id == FIXUP_ERASED_FILE_AST_ID_MARKER {
            return span;
        }
        Span { range: TextRange::empty(span.range.start()), ..span }
    }

    fn line(&mut self, _span: Self::Span) -> usize {
        // FIXME requires db to resolve line index, THIS IS NOT INCREMENTAL
        1
    }

    fn column(&mut self, _span: Self::Span) -> usize {
        // FIXME requires db to resolve line index, THIS IS NOT INCREMENTAL
        1
    }
}

impl server::Symbol for RaSpanServer {
    fn normalize_and_validate_ident(&mut self, string: &str) -> Result<Self::Symbol, ()> {
        // FIXME: nfc-normalize and validate idents
        Ok(<Self as server::Server>::intern_symbol(string))
    }
}

impl server::Server for RaSpanServer {
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
    use span::{EditionedFileId, FileId, SyntaxContext};

    use super::*;

    #[test]
    fn test_ra_server_to_string() {
        let span = Span {
            range: TextRange::empty(TextSize::new(0)),
            anchor: span::SpanAnchor {
                file_id: EditionedFileId::current_edition(FileId::from_raw(0)),
                ast_id: span::ErasedFileAstId::from_raw(0),
            },
            ctx: SyntaxContext::root(span::Edition::CURRENT),
        };
        let s = TokenStream {
            token_trees: vec![
                tt::TokenTree::Leaf(tt::Leaf::Ident(tt::Ident {
                    sym: Symbol::intern("struct"),
                    span,
                    is_raw: tt::IdentIsRaw::No,
                })),
                tt::TokenTree::Leaf(tt::Leaf::Ident(tt::Ident {
                    sym: Symbol::intern("T"),
                    span,
                    is_raw: tt::IdentIsRaw::No,
                })),
                tt::TokenTree::Subtree(tt::Subtree {
                    delimiter: tt::Delimiter {
                        open: span,
                        close: span,
                        kind: tt::DelimiterKind::Brace,
                    },
                    len: 1,
                }),
                tt::TokenTree::Leaf(tt::Leaf::Literal(tt::Literal {
                    kind: tt::LitKind::Str,
                    symbol: Symbol::intern("string"),
                    suffix: None,
                    span,
                })),
            ],
        };

        assert_eq!(s.to_string(), "struct T {\"string\"}");
    }

    #[test]
    fn test_ra_server_from_str() {
        let span = Span {
            range: TextRange::empty(TextSize::new(0)),
            anchor: span::SpanAnchor {
                file_id: EditionedFileId::current_edition(FileId::from_raw(0)),
                ast_id: span::ErasedFileAstId::from_raw(0),
            },
            ctx: SyntaxContext::root(span::Edition::CURRENT),
        };
        let subtree_paren_a = vec![
            tt::TokenTree::Subtree(tt::Subtree {
                delimiter: tt::Delimiter {
                    open: span,
                    close: span,
                    kind: tt::DelimiterKind::Parenthesis,
                },
                len: 1,
            }),
            tt::TokenTree::Leaf(tt::Leaf::Ident(tt::Ident {
                is_raw: tt::IdentIsRaw::No,
                sym: Symbol::intern("a"),
                span,
            })),
        ];

        let t1 = TokenStream::from_str("(a)", span).unwrap();
        assert_eq!(t1.token_trees.len(), 2);
        assert!(t1.token_trees == subtree_paren_a);

        let t2 = TokenStream::from_str("(a);", span).unwrap();
        assert_eq!(t2.token_trees.len(), 3);
        assert!(t2.token_trees[0..2] == subtree_paren_a);

        let underscore = TokenStream::from_str("_", span).unwrap();
        assert!(
            underscore.token_trees[0]
                == tt::TokenTree::Leaf(tt::Leaf::Ident(tt::Ident {
                    sym: Symbol::intern("_"),
                    span,
                    is_raw: tt::IdentIsRaw::No,
                }))
        );
    }
}
