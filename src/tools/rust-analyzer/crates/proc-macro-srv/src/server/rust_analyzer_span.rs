//! proc-macro server backend based on rust-analyzer's internal span representation
//! This backend is used solely by rust-analyzer as it ties into rust-analyzer internals.
//!
//! It is an unfortunate result of how the proc-macro API works that we need to look into the
//! concrete representation of the spans, and as such, RustRover cannot make use of this unless they
//! change their representation to be compatible with rust-analyzer's.
use std::{
    collections::{HashMap, HashSet},
    iter,
    ops::{Bound, Range},
};

use ::tt::{TextRange, TextSize};
use proc_macro::bridge::{self, server};
use span::{Span, FIXUP_ERASED_FILE_AST_ID_MARKER};

use crate::server::{
    delim_to_external, delim_to_internal, token_stream::TokenStreamBuilder, LiteralFormatter,
    Symbol, SymbolInternerRef, SYMBOL_INTERNER,
};
mod tt {
    pub use ::tt::*;

    pub type Subtree = ::tt::Subtree<super::Span>;
    pub type TokenTree = ::tt::TokenTree<super::Span>;
    pub type Leaf = ::tt::Leaf<super::Span>;
    pub type Literal = ::tt::Literal<super::Span>;
    pub type Punct = ::tt::Punct<super::Span>;
    pub type Ident = ::tt::Ident<super::Span>;
}

type TokenStream = crate::server::TokenStream<Span>;

#[derive(Clone)]
pub struct SourceFile;
pub struct FreeFunctions;

pub struct RaSpanServer {
    pub(crate) interner: SymbolInternerRef,
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
    type SourceFile = SourceFile;
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
        // FIXME: keep track of LitKind and Suffix
        Ok(bridge::Literal {
            kind: bridge::LitKind::Integer, // dummy
            symbol: Symbol::intern(self.interner, s),
            suffix: None,
            span: self.call_site,
        })
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
                let group = tt::Subtree {
                    delimiter: delim_to_internal(group.delimiter, group.span),
                    token_trees: match group.stream {
                        Some(stream) => stream.into_iter().collect(),
                        None => Box::new([]),
                    },
                };
                let tree = tt::TokenTree::from(group);
                Self::TokenStream::from_iter(iter::once(tree))
            }

            bridge::TokenTree::Ident(ident) => {
                let text = ident.sym.text(self.interner);
                let text =
                    if ident.is_raw { ::tt::SmolStr::from_iter(["r#", &text]) } else { text };
                let ident: tt::Ident = tt::Ident { text, span: ident.span };
                let leaf = tt::Leaf::from(ident);
                let tree = tt::TokenTree::from(leaf);
                Self::TokenStream::from_iter(iter::once(tree))
            }

            bridge::TokenTree::Literal(literal) => {
                let literal = LiteralFormatter(literal);
                let text = literal.with_stringify_parts(self.interner, |parts| {
                    ::tt::SmolStr::from_iter(parts.iter().copied())
                });

                let literal = tt::Literal { text, span: literal.0.span };
                let leaf: tt::Leaf = tt::Leaf::from(literal);
                let tree = tt::TokenTree::from(leaf);
                Self::TokenStream::from_iter(iter::once(tree))
            }

            bridge::TokenTree::Punct(p) => {
                let punct = tt::Punct {
                    char: p.ch as char,
                    spacing: if p.joint { tt::Spacing::Joint } else { tt::Spacing::Alone },
                    span: p.span,
                };
                let leaf = tt::Leaf::from(punct);
                let tree = tt::TokenTree::from(leaf);
                Self::TokenStream::from_iter(iter::once(tree))
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
                        // FIXME: handle literal kinds
                        kind: bridge::LitKind::Integer, // dummy
                        symbol: Symbol::intern(self.interner, &lit.text),
                        // FIXME: handle suffixes
                        suffix: None,
                        span: lit.span,
                    })
                }
                tt::TokenTree::Leaf(tt::Leaf::Punct(punct)) => {
                    bridge::TokenTree::Punct(bridge::Punct {
                        ch: punct.char as u8,
                        joint: punct.spacing == tt::Spacing::Joint,
                        span: punct.span,
                    })
                }
                tt::TokenTree::Subtree(subtree) => bridge::TokenTree::Group(bridge::Group {
                    delimiter: delim_to_external(subtree.delimiter),
                    stream: if subtree.token_trees.is_empty() {
                        None
                    } else {
                        Some(subtree.token_trees.into_vec().into_iter().collect())
                    },
                    span: bridge::DelimSpan::from_single(subtree.delimiter.open),
                }),
            })
            .collect()
    }
}

impl server::SourceFile for RaSpanServer {
    // FIXME these are all stubs
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

impl server::Span for RaSpanServer {
    fn debug(&mut self, span: Self::Span) -> String {
        format!("{:?}", span)
    }
    fn source_file(&mut self, _span: Self::Span) -> Self::SourceFile {
        // FIXME stub, requires db
        SourceFile {}
    }
    fn save_span(&mut self, _span: Self::Span) -> usize {
        // FIXME stub, requires builtin quote! implementation
        0
    }
    fn recover_proc_macro_span(&mut self, _id: usize) -> Self::Span {
        // FIXME stub, requires builtin quote! implementation
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
        0
    }

    fn column(&mut self, _span: Self::Span) -> usize {
        // FIXME requires db to resolve line index, THIS IS NOT INCREMENTAL
        0
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
        // FIXME: should be `self.interner` once the proc-macro api allows it.
        Symbol::intern(&SYMBOL_INTERNER, &::tt::SmolStr::from(ident))
    }

    fn with_symbol_string(symbol: &Self::Symbol, f: impl FnOnce(&str)) {
        // FIXME: should be `self.interner` once the proc-macro api allows it.
        f(symbol.text(&SYMBOL_INTERNER).as_str())
    }
}
