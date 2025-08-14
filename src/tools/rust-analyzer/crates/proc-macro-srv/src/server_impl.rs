//! proc-macro server implementation
//!
//! Based on idea from <https://github.com/fedochet/rust-proc-macro-expander>
//! The lib-proc-macro server backend is `TokenStream`-agnostic, such that
//! we could provide any TokenStream implementation.
//! The original idea from fedochet is using proc-macro2 as backend,
//! we use tt instead for better integration with RA.
//!
//! FIXME: No span and source file information is implemented yet

use std::fmt;

use intern::Symbol;
use proc_macro::bridge;

mod token_stream;
pub use token_stream::TokenStream;

pub mod rust_analyzer_span;
pub mod token_id;

use tt::Spacing;

#[derive(Clone)]
pub(crate) struct TopSubtree<S>(pub(crate) Vec<tt::TokenTree<S>>);

impl<S: Copy + fmt::Debug> fmt::Debug for TopSubtree<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&tt::TokenTreesView::new(&self.0), f)
    }
}

impl<S: Copy> TopSubtree<S> {
    pub(crate) fn top_subtree(&self) -> &tt::Subtree<S> {
        let tt::TokenTree::Subtree(subtree) = &self.0[0] else {
            unreachable!("the first token tree is always the top subtree");
        };
        subtree
    }

    pub(crate) fn from_bridge(group: bridge::Group<TokenStream<S>, S>) -> Self {
        let delimiter = delim_to_internal(group.delimiter, group.span);
        let mut tts =
            group.stream.map(|it| it.token_trees).unwrap_or_else(|| Vec::with_capacity(1));
        tts.insert(0, tt::TokenTree::Subtree(tt::Subtree { delimiter, len: tts.len() as u32 }));
        TopSubtree(tts)
    }
}

fn delim_to_internal<S>(d: proc_macro::Delimiter, span: bridge::DelimSpan<S>) -> tt::Delimiter<S> {
    let kind = match d {
        proc_macro::Delimiter::Parenthesis => tt::DelimiterKind::Parenthesis,
        proc_macro::Delimiter::Brace => tt::DelimiterKind::Brace,
        proc_macro::Delimiter::Bracket => tt::DelimiterKind::Bracket,
        proc_macro::Delimiter::None => tt::DelimiterKind::Invisible,
    };
    tt::Delimiter { open: span.open, close: span.close, kind }
}

fn delim_to_external<S>(d: tt::Delimiter<S>) -> proc_macro::Delimiter {
    match d.kind {
        tt::DelimiterKind::Parenthesis => proc_macro::Delimiter::Parenthesis,
        tt::DelimiterKind::Brace => proc_macro::Delimiter::Brace,
        tt::DelimiterKind::Bracket => proc_macro::Delimiter::Bracket,
        tt::DelimiterKind::Invisible => proc_macro::Delimiter::None,
    }
}

#[allow(unused)]
fn spacing_to_internal(spacing: proc_macro::Spacing) -> Spacing {
    match spacing {
        proc_macro::Spacing::Alone => Spacing::Alone,
        proc_macro::Spacing::Joint => Spacing::Joint,
    }
}

#[allow(unused)]
fn spacing_to_external(spacing: Spacing) -> proc_macro::Spacing {
    match spacing {
        Spacing::Alone | Spacing::JointHidden => proc_macro::Spacing::Alone,
        Spacing::Joint => proc_macro::Spacing::Joint,
    }
}

fn literal_kind_to_external(kind: tt::LitKind) -> bridge::LitKind {
    match kind {
        tt::LitKind::Byte => bridge::LitKind::Byte,
        tt::LitKind::Char => bridge::LitKind::Char,
        tt::LitKind::Integer => bridge::LitKind::Integer,
        tt::LitKind::Float => bridge::LitKind::Float,
        tt::LitKind::Str => bridge::LitKind::Str,
        tt::LitKind::StrRaw(r) => bridge::LitKind::StrRaw(r),
        tt::LitKind::ByteStr => bridge::LitKind::ByteStr,
        tt::LitKind::ByteStrRaw(r) => bridge::LitKind::ByteStrRaw(r),
        tt::LitKind::CStr => bridge::LitKind::CStr,
        tt::LitKind::CStrRaw(r) => bridge::LitKind::CStrRaw(r),
        tt::LitKind::Err(_) => bridge::LitKind::ErrWithGuar,
    }
}

fn literal_kind_to_internal(kind: bridge::LitKind) -> tt::LitKind {
    match kind {
        bridge::LitKind::Byte => tt::LitKind::Byte,
        bridge::LitKind::Char => tt::LitKind::Char,
        bridge::LitKind::Str => tt::LitKind::Str,
        bridge::LitKind::StrRaw(r) => tt::LitKind::StrRaw(r),
        bridge::LitKind::ByteStr => tt::LitKind::ByteStr,
        bridge::LitKind::ByteStrRaw(r) => tt::LitKind::ByteStrRaw(r),
        bridge::LitKind::CStr => tt::LitKind::CStr,
        bridge::LitKind::CStrRaw(r) => tt::LitKind::CStrRaw(r),
        bridge::LitKind::Integer => tt::LitKind::Integer,
        bridge::LitKind::Float => tt::LitKind::Float,
        bridge::LitKind::ErrWithGuar => tt::LitKind::Err(()),
    }
}

pub(super) fn literal_from_str<Span: Copy>(
    s: &str,
    span: Span,
) -> Result<bridge::Literal<Span, Symbol>, ()> {
    use proc_macro::bridge::LitKind;
    use rustc_lexer::{LiteralKind, Token, TokenKind};

    let mut tokens = rustc_lexer::tokenize(s, rustc_lexer::FrontmatterAllowed::No);
    let minus_or_lit = tokens.next().unwrap_or(Token { kind: TokenKind::Eof, len: 0 });

    let lit = if minus_or_lit.kind == TokenKind::Minus {
        let lit = tokens.next().ok_or(())?;
        if !matches!(
            lit.kind,
            TokenKind::Literal { kind: LiteralKind::Int { .. } | LiteralKind::Float { .. }, .. }
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

    Ok(bridge::Literal { kind, symbol: Symbol::intern(lit), suffix, span })
}

pub(super) fn from_token_tree<Span: Copy>(
    tree: bridge::TokenTree<TokenStream<Span>, Span, Symbol>,
) -> TokenStream<Span> {
    match tree {
        bridge::TokenTree::Group(group) => {
            let group = TopSubtree::from_bridge(group);
            TokenStream { token_trees: group.0 }
        }

        bridge::TokenTree::Ident(ident) => {
            let text = ident.sym;
            let ident: tt::Ident<Span> = tt::Ident {
                sym: text,
                span: ident.span,
                is_raw: if ident.is_raw { tt::IdentIsRaw::Yes } else { tt::IdentIsRaw::No },
            };
            let leaf = tt::Leaf::from(ident);
            let tree = tt::TokenTree::from(leaf);
            TokenStream { token_trees: vec![tree] }
        }

        bridge::TokenTree::Literal(literal) => {
            let mut token_trees = Vec::new();
            let mut symbol = literal.symbol;
            if matches!(
                literal.kind,
                proc_macro::bridge::LitKind::Integer | proc_macro::bridge::LitKind::Float
            ) && symbol.as_str().starts_with('-')
            {
                token_trees.push(tt::TokenTree::Leaf(tt::Leaf::Punct(tt::Punct {
                    spacing: tt::Spacing::Alone,
                    span: literal.span,
                    char: '-',
                })));
                symbol = Symbol::intern(&symbol.as_str()[1..]);
            }
            let literal = tt::Literal {
                symbol,
                suffix: literal.suffix,
                span: literal.span,
                kind: literal_kind_to_internal(literal.kind),
            };
            let leaf: tt::Leaf<Span> = tt::Leaf::from(literal);
            let tree = tt::TokenTree::from(leaf);
            token_trees.push(tree);
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
