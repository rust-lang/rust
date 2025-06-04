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
