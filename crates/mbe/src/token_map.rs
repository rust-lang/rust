//! Mapping between `TokenId`s and the token's position in macro definitions or inputs.

use std::hash::Hash;

use stdx::itertools::Itertools;
use syntax::TextRange;
use tt::Span;

// pub type HirFile = u32;
// pub type FileRange = (HirFile, TextRange);
// Option<MacroCallId>, LocalSyntaxContet
// pub type SyntaxContext = ();
// pub type LocalSyntaxContext = u32;

/// Maps absolute text ranges for the corresponding file to the relevant span data.
#[derive(Debug, PartialEq, Eq, Clone, Hash)]
// FIXME: Rename to SpanMap
pub struct TokenMap<S: Span> {
    // FIXME: This needs to be sorted by (FileId, AstId)
    // Then we can do a binary search on the file id,
    // then a bin search on the ast id
    pub span_map: Vec<(TextRange, S)>,
    // span_map2: rustc_hash::FxHashMap<TextRange, usize>,
}

impl<S: Span> TokenMap<S> {
    pub fn empty() -> Self {
        Self { span_map: Vec::new() }
    }

    pub fn finish(&mut self) {
        debug_assert_eq!(
            self.span_map
                .iter()
                .sorted_by_key(|it| (it.0.start(), it.0.end()))
                .tuple_windows()
                .find(|(range, next)| range.0.end() != next.0.start()),
            None,
            "span map has holes!"
        );
        self.span_map.shrink_to_fit();
    }

    pub(crate) fn insert(&mut self, range: TextRange, span: S) {
        self.span_map.push((range, span));
    }

    pub fn ranges_with_span(&self, span: S) -> impl Iterator<Item = TextRange> + '_ {
        // FIXME: linear search
        // FIXME: Disregards resolving spans to get more matches! See ExpansionInfo::map_token_down
        self.span_map.iter().filter_map(
            move |(range, s)| {
                if s == &span {
                    Some(*range)
                } else {
                    None
                }
            },
        )
    }

    // FIXME: We need APIs for fetching the span of a token as well as for a whole node. The node
    // one *is* fallible though.
    // Token span fetching technically only needs an offset really, as the entire file span is
    // populated, where node fetching is more like fetching the spans at all source positions, and
    // then we need to verify that all those positions have the same context, if not we fail! But
    // how do we handle them having different span ranges?

    pub fn span_for_range(&self, range: TextRange) -> S {
        // TODO FIXME: make this proper
        self.span_map
            .iter()
            .filter_map(|(r, s)| Some((r, s, r.intersect(range).filter(|it| !it.is_empty())?)))
            .max_by_key(|(_, _, intersection)| intersection.len())
            .map_or_else(
                || panic!("no span for range {:?} in {:#?}", range, self.span_map),
                |(_, &s, _)| s,
            )
    }

    pub fn spans_for_node_range(&self, range: TextRange) -> impl Iterator<Item = S> + '_ {
        // TODO FIXME: make this proper
        self.span_map
            .iter()
            .filter(move |(r, _)| r.intersect(range).filter(|it| !it.is_empty()).is_some())
            .map(|&(_, s)| s)
    }

    // pub fn ranges_by_token(
    //     &self,
    //     token_id: tt::TokenId,
    //     kind: SyntaxKind,
    // ) -> impl Iterator<Item = TextRange> + '_ {
    //     self.entries
    //         .iter()
    //         .filter(move |&&(tid, _)| tid == token_id)
    //         .filter_map(move |(_, range)| range.by_kind(kind))
    // }

    // pub(crate) fn remove_delim(&mut self, idx: usize) {
    //     // FIXME: This could be accidentally quadratic
    //     self.entries.remove(idx);
    // }

    // pub fn entries(&self) -> impl Iterator<Item = (tt::TokenId, TextRange)> + '_ {
    //     self.entries.iter().filter_map(|&(tid, tr)| match tr {
    //         TokenTextRange::Token(range) => Some((tid, range)),
    //         TokenTextRange::Delimiter(_) => None,
    //     })
    // }

    // pub fn filter(&mut self, id: impl Fn(tt::TokenId) -> bool) {
    //     self.entries.retain(|&(tid, _)| id(tid));
    // }
    // pub fn synthetic_token_id(&self, token_id: tt::TokenId) -> Option<SyntheticTokenId> {
    //     self.synthetic_entries.iter().find(|(tid, _)| *tid == token_id).map(|(_, id)| *id)
    // }

    // pub fn first_range_by_token(
    //     &self,
    //     token_id: tt::TokenId,
    //     kind: SyntaxKind,
    // ) -> Option<TextRange> {
    //     self.ranges_by_token(token_id, kind).next()
    // }

    // pub(crate) fn insert(&mut self, token_id: tt::TokenId, relative_range: TextRange) {
    //     self.entries.push((token_id, TokenTextRange::Token(relative_range)));
    // }

    // pub(crate) fn insert_synthetic(&mut self, token_id: tt::TokenId, id: SyntheticTokenId) {
    //     self.synthetic_entries.push((token_id, id));
    // }

    // pub(crate) fn insert_delim(
    //     &mut self,
    //     token_id: tt::TokenId,
    //     open_relative_range: TextRange,
    //     close_relative_range: TextRange,
    // ) -> usize {
    //     let res = self.entries.len();
    //     let cover = open_relative_range.cover(close_relative_range);

    //     self.entries.push((token_id, TokenTextRange::Delimiter(cover)));
    //     res
    // }

    // pub(crate) fn update_close_delim(&mut self, idx: usize, close_relative_range: TextRange) {
    //     let (_, token_text_range) = &mut self.entries[idx];
    //     if let TokenTextRange::Delimiter(dim) = token_text_range {
    //         let cover = dim.cover(close_relative_range);
    //         *token_text_range = TokenTextRange::Delimiter(cover);
    //     }
    // }
}
