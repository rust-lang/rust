//! Mapping between `TokenId`s and the token's position in macro definitions or inputs.

use std::hash::Hash;

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
pub struct TokenMap<S> {
    // FIXME: This needs to be sorted by (FileId, AstId)
    // Then we can do a binary search on the file id,
    // then a bin search on the ast id
    pub span_map: Vec<(TextRange, S)>,
    // span_map2: rustc_hash::FxHashMap<TextRange, usize>,
    pub real_file: bool,
}

impl<S> Default for TokenMap<S> {
    fn default() -> Self {
        Self { span_map: Vec::new(), real_file: true }
    }
}

impl<S: Span> TokenMap<S> {
    pub(crate) fn shrink_to_fit(&mut self) {
        self.span_map.shrink_to_fit();
    }

    pub(crate) fn insert(&mut self, range: TextRange, span: S) {
        self.span_map.push((range, span));
    }

    pub fn ranges_with_span(&self, span: S) -> impl Iterator<Item = TextRange> + '_ {
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

    // FIXME: Should be infallible
    pub fn span_for_range(&self, range: TextRange) -> Option<S> {
        // TODO FIXME: make this proper
        self.span_map
            .iter()
            .filter_map(|(r, s)| Some((r, s, r.intersect(range)?)))
            .max_by_key(|(_, _, intersection)| intersection.len())
            .map(|(_, &s, _)| s)
            .or_else(|| {
                if self.real_file {
                    None
                } else {
                    panic!("no span for range {range:?} in {:#?}", self.span_map)
                }
            })
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
