//! Mapping between `TokenId`s and the token's position in macro definitions or inputs.

use std::hash::Hash;

use parser::{SyntaxKind, T};
use syntax::{TextRange, TextSize};

use crate::syntax_bridge::SyntheticTokenId;

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
enum TokenTextRange {
    Token(TextRange),
    Delimiter(TextRange),
}

impl TokenTextRange {
    fn by_kind(self, kind: SyntaxKind) -> Option<TextRange> {
        match self {
            TokenTextRange::Token(it) => Some(it),
            TokenTextRange::Delimiter(it) => match kind {
                T!['{'] | T!['('] | T!['['] => Some(TextRange::at(it.start(), 1.into())),
                T!['}'] | T![')'] | T![']'] => {
                    Some(TextRange::at(it.end() - TextSize::of('}'), 1.into()))
                }
                _ => None,
            },
        }
    }
}

/// Maps `tt::TokenId` to the relative range of the original token.
#[derive(Debug, PartialEq, Eq, Clone, Default, Hash)]
pub struct TokenMap {
    /// Maps `tt::TokenId` to the *relative* source range.
    entries: Vec<(tt::TokenId, TokenTextRange)>,
    pub synthetic_entries: Vec<(tt::TokenId, SyntheticTokenId)>,
}

impl TokenMap {
    pub fn token_by_range(&self, relative_range: TextRange) -> Option<tt::TokenId> {
        let &(token_id, _) = self.entries.iter().find(|(_, range)| match range {
            TokenTextRange::Token(it) => *it == relative_range,
            TokenTextRange::Delimiter(it) => {
                let open = TextRange::at(it.start(), 1.into());
                let close = TextRange::at(it.end() - TextSize::of('}'), 1.into());
                open == relative_range || close == relative_range
            }
        })?;
        Some(token_id)
    }

    pub fn ranges_by_token(
        &self,
        token_id: tt::TokenId,
        kind: SyntaxKind,
    ) -> impl Iterator<Item = TextRange> + '_ {
        self.entries
            .iter()
            .filter(move |&&(tid, _)| tid == token_id)
            .filter_map(move |(_, range)| range.by_kind(kind))
    }

    pub fn synthetic_token_id(&self, token_id: tt::TokenId) -> Option<SyntheticTokenId> {
        self.synthetic_entries.iter().find(|(tid, _)| *tid == token_id).map(|(_, id)| *id)
    }

    pub fn first_range_by_token(
        &self,
        token_id: tt::TokenId,
        kind: SyntaxKind,
    ) -> Option<TextRange> {
        self.ranges_by_token(token_id, kind).next()
    }

    pub(crate) fn shrink_to_fit(&mut self) {
        self.entries.shrink_to_fit();
        self.synthetic_entries.shrink_to_fit();
    }

    pub(crate) fn insert(&mut self, token_id: tt::TokenId, relative_range: TextRange) {
        self.entries.push((token_id, TokenTextRange::Token(relative_range)));
    }

    pub(crate) fn insert_synthetic(&mut self, token_id: tt::TokenId, id: SyntheticTokenId) {
        self.synthetic_entries.push((token_id, id));
    }

    pub(crate) fn insert_delim(
        &mut self,
        token_id: tt::TokenId,
        open_relative_range: TextRange,
        close_relative_range: TextRange,
    ) -> usize {
        let res = self.entries.len();
        let cover = open_relative_range.cover(close_relative_range);

        self.entries.push((token_id, TokenTextRange::Delimiter(cover)));
        res
    }

    pub(crate) fn update_close_delim(&mut self, idx: usize, close_relative_range: TextRange) {
        let (_, token_text_range) = &mut self.entries[idx];
        if let TokenTextRange::Delimiter(dim) = token_text_range {
            let cover = dim.cover(close_relative_range);
            *token_text_range = TokenTextRange::Delimiter(cover);
        }
    }

    pub(crate) fn remove_delim(&mut self, idx: usize) {
        // FIXME: This could be accidentally quadratic
        self.entries.remove(idx);
    }

    pub fn entries(&self) -> impl Iterator<Item = (tt::TokenId, TextRange)> + '_ {
        self.entries.iter().filter_map(|&(tid, tr)| match tr {
            TokenTextRange::Token(range) => Some((tid, range)),
            TokenTextRange::Delimiter(_) => None,
        })
    }
}
