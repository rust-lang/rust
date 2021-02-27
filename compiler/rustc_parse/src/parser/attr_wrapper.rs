use super::attr::{InnerAttrPolicy, DEFAULT_UNEXPECTED_INNER_ATTR_ERR_MSG};
use super::{Capturing, FlatToken, ForceCollect, Parser, ReplaceRange, TokenCursor, TrailingToken};
use rustc_ast::token::{self, DelimToken, Token, TokenKind};
use rustc_ast::tokenstream::{
    AttributesData, CreateTokenStream, PreexpTokenStream, PreexpTokenTree,
};
use rustc_ast::tokenstream::{DelimSpan, LazyTokenStream, Spacing};
use rustc_ast::AstLike;
use rustc_ast::AttrVec;
use rustc_ast::{self as ast};
use rustc_errors::{error_code, PResult};
use rustc_span::{Span, DUMMY_SP};

use std::convert::TryInto;
use std::ops::Range;

use tracing::debug;

/// A wrapper type to ensure that the parser handles outer attributes correctly.
/// When we parse outer attributes, we need to ensure that we capture tokens
/// for the attribute target. This allows us to perform cfg-expansion on
/// a token stream before we invoke a derive proc-macro.
///
/// This wrapper prevents direct access to the underlying `Vec<ast::Attribute>`.
/// Parsing code can only get access to the underlying attributes
/// by passing an `AttrWrapper` to `collect_tokens_trailing_tokens`.
/// This makes it difficult to accidentally construct an AST node
/// (which stores a `Vec<ast::Attribute>`) without first collecting tokens.
///
/// This struct has its own module, to ensure that the parser code
/// cannot directly access the `attrs` field
#[derive(Debug, Clone)]
pub struct AttrWrapper {
    attrs: AttrVec,
    start_pos: usize,
}

// This struct is passed around very frequently,
// so make sure it doesn't accidentally get larger
#[cfg(target_arch = "x86_64")]
rustc_data_structures::static_assert_size!(AttrWrapper, 16);

impl AttrWrapper {
    pub fn empty() -> AttrWrapper {
        AttrWrapper { attrs: AttrVec::new(), start_pos: usize::MAX }
    }
    // FIXME: Delay span bug here?
    pub(crate) fn take_for_recovery(self) -> AttrVec {
        self.attrs
    }
    pub fn is_empty(&self) -> bool {
        self.attrs.is_empty()
    }

    pub fn maybe_needs_tokens(&self) -> bool {
        crate::parser::attr::maybe_needs_tokens(&self.attrs)
    }
}

// Produces a `TokenStream` on-demand. Using `cursor_snapshot`
// and `num_calls`, we can reconstruct the `TokenStream` seen
// by the callback. This allows us to avoid producing a `TokenStream`
// if it is never needed - for example, a captured `macro_rules!`
// argument that is never passed to a proc macro.
// In practice token stream creation happens rarely compared to
// calls to `collect_tokens` (see some statistics in #78736),
// so we are doing as little up-front work as possible.
//
// This also makes `Parser` very cheap to clone, since
// there is no intermediate collection buffer to clone.
#[derive(Clone)]
struct LazyTokenStreamImpl {
    start_token: (Token, Spacing),
    cursor_snapshot: TokenCursor,
    num_calls: u32,
    desugar_doc_comments: bool,
    append_unglued_token: Option<(Token, Spacing)>,
    replace_ranges: Box<[ReplaceRange]>,
}

impl CreateTokenStream for LazyTokenStreamImpl {
    fn create_token_stream(&self) -> PreexpTokenStream {
        let num_calls = self.num_calls;
        // The token produced by the final call to `next` or `next_desugared`
        // was not actually consumed by the callback. The combination
        // of chaining the initial token and using `take` produces the desired
        // result - we produce an empty `TokenStream` if no calls were made,
        // and omit the final token otherwise.
        let mut cursor_snapshot = self.cursor_snapshot.clone();
        let tokens = std::iter::once(self.start_token.clone())
            .chain((0..self.num_calls).map(|_| {
                if self.desugar_doc_comments {
                    cursor_snapshot.next_desugared()
                } else {
                    cursor_snapshot.next()
                }
            }))
            .take(num_calls as usize)
            .map(|(token, spacing)| (FlatToken::Token(token), spacing));

        if !self.replace_ranges.is_empty() {
            let mut tokens: Vec<_> = tokens.collect();

            let mut replace_ranges = self.replace_ranges.clone();
            replace_ranges.sort_by_key(|(range, _)| range.start);
            replace_ranges.reverse();

            for (range, new_tokens) in replace_ranges.iter() {
                assert!(!range.is_empty(), "Cannot replace an empty range: {:?}", range);
                // Replace ranges are only allowed to decrease the number of tokens.
                assert!(
                    range.len() >= new_tokens.len(),
                    "Range {:?} has greater len than {:?}",
                    range,
                    new_tokens
                );

                // Replace any removed tokens with `FlatToken::Empty`.
                // This keeps the total length of `tokens` constant throughout the
                // replacement process, allowing us to use all of the `ReplaceRanges` entries
                // without adjusting indices.
                let filler = std::iter::repeat((FlatToken::Empty, Spacing::Alone))
                    .take(range.len() - new_tokens.len());

                tokens.splice(
                    (range.start as usize)..(range.end as usize),
                    new_tokens.clone().into_iter().chain(filler),
                );
            }

            make_token_stream(tokens.into_iter(), self.append_unglued_token.clone())
        } else {
            make_token_stream(tokens, self.append_unglued_token.clone())
        }
    }
}

impl<'a> Parser<'a> {
    /// Parses attributes that appear before an item.
    pub(super) fn parse_outer_attributes(&mut self) -> PResult<'a, AttrWrapper> {
        let mut attrs: Vec<ast::Attribute> = Vec::new();
        let mut just_parsed_doc_comment = false;
        let start_pos = self.token_cursor.num_next_calls;
        loop {
            debug!("parse_outer_attributes: self.token={:?}", self.token);
            let attr = if self.check(&token::Pound) {
                let inner_error_reason = if just_parsed_doc_comment {
                    "an inner attribute is not permitted following an outer doc comment"
                } else if !attrs.is_empty() {
                    "an inner attribute is not permitted following an outer attribute"
                } else {
                    DEFAULT_UNEXPECTED_INNER_ATTR_ERR_MSG
                };
                let inner_parse_policy = InnerAttrPolicy::Forbidden {
                    reason: inner_error_reason,
                    saw_doc_comment: just_parsed_doc_comment,
                    prev_attr_sp: attrs.last().map(|a| a.span),
                };
                just_parsed_doc_comment = false;
                Some(self.parse_attribute(inner_parse_policy)?)
            } else if let token::DocComment(comment_kind, attr_style, data) = self.token.kind {
                if attr_style != ast::AttrStyle::Outer {
                    self.sess
                        .span_diagnostic
                        .struct_span_err_with_code(
                            self.token.span,
                            "expected outer doc comment",
                            error_code!(E0753),
                        )
                        .note(
                            "inner doc comments like this (starting with \
                         `//!` or `/*!`) can only appear before items",
                        )
                        .emit();
                }
                self.bump();
                just_parsed_doc_comment = true;
                Some(rustc_ast::attr::mk_doc_comment(
                    comment_kind,
                    attr_style,
                    data,
                    self.prev_token.span,
                ))
            } else {
                None
            };

            if let Some(attr) = attr {
                attrs.push(attr);
            } else {
                break;
            }
        }
        Ok(AttrWrapper { attrs: attrs.into(), start_pos })
    }

    /// Records all tokens consumed by the provided callback,
    /// including the current token. These tokens are collected
    /// into a `LazyTokenStream`, and returned along with the result
    /// of the callback.
    ///
    /// Note: If your callback consumes an opening delimiter
    /// (including the case where you call `collect_tokens`
    /// when the current token is an opening delimeter),
    /// you must also consume the corresponding closing delimiter.
    ///
    /// That is, you can consume
    /// `something ([{ }])` or `([{}])`, but not `([{}]`
    ///
    /// This restriction shouldn't be an issue in practice,
    /// since this function is used to record the tokens for
    /// a parsed AST item, which always has matching delimiters.
    pub fn collect_tokens_trailing_token<R: AstLike>(
        &mut self,
        attrs: AttrWrapper,
        force_collect: ForceCollect,
        f: impl FnOnce(&mut Self, Vec<ast::Attribute>) -> PResult<'a, (R, TrailingToken)>,
    ) -> PResult<'a, R> {
        // We have no attributes that could observe the tokens, and there
        // are no encloding `capture_tokens` calls that need our tokens for
        // eager expansion of attributes.
        if matches!(force_collect, ForceCollect::No)
            && !attrs.maybe_needs_tokens()
            && !R::SUPPORTS_INNER_ATTRS
            && !(matches!(self.capture_state.capturing, Capturing::Yes { tokens_for_attrs: true })
                && ast::ast_like::has_cfg_or_cfg_any(&attrs.attrs))
        {
            return Ok(f(self, attrs.attrs.into())?.0);
        }

        let start_token = (self.token.clone(), self.token_spacing);
        let cursor_snapshot = self.token_cursor.clone();

        let has_outer_attrs = !attrs.attrs.is_empty();
        let prev_capturing = self.capture_state.capturing;
        let outer_attrs_needs_tokens = super::attr::maybe_needs_tokens(&attrs.attrs);
        self.capture_state.capturing = match prev_capturing {
            Capturing::No => Capturing::Yes { tokens_for_attrs: outer_attrs_needs_tokens },
            Capturing::Yes { tokens_for_attrs } => {
                Capturing::Yes { tokens_for_attrs: tokens_for_attrs || outer_attrs_needs_tokens }
            }
        };
        let replace_ranges_start = self.capture_state.replace_ranges.len();

        let ret = f(self, attrs.attrs.into());

        let replace_ranges_end = self.capture_state.replace_ranges.len();
        self.capture_state.capturing = prev_capturing;

        let (mut ret, trailing) = ret?;

        // We have no attributes that could observe the tokens, and there
        // are no encloding `capture_tokens` calls that need our tokens for
        // eager expansion of attributes.
        if matches!(force_collect, ForceCollect::No)
            && !crate::parser::attr::maybe_needs_tokens(ret.attrs())
            // Subtle: We call `has_cfg_or_cfg_any` with the attrs from `ret`.
            // This ensures that we consider inner attributes (e.g. `#![cfg]`),
            // which require us to have tokens available
            // We also call `has_cfg_or_cfg_any` at the beginning of this function,
            // but we only bail out if there's no possibility of inner attributes
            // (!R::SUPPORTS_INNER_ATTRS)
            && !(matches!(self.capture_state.capturing, Capturing:: Yes { tokens_for_attrs: true })
                 && ast::ast_like::has_cfg_or_cfg_any(ret.attrs()))
        {
            return Ok(ret);
        }


        let cursor_snapshot_next_calls = cursor_snapshot.num_next_calls;
        let mut end_pos = self.token_cursor.num_next_calls;

        match trailing {
            TrailingToken::None => {}
            TrailingToken::Semi => {
                assert_eq!(self.token.kind, token::Semi);
                end_pos += 1;
            }
            TrailingToken::MaybeComma => {
                if self.token.kind == token::Comma {
                    end_pos += 1;
                }
            }
        }

        let num_calls = end_pos - cursor_snapshot_next_calls;

        // Handle previous replace ranges
        let replace_ranges: Box<[ReplaceRange]> = if ret.attrs().is_empty() {
            Box::new([])
        } else {
            let start_calls: u32 = cursor_snapshot_next_calls.try_into().unwrap();
            self.capture_state.replace_ranges[replace_ranges_start..replace_ranges_end]
                .iter()
                .cloned()
                .map(|(range, tokens)| {
                    ((range.start - start_calls)..(range.end - start_calls), tokens)
                })
                .collect()
        };

        let tokens = LazyTokenStream::new(LazyTokenStreamImpl {
            start_token,
            num_calls: num_calls.try_into().unwrap(),
            cursor_snapshot,
            desugar_doc_comments: self.desugar_doc_comments,
            append_unglued_token: self.token_cursor.append_unglued_token.clone(),
            replace_ranges: replace_ranges.into(),
        });

        let final_attrs: Option<AttributesData> = ret.finalize_tokens(tokens);
        if let Some(final_attrs) = final_attrs {
            if matches!(self.capture_state.capturing, Capturing::Yes { tokens_for_attrs: true }) {
                let start_pos =
                    if has_outer_attrs { attrs.start_pos } else { cursor_snapshot_next_calls };
                let mut new_tokens = vec![(FlatToken::AttrTarget(final_attrs), Spacing::Alone)];
                if let Some((unglued, spacing)) = self.token_cursor.append_unglued_token.clone() {
                    end_pos += 1;
                    new_tokens.push((FlatToken::Token(unglued), spacing));
                }
                let range: Range<u32> =
                    (start_pos.try_into().unwrap())..(end_pos.try_into().unwrap());
                self.capture_state.replace_ranges.push((range, new_tokens));
            }
        }

        // We only need replace ranges to handle `#[derive]`. If all of
        // the outer calls to `capture_tokens` had no outer attributes,
        // then we can't possibly have a `derive`
        if !matches!(self.capture_state.capturing, Capturing::Yes { tokens_for_attrs: true }) {
            self.capture_state.replace_ranges.clear();
        }
        Ok(ret)
    }
}

/// Converts a flattened iterator of tokens (including open and close delimiter tokens)
/// into a `TokenStream`, creating a `TokenTree::Delimited` for each matching pair
/// of open and close delims.
fn make_token_stream(
    tokens: impl Iterator<Item = (FlatToken, Spacing)>,
    append_unglued_token: Option<(Token, Spacing)>,
) -> PreexpTokenStream {
    //let orig_tokens = tokens.clone();
    #[derive(Debug)]
    struct FrameData {
        open: Span,
        open_delim: DelimToken,
        inner: Vec<(PreexpTokenTree, Spacing)>,
    }
    let mut stack =
        vec![FrameData { open: DUMMY_SP, open_delim: DelimToken::NoDelim, inner: vec![] }];
    for (token, spacing) in tokens {
        match token {
            FlatToken::Token(Token { kind: TokenKind::OpenDelim(delim), span }) => {
                stack.push(FrameData { open: span, open_delim: delim, inner: vec![] });
            }
            FlatToken::Token(Token { kind: TokenKind::CloseDelim(delim), span }) => {
                let frame_data = stack.pop().expect("Token stack was empty!");
                if stack.is_empty() {
                    panic!("Popped token {:?} for last frame {:?}", token, frame_data);
                }
                assert_eq!(
                    frame_data.open_delim, delim,
                    "Mismatched open/close delims: open={:?} close={:?}",
                    frame_data.open, span
                );
                let dspan = DelimSpan::from_pair(frame_data.open, span);
                let stream = PreexpTokenStream::new(frame_data.inner);
                let delimited = PreexpTokenTree::Delimited(dspan, delim, stream);
                stack
                    .last_mut()
                    .unwrap_or_else(|| {
                        panic!("Bottom token frame is missing for token: {:?}", token)
                    })
                    .inner
                    .push((delimited, Spacing::Alone));
            }
            FlatToken::Token(token) => stack
                .last_mut()
                .expect("Bottom token frame is missing!")
                .inner
                .push((PreexpTokenTree::Token(token), spacing)),
            FlatToken::AttrTarget(data) => stack
                .last_mut()
                .expect("Bottom token frame is missing!")
                .inner
                .push((PreexpTokenTree::Attributes(data), spacing)),
            FlatToken::Empty => {}
        }
    }
    let mut final_buf = stack.pop().expect("Missing final buf!");
    if let Some((append_unglued_token, spacing)) = append_unglued_token {
        final_buf.inner.push((PreexpTokenTree::Token(append_unglued_token), spacing));
    }
    assert!(stack.is_empty(), "Stack should be empty: final_buf={:?} stack={:?}", final_buf, stack);
    PreexpTokenStream::new(final_buf.inner)
}
