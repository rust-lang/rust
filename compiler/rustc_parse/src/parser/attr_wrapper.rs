use super::{Capturing, FlatToken, ForceCollect, Parser, ReplaceRange, TokenCursor, TrailingToken};
use rustc_ast::token::{self, Delimiter, Token, TokenKind};
use rustc_ast::tokenstream::{AttrTokenStream, AttributesData, ToAttrTokenStream};
use rustc_ast::tokenstream::{AttrTokenTree, DelimSpan, LazyAttrTokenStream, Spacing};
use rustc_ast::{self as ast};
use rustc_ast::{AttrVec, Attribute, HasAttrs, HasTokens};
use rustc_errors::PResult;
use rustc_session::parse::ParseSess;
use rustc_span::{sym, Span, DUMMY_SP};

use std::ops::Range;

/// A wrapper type to ensure that the parser handles outer attributes correctly.
/// When we parse outer attributes, we need to ensure that we capture tokens
/// for the attribute target. This allows us to perform cfg-expansion on
/// a token stream before we invoke a derive proc-macro.
///
/// This wrapper prevents direct access to the underlying `ast::AttrVec>`.
/// Parsing code can only get access to the underlying attributes
/// by passing an `AttrWrapper` to `collect_tokens_trailing_tokens`.
/// This makes it difficult to accidentally construct an AST node
/// (which stores an `ast::AttrVec`) without first collecting tokens.
///
/// This struct has its own module, to ensure that the parser code
/// cannot directly access the `attrs` field
#[derive(Debug, Clone)]
pub struct AttrWrapper {
    attrs: AttrVec,
    // The start of the outer attributes in the token cursor.
    // This allows us to create a `ReplaceRange` for the entire attribute
    // target, including outer attributes.
    start_pos: usize,
}

impl AttrWrapper {
    pub(super) fn new(attrs: AttrVec, start_pos: usize) -> AttrWrapper {
        AttrWrapper { attrs, start_pos }
    }
    pub fn empty() -> AttrWrapper {
        AttrWrapper { attrs: AttrVec::new(), start_pos: usize::MAX }
    }

    pub(crate) fn take_for_recovery(self, sess: &ParseSess) -> AttrVec {
        sess.span_diagnostic.delay_span_bug(
            self.attrs.get(0).map(|attr| attr.span).unwrap_or(DUMMY_SP),
            "AttrVec is taken for recovery but no error is produced",
        );

        self.attrs
    }

    /// Prepend `self.attrs` to `attrs`.
    // FIXME: require passing an NT to prevent misuse of this method
    pub(crate) fn prepend_to_nt_inner(self, attrs: &mut AttrVec) {
        let mut self_attrs = self.attrs;
        std::mem::swap(attrs, &mut self_attrs);
        attrs.extend(self_attrs);
    }

    pub fn is_empty(&self) -> bool {
        self.attrs.is_empty()
    }

    pub fn is_complete(&self) -> bool {
        crate::parser::attr::is_complete(&self.attrs)
    }
}

/// Returns `true` if `attrs` contains a `cfg` or `cfg_attr` attribute
fn has_cfg_or_cfg_attr(attrs: &[Attribute]) -> bool {
    // NOTE: Builtin attributes like `cfg` and `cfg_attr` cannot be renamed via imports.
    // Therefore, the absence of a literal `cfg` or `cfg_attr` guarantees that
    // we don't need to do any eager expansion.
    attrs.iter().any(|attr| {
        attr.ident().is_some_and(|ident| ident.name == sym::cfg || ident.name == sym::cfg_attr)
    })
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
struct LazyAttrTokenStreamImpl {
    start_token: (Token, Spacing),
    cursor_snapshot: TokenCursor,
    num_calls: usize,
    break_last_token: bool,
    replace_ranges: Box<[ReplaceRange]>,
}

impl ToAttrTokenStream for LazyAttrTokenStreamImpl {
    fn to_attr_token_stream(&self) -> AttrTokenStream {
        // The token produced by the final call to `{,inlined_}next` was not
        // actually consumed by the callback. The combination of chaining the
        // initial token and using `take` produces the desired result - we
        // produce an empty `TokenStream` if no calls were made, and omit the
        // final token otherwise.
        let mut cursor_snapshot = self.cursor_snapshot.clone();
        let tokens =
            std::iter::once((FlatToken::Token(self.start_token.0.clone()), self.start_token.1))
                .chain((0..self.num_calls).map(|_| {
                    let token = cursor_snapshot.next(cursor_snapshot.desugar_doc_comments);
                    (FlatToken::Token(token.0), token.1)
                }))
                .take(self.num_calls);

        if !self.replace_ranges.is_empty() {
            let mut tokens: Vec<_> = tokens.collect();
            let mut replace_ranges = self.replace_ranges.to_vec();
            replace_ranges.sort_by_key(|(range, _)| range.start);

            #[cfg(debug_assertions)]
            {
                for [(range, tokens), (next_range, next_tokens)] in replace_ranges.array_windows() {
                    assert!(
                        range.end <= next_range.start || range.end >= next_range.end,
                        "Replace ranges should either be disjoint or nested: ({:?}, {:?}) ({:?}, {:?})",
                        range,
                        tokens,
                        next_range,
                        next_tokens,
                    );
                }
            }

            // Process the replace ranges, starting from the highest start
            // position and working our way back. If have tokens like:
            //
            // `#[cfg(FALSE)] struct Foo { #[cfg(FALSE)] field: bool }`
            //
            // Then we will generate replace ranges for both
            // the `#[cfg(FALSE)] field: bool` and the entire
            // `#[cfg(FALSE)] struct Foo { #[cfg(FALSE)] field: bool }`
            //
            // By starting processing from the replace range with the greatest
            // start position, we ensure that any replace range which encloses
            // another replace range will capture the *replaced* tokens for the inner
            // range, not the original tokens.
            for (range, new_tokens) in replace_ranges.into_iter().rev() {
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
                    new_tokens.into_iter().chain(filler),
                );
            }
            make_token_stream(tokens.into_iter(), self.break_last_token)
        } else {
            make_token_stream(tokens, self.break_last_token)
        }
    }
}

impl<'a> Parser<'a> {
    /// Records all tokens consumed by the provided callback,
    /// including the current token. These tokens are collected
    /// into a `LazyAttrTokenStream`, and returned along with the result
    /// of the callback.
    ///
    /// Note: If your callback consumes an opening delimiter
    /// (including the case where you call `collect_tokens`
    /// when the current token is an opening delimiter),
    /// you must also consume the corresponding closing delimiter.
    ///
    /// That is, you can consume
    /// `something ([{ }])` or `([{}])`, but not `([{}]`
    ///
    /// This restriction shouldn't be an issue in practice,
    /// since this function is used to record the tokens for
    /// a parsed AST item, which always has matching delimiters.
    pub fn collect_tokens_trailing_token<R: HasAttrs + HasTokens>(
        &mut self,
        attrs: AttrWrapper,
        force_collect: ForceCollect,
        f: impl FnOnce(&mut Self, ast::AttrVec) -> PResult<'a, (R, TrailingToken)>,
    ) -> PResult<'a, R> {
        // We only bail out when nothing could possibly observe the collected tokens:
        // 1. We cannot be force collecting tokens (since force-collecting requires tokens
        //    by definition
        if matches!(force_collect, ForceCollect::No)
            // None of our outer attributes can require tokens (e.g. a proc-macro)
            && attrs.is_complete()
            // If our target supports custom inner attributes, then we cannot bail
            // out early, since we may need to capture tokens for a custom inner attribute
            // invocation.
            && !R::SUPPORTS_CUSTOM_INNER_ATTRS
            // Never bail out early in `capture_cfg` mode, since there might be `#[cfg]`
            // or `#[cfg_attr]` attributes.
            && !self.capture_cfg
        {
            return Ok(f(self, attrs.attrs)?.0);
        }

        let start_token = (self.token.clone(), self.token_spacing);
        let cursor_snapshot = self.token_cursor.clone();

        let has_outer_attrs = !attrs.attrs.is_empty();
        let prev_capturing = std::mem::replace(&mut self.capture_state.capturing, Capturing::Yes);
        let replace_ranges_start = self.capture_state.replace_ranges.len();

        let ret = f(self, attrs.attrs);

        self.capture_state.capturing = prev_capturing;

        let (mut ret, trailing) = ret?;

        // When we're not in `capture-cfg` mode, then bail out early if:
        // 1. Our target doesn't support tokens at all (e.g we're parsing an `NtIdent`)
        //    so there's nothing for us to do.
        // 2. Our target already has tokens set (e.g. we've parsed something
        // like `#[my_attr] $item`. The actual parsing code takes care of prepending
        // any attributes to the nonterminal, so we don't need to modify the
        // already captured tokens.
        // Note that this check is independent of `force_collect`- if we already
        // have tokens, or can't even store them, then there's never a need to
        // force collection of new tokens.
        if !self.capture_cfg && matches!(ret.tokens_mut(), None | Some(Some(_))) {
            return Ok(ret);
        }

        // This is very similar to the bail out check at the start of this function.
        // Now that we've parsed an AST node, we have more information available.
        if matches!(force_collect, ForceCollect::No)
            // We now have inner attributes available, so this check is more precise
            // than `attrs.is_complete()` at the start of the function.
            // As a result, we don't need to check `R::SUPPORTS_CUSTOM_INNER_ATTRS`
            && crate::parser::attr::is_complete(ret.attrs())
            // Subtle: We call `has_cfg_or_cfg_attr` with the attrs from `ret`.
            // This ensures that we consider inner attributes (e.g. `#![cfg]`),
            // which require us to have tokens available
            // We also call `has_cfg_or_cfg_attr` at the beginning of this function,
            // but we only bail out if there's no possibility of inner attributes
            // (!R::SUPPORTS_CUSTOM_INNER_ATTRS)
            // We only capture about `#[cfg]` or `#[cfg_attr]` in `capture_cfg`
            // mode - during normal parsing, we don't need any special capturing
            // for those attributes, since they're builtin.
            && !(self.capture_cfg && has_cfg_or_cfg_attr(ret.attrs()))
        {
            return Ok(ret);
        }

        let mut inner_attr_replace_ranges = Vec::new();
        // Take the captured ranges for any inner attributes that we parsed.
        for inner_attr in ret.attrs().iter().filter(|a| a.style == ast::AttrStyle::Inner) {
            if let Some(attr_range) = self.capture_state.inner_attr_ranges.remove(&inner_attr.id) {
                inner_attr_replace_ranges.push(attr_range);
            } else {
                self.sess
                    .span_diagnostic
                    .delay_span_bug(inner_attr.span, "Missing token range for attribute");
            }
        }

        let replace_ranges_end = self.capture_state.replace_ranges.len();

        let cursor_snapshot_next_calls = cursor_snapshot.num_next_calls;
        let mut end_pos = self.token_cursor.num_next_calls;

        let mut captured_trailing = false;

        // Capture a trailing token if requested by the callback 'f'
        match trailing {
            TrailingToken::None => {}
            TrailingToken::Gt => {
                assert_eq!(self.token.kind, token::Gt);
            }
            TrailingToken::Semi => {
                assert_eq!(self.token.kind, token::Semi);
                end_pos += 1;
                captured_trailing = true;
            }
            TrailingToken::MaybeComma => {
                if self.token.kind == token::Comma {
                    end_pos += 1;
                    captured_trailing = true;
                }
            }
        }

        // If we 'broke' the last token (e.g. breaking a '>>' token to two '>' tokens),
        // then extend the range of captured tokens to include it, since the parser
        // was not actually bumped past it. When the `LazyAttrTokenStream` gets converted
        // into an `AttrTokenStream`, we will create the proper token.
        if self.token_cursor.break_last_token {
            assert!(!captured_trailing, "Cannot set break_last_token and have trailing token");
            end_pos += 1;
        }

        let num_calls = end_pos - cursor_snapshot_next_calls;

        // If we have no attributes, then we will never need to
        // use any replace ranges.
        let replace_ranges: Box<[ReplaceRange]> = if ret.attrs().is_empty() && !self.capture_cfg {
            Box::new([])
        } else {
            // Grab any replace ranges that occur *inside* the current AST node.
            // We will perform the actual replacement when we convert the `LazyAttrTokenStream`
            // to an `AttrTokenStream`.
            let start_calls: u32 = cursor_snapshot_next_calls.try_into().unwrap();
            self.capture_state.replace_ranges[replace_ranges_start..replace_ranges_end]
                .iter()
                .cloned()
                .chain(inner_attr_replace_ranges.iter().cloned())
                .map(|(range, tokens)| {
                    ((range.start - start_calls)..(range.end - start_calls), tokens)
                })
                .collect()
        };

        let tokens = LazyAttrTokenStream::new(LazyAttrTokenStreamImpl {
            start_token,
            num_calls,
            cursor_snapshot,
            break_last_token: self.token_cursor.break_last_token,
            replace_ranges,
        });

        // If we support tokens at all
        if let Some(target_tokens) = ret.tokens_mut() {
            if target_tokens.is_none() {
                // Store se our newly captured tokens into the AST node
                *target_tokens = Some(tokens.clone());
            }
        }

        let final_attrs = ret.attrs();

        // If `capture_cfg` is set and we're inside a recursive call to
        // `collect_tokens_trailing_token`, then we need to register a replace range
        // if we have `#[cfg]` or `#[cfg_attr]`. This allows us to run eager cfg-expansion
        // on the captured token stream.
        if self.capture_cfg
            && matches!(self.capture_state.capturing, Capturing::Yes)
            && has_cfg_or_cfg_attr(final_attrs)
        {
            let attr_data = AttributesData { attrs: final_attrs.iter().cloned().collect(), tokens };

            // Replace the entire AST node that we just parsed, including attributes,
            // with a `FlatToken::AttrTarget`. If this AST node is inside an item
            // that has `#[derive]`, then this will allow us to cfg-expand this
            // AST node.
            let start_pos =
                if has_outer_attrs { attrs.start_pos } else { cursor_snapshot_next_calls };
            let new_tokens = vec![(FlatToken::AttrTarget(attr_data), Spacing::Alone)];

            assert!(
                !self.token_cursor.break_last_token,
                "Should not have unglued last token with cfg attr"
            );
            let range: Range<u32> = (start_pos.try_into().unwrap())..(end_pos.try_into().unwrap());
            self.capture_state.replace_ranges.push((range, new_tokens));
            self.capture_state.replace_ranges.extend(inner_attr_replace_ranges);
        }

        // Only clear our `replace_ranges` when we're finished capturing entirely.
        if matches!(self.capture_state.capturing, Capturing::No) {
            self.capture_state.replace_ranges.clear();
            // We don't clear `inner_attr_ranges`, as doing so repeatedly
            // had a measurable performance impact. Most inner attributes that
            // we insert will get removed - when we drop the parser, we'll free
            // up the memory used by any attributes that we didn't remove from the map.
        }
        Ok(ret)
    }
}

/// Converts a flattened iterator of tokens (including open and close delimiter tokens)
/// into a `TokenStream`, creating a `TokenTree::Delimited` for each matching pair
/// of open and close delims.
fn make_token_stream(
    mut iter: impl Iterator<Item = (FlatToken, Spacing)>,
    break_last_token: bool,
) -> AttrTokenStream {
    #[derive(Debug)]
    struct FrameData {
        // This is `None` for the first frame, `Some` for all others.
        open_delim_sp: Option<(Delimiter, Span)>,
        inner: Vec<AttrTokenTree>,
    }
    let mut stack = vec![FrameData { open_delim_sp: None, inner: vec![] }];
    let mut token_and_spacing = iter.next();
    while let Some((token, spacing)) = token_and_spacing {
        match token {
            FlatToken::Token(Token { kind: TokenKind::OpenDelim(delim), span }) => {
                stack.push(FrameData { open_delim_sp: Some((delim, span)), inner: vec![] });
            }
            FlatToken::Token(Token { kind: TokenKind::CloseDelim(delim), span }) => {
                let frame_data = stack
                    .pop()
                    .unwrap_or_else(|| panic!("Token stack was empty for token: {:?}", token));

                let (open_delim, open_sp) = frame_data.open_delim_sp.unwrap();
                assert_eq!(
                    open_delim, delim,
                    "Mismatched open/close delims: open={:?} close={:?}",
                    open_delim, span
                );
                let dspan = DelimSpan::from_pair(open_sp, span);
                let stream = AttrTokenStream::new(frame_data.inner);
                let delimited = AttrTokenTree::Delimited(dspan, delim, stream);
                stack
                    .last_mut()
                    .unwrap_or_else(|| {
                        panic!("Bottom token frame is missing for token: {:?}", token)
                    })
                    .inner
                    .push(delimited);
            }
            FlatToken::Token(token) => stack
                .last_mut()
                .expect("Bottom token frame is missing!")
                .inner
                .push(AttrTokenTree::Token(token, spacing)),
            FlatToken::AttrTarget(data) => stack
                .last_mut()
                .expect("Bottom token frame is missing!")
                .inner
                .push(AttrTokenTree::Attributes(data)),
            FlatToken::Empty => {}
        }
        token_and_spacing = iter.next();
    }
    let mut final_buf = stack.pop().expect("Missing final buf!");
    if break_last_token {
        let last_token = final_buf.inner.pop().unwrap();
        if let AttrTokenTree::Token(last_token, spacing) = last_token {
            let unglued_first = last_token.kind.break_two_token_op().unwrap().0;

            // An 'unglued' token is always two ASCII characters
            let mut first_span = last_token.span.shrink_to_lo();
            first_span = first_span.with_hi(first_span.lo() + rustc_span::BytePos(1));

            final_buf
                .inner
                .push(AttrTokenTree::Token(Token::new(unglued_first, first_span), spacing));
        } else {
            panic!("Unexpected last token {:?}", last_token)
        }
    }
    AttrTokenStream::new(final_buf.inner)
}

// Some types are used a lot. Make sure they don't unintentionally get bigger.
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
mod size_asserts {
    use super::*;
    use rustc_data_structures::static_assert_size;
    // tidy-alphabetical-start
    static_assert_size!(AttrWrapper, 16);
    static_assert_size!(LazyAttrTokenStreamImpl, 120);
    // tidy-alphabetical-end
}
