use super::{Capturing, FlatToken, ForceCollect, Parser, ReplaceRange, TokenCursor};
use rustc_ast::token::{Delimiter, Token, TokenKind};
use rustc_ast::tokenstream::{AttrTokenStream, AttrTokenTree, AttrsTarget, DelimSpacing};
use rustc_ast::tokenstream::{DelimSpan, LazyAttrTokenStream, Spacing, ToAttrTokenStream};
use rustc_ast::{self as ast};
use rustc_ast::{AttrVec, Attribute, HasAttrs, HasTokens};
use rustc_errors::PResult;
use rustc_session::parse::ParseSess;
use rustc_span::{sym, Span, DUMMY_SP};

use std::{iter, mem};

/// A wrapper type to ensure that the parser handles outer attributes correctly.
/// When we parse outer attributes, we need to ensure that we capture tokens
/// for the attribute target. This allows us to perform cfg-expansion on
/// a token stream before we invoke a derive proc-macro.
///
/// This wrapper prevents direct access to the underlying `ast::AttrVec`.
/// Parsing code can only get access to the underlying attributes
/// by passing an `AttrWrapper` to `collect_tokens_trailing_token`.
/// This makes it difficult to accidentally construct an AST node
/// (which stores an `ast::AttrVec`) without first collecting tokens.
///
/// This struct has its own module, to ensure that the parser code
/// cannot directly access the `attrs` field.
#[derive(Debug, Clone)]
pub struct AttrWrapper {
    attrs: AttrVec,
    // The start of the outer attributes in the token cursor.
    // This allows us to create a `ReplaceRange` for the entire attribute
    // target, including outer attributes.
    start_pos: u32,
}

impl AttrWrapper {
    pub(super) fn new(attrs: AttrVec, start_pos: u32) -> AttrWrapper {
        AttrWrapper { attrs, start_pos }
    }
    pub fn empty() -> AttrWrapper {
        AttrWrapper { attrs: AttrVec::new(), start_pos: u32::MAX }
    }

    pub(crate) fn take_for_recovery(self, psess: &ParseSess) -> AttrVec {
        psess.dcx().span_delayed_bug(
            self.attrs.get(0).map(|attr| attr.span).unwrap_or(DUMMY_SP),
            "AttrVec is taken for recovery but no error is produced",
        );

        self.attrs
    }

    /// Prepend `self.attrs` to `attrs`.
    // FIXME: require passing an NT to prevent misuse of this method
    pub(crate) fn prepend_to_nt_inner(self, attrs: &mut AttrVec) {
        let mut self_attrs = self.attrs;
        mem::swap(attrs, &mut self_attrs);
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

// From a value of this type we can reconstruct the `TokenStream` seen by the
// `f` callback passed to a call to `Parser::collect_tokens_trailing_token`, by
// replaying the getting of the tokens. This saves us producing a `TokenStream`
// if it is never needed, e.g. a captured `macro_rules!` argument that is never
// passed to a proc macro. In practice, token stream creation happens rarely
// compared to calls to `collect_tokens` (see some statistics in #78736) so we
// are doing as little up-front work as possible.
//
// This also makes `Parser` very cheap to clone, since
// there is no intermediate collection buffer to clone.
struct LazyAttrTokenStreamImpl {
    start_token: (Token, Spacing),
    cursor_snapshot: TokenCursor,
    num_calls: u32,
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
        let tokens = iter::once(FlatToken::Token(self.start_token.clone()))
            .chain(iter::repeat_with(|| FlatToken::Token(cursor_snapshot.next())))
            .take(self.num_calls as usize);

        if self.replace_ranges.is_empty() {
            make_attr_token_stream(tokens, self.break_last_token)
        } else {
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
            for (range, target) in replace_ranges.into_iter().rev() {
                assert!(!range.is_empty(), "Cannot replace an empty range: {range:?}");

                // Replace the tokens in range with zero or one `FlatToken::AttrsTarget`s, plus
                // enough `FlatToken::Empty`s to fill up the rest of the range. This keeps the
                // total length of `tokens` constant throughout the replacement process, allowing
                // us to use all of the `ReplaceRanges` entries without adjusting indices.
                let target_len = target.is_some() as usize;
                tokens.splice(
                    (range.start as usize)..(range.end as usize),
                    target
                        .into_iter()
                        .map(|target| FlatToken::AttrsTarget(target))
                        .chain(iter::repeat(FlatToken::Empty).take(range.len() - target_len)),
                );
            }
            make_attr_token_stream(tokens.into_iter(), self.break_last_token)
        }
    }
}

impl<'a> Parser<'a> {
    /// Parses code with `f`. If appropriate, it records the tokens (in
    /// `LazyAttrTokenStream` form) that were parsed in the result, accessible
    /// via the `HasTokens` trait. The second (bool) part of the callback's
    /// result indicates if an extra token should be captured, e.g. a comma or
    /// semicolon.
    ///
    /// The `attrs` passed in are in `AttrWrapper` form, which is opaque. The
    /// `AttrVec` within is passed to `f`. See the comment on `AttrWrapper` for
    /// details.
    ///
    /// Note: If your callback consumes an opening delimiter (including the
    /// case where `self.token` is an opening delimiter on entry to this
    /// function), you must also consume the corresponding closing delimiter.
    /// E.g. you can consume `something ([{ }])` or `([{}])`, but not `([{}]`.
    /// This restriction isn't a problem in practice, because parsed AST items
    /// always have matching delimiters.
    ///
    /// The following example code will be used to explain things in comments
    /// below. It has an outer attribute and an inner attribute. Parsing it
    /// involves two calls to this method, one of which is indirectly
    /// recursive.
    /// ```ignore (fake attributes)
    /// #[cfg_eval]                         // token pos
    /// mod m {                             //   0.. 3
    ///     #[cfg_attr(cond1, attr1)]       //   3..12
    ///     fn g() {                        //  12..17
    ///         #![cfg_attr(cond2, attr2)]  //  17..27
    ///         let _x = 3;                 //  27..32
    ///     }                               //  32..33
    /// }                                   //  33..34
    /// ```
    pub fn collect_tokens_trailing_token<R: HasAttrs + HasTokens>(
        &mut self,
        attrs: AttrWrapper,
        force_collect: ForceCollect,
        f: impl FnOnce(&mut Self, ast::AttrVec) -> PResult<'a, (R, bool)>,
    ) -> PResult<'a, R> {
        // Skip collection when nothing could observe the collected tokens, i.e.
        // all of the following conditions hold.
        // - We are not force collecting tokens (because force collection
        //   requires tokens by definition).
        if matches!(force_collect, ForceCollect::No)
            // - None of our outer attributes require tokens.
            && attrs.is_complete()
            // - Our target doesn't support custom inner attributes (custom
            //   inner attribute invocation might require token capturing).
            && !R::SUPPORTS_CUSTOM_INNER_ATTRS
            // - We are not in `capture_cfg` mode (which requires tokens if
            //   the parsed node has `#[cfg]` or `#[cfg_attr]` attributes).
            && !self.capture_cfg
        {
            return Ok(f(self, attrs.attrs)?.0);
        }

        let start_token = (self.token.clone(), self.token_spacing);
        let cursor_snapshot = self.token_cursor.clone();
        let start_pos = self.num_bump_calls;
        let has_outer_attrs = !attrs.attrs.is_empty();
        let replace_ranges_start = self.capture_state.replace_ranges.len();

        // We set and restore `Capturing::Yes` on either side of the call to
        // `f`, so we can distinguish the outermost call to
        // `collect_tokens_trailing_token` (e.g. parsing `m` in the example
        // above) from any inner (indirectly recursive) calls (e.g. parsing `g`
        // in the example above). This distinction is used below and in
        // `Parser::parse_inner_attributes`.
        let (mut ret, capture_trailing) = {
            let prev_capturing = mem::replace(&mut self.capture_state.capturing, Capturing::Yes);
            let ret_and_trailing = f(self, attrs.attrs);
            self.capture_state.capturing = prev_capturing;
            ret_and_trailing?
        };

        // When we're not in `capture_cfg` mode, then skip collecting and
        // return early if either of the following conditions hold.
        // - `None`: Our target doesn't support tokens at all (e.g. `NtIdent`).
        // - `Some(Some(_))`: Our target already has tokens set (e.g. we've
        //   parsed something like `#[my_attr] $item`). The actual parsing code
        //   takes care of prepending any attributes to the nonterminal, so we
        //   don't need to modify the already captured tokens.
        //
        // Note that this check is independent of `force_collect`. There's no
        // need to collect tokens when we don't support tokens or already have
        // tokens.
        if !self.capture_cfg && matches!(ret.tokens_mut(), None | Some(Some(_))) {
            return Ok(ret);
        }

        // This is similar to the "skip collection" check at the start of this
        // function, but now that we've parsed an AST node we have more
        // information available. (If we return early here that means the
        // setup, such as cloning the token cursor, was unnecessary. That's
        // hard to avoid.)
        //
        // Skip collection when nothing could observe the collected tokens, i.e.
        // all of the following conditions hold.
        // - We are not force collecting tokens.
        if matches!(force_collect, ForceCollect::No)
            // - None of our outer *or* inner attributes require tokens.
            //   (`attrs` was just outer attributes, but `ret.attrs()` is outer
            //   and inner attributes. That makes this check more precise than
            //   `attrs.is_complete()` at the start of the function, and we can
            //   skip the subsequent check on `R::SUPPORTS_CUSTOM_INNER_ATTRS`.
            && crate::parser::attr::is_complete(ret.attrs())
            // - We are not in `capture_cfg` mode, or we are but there are no
            //   `#[cfg]` or `#[cfg_attr]` attributes. (During normal
            //   non-`capture_cfg` parsing, we don't need any special capturing
            //   for those attributes, because they're builtin.)
            && (!self.capture_cfg || !has_cfg_or_cfg_attr(ret.attrs()))
        {
            return Ok(ret);
        }

        let replace_ranges_end = self.capture_state.replace_ranges.len();

        assert!(
            !(self.break_last_token && capture_trailing),
            "Cannot set break_last_token and have trailing token"
        );

        let end_pos = self.num_bump_calls
            + capture_trailing as u32
            // If we 'broke' the last token (e.g. breaking a '>>' token to two '>' tokens), then
            // extend the range of captured tokens to include it, since the parser was not actually
            // bumped past it. When the `LazyAttrTokenStream` gets converted into an
            // `AttrTokenStream`, we will create the proper token.
            + self.break_last_token as u32;

        let num_calls = end_pos - start_pos;

        // Take the captured ranges for any inner attributes that we parsed in
        // `Parser::parse_inner_attributes`, and pair them in a `ReplaceRange`
        // with `None`, which means the relevant tokens will be removed. (More
        // details below.)
        let mut inner_attr_replace_ranges = Vec::new();
        for inner_attr in ret.attrs().iter().filter(|a| a.style == ast::AttrStyle::Inner) {
            if let Some(attr_range) = self.capture_state.inner_attr_ranges.remove(&inner_attr.id) {
                inner_attr_replace_ranges.push((attr_range, None));
            } else {
                self.dcx().span_delayed_bug(inner_attr.span, "Missing token range for attribute");
            }
        }

        // This is hot enough for `deep-vector` that checking the conditions for an empty iterator
        // is measurably faster than actually executing the iterator.
        let replace_ranges: Box<[ReplaceRange]> =
            if replace_ranges_start == replace_ranges_end && inner_attr_replace_ranges.is_empty() {
                Box::new([])
            } else {
                // Grab any replace ranges that occur *inside* the current AST node. We will
                // perform the actual replacement only when we convert the `LazyAttrTokenStream` to
                // an `AttrTokenStream`.
                self.capture_state.replace_ranges[replace_ranges_start..replace_ranges_end]
                    .iter()
                    .cloned()
                    .chain(inner_attr_replace_ranges.iter().cloned())
                    .map(|(range, data)| ((range.start - start_pos)..(range.end - start_pos), data))
                    .collect()
            };

        // What is the status here when parsing the example code at the top of this method?
        //
        // When parsing `g`:
        // - `start_pos..end_pos` is `12..33` (`fn g { ... }`, excluding the outer attr).
        // - `inner_attr_replace_ranges` has one entry (`5..15`, when counting from `fn`), to
        //   delete the inner attr's tokens.
        //   - This entry is put into the lazy tokens for `g`, i.e. deleting the inner attr from
        //     those tokens (if they get evaluated).
        //   - Those lazy tokens are also put into an `AttrsTarget` that is appended to `self`'s
        //     replace ranges at the bottom of this function, for processing when parsing `m`.
        // - `replace_ranges_start..replace_ranges_end` is empty.
        //
        // When parsing `m`:
        // - `start_pos..end_pos` is `0..34` (`mod m`, excluding the `#[cfg_eval]` attribute).
        // - `inner_attr_replace_ranges` is empty.
        // - `replace_range_start..replace_ranges_end` has two entries.
        //   - One to delete the inner attribute (`17..27`), obtained when parsing `g` (see above).
        //   - One `AttrsTarget` (added below when parsing `g`) to replace all of `g` (`3..33`,
        //     including its outer attribute), with:
        //     - `attrs`: includes the outer and the inner attr.
        //     - `tokens`: lazy tokens for `g` (with its inner attr deleted).

        let tokens = LazyAttrTokenStream::new(LazyAttrTokenStreamImpl {
            start_token,
            num_calls,
            cursor_snapshot,
            break_last_token: self.break_last_token,
            replace_ranges,
        });

        // If we support tokens and don't already have them, store the newly captured tokens.
        if let Some(target_tokens @ None) = ret.tokens_mut() {
            *target_tokens = Some(tokens.clone());
        }

        // If `capture_cfg` is set and we're inside a recursive call to
        // `collect_tokens_trailing_token`, then we need to register a replace range
        // if we have `#[cfg]` or `#[cfg_attr]`. This allows us to run eager cfg-expansion
        // on the captured token stream.
        if self.capture_cfg
            && matches!(self.capture_state.capturing, Capturing::Yes)
            && has_cfg_or_cfg_attr(ret.attrs())
        {
            assert!(!self.break_last_token, "Should not have unglued last token with cfg attr");

            // What is the status here when parsing the example code at the top of this method?
            //
            // When parsing `g`, we add two entries:
            // - The `start_pos..end_pos` (`3..33`) entry has a new `AttrsTarget` with:
            //   - `attrs`: includes the outer and the inner attr.
            //   - `tokens`: lazy tokens for `g` (with its inner attr deleted).
            // - `inner_attr_replace_ranges` contains the one entry to delete the inner attr's
            //   tokens (`17..27`).
            //
            // When parsing `m`, we do nothing here.

            // Set things up so that the entire AST node that we just parsed, including attributes,
            // will be replaced with `target` in the lazy token stream. This will allow us to
            // cfg-expand this AST node.
            let start_pos = if has_outer_attrs { attrs.start_pos } else { start_pos };
            let target = AttrsTarget { attrs: ret.attrs().iter().cloned().collect(), tokens };
            self.capture_state.replace_ranges.push((start_pos..end_pos, Some(target)));
            self.capture_state.replace_ranges.extend(inner_attr_replace_ranges);
        } else if matches!(self.capture_state.capturing, Capturing::No) {
            // Only clear the ranges once we've finished capturing entirely, i.e. we've finished
            // the outermost call to this method.
            self.capture_state.replace_ranges.clear();
            self.capture_state.inner_attr_ranges.clear();
        }
        Ok(ret)
    }
}

/// Converts a flattened iterator of tokens (including open and close delimiter tokens) into an
/// `AttrTokenStream`, creating an `AttrTokenTree::Delimited` for each matching pair of open and
/// close delims.
fn make_attr_token_stream(
    iter: impl Iterator<Item = FlatToken>,
    break_last_token: bool,
) -> AttrTokenStream {
    #[derive(Debug)]
    struct FrameData {
        // This is `None` for the first frame, `Some` for all others.
        open_delim_sp: Option<(Delimiter, Span, Spacing)>,
        inner: Vec<AttrTokenTree>,
    }
    // The stack always has at least one element. Storing it separately makes for shorter code.
    let mut stack_top = FrameData { open_delim_sp: None, inner: vec![] };
    let mut stack_rest = vec![];
    for flat_token in iter {
        match flat_token {
            FlatToken::Token((Token { kind: TokenKind::OpenDelim(delim), span }, spacing)) => {
                stack_rest.push(mem::replace(
                    &mut stack_top,
                    FrameData { open_delim_sp: Some((delim, span, spacing)), inner: vec![] },
                ));
            }
            FlatToken::Token((Token { kind: TokenKind::CloseDelim(delim), span }, spacing)) => {
                let frame_data = mem::replace(&mut stack_top, stack_rest.pop().unwrap());
                let (open_delim, open_sp, open_spacing) = frame_data.open_delim_sp.unwrap();
                assert_eq!(
                    open_delim, delim,
                    "Mismatched open/close delims: open={open_delim:?} close={span:?}"
                );
                let dspan = DelimSpan::from_pair(open_sp, span);
                let dspacing = DelimSpacing::new(open_spacing, spacing);
                let stream = AttrTokenStream::new(frame_data.inner);
                let delimited = AttrTokenTree::Delimited(dspan, dspacing, delim, stream);
                stack_top.inner.push(delimited);
            }
            FlatToken::Token((token, spacing)) => {
                stack_top.inner.push(AttrTokenTree::Token(token, spacing))
            }
            FlatToken::AttrsTarget(target) => {
                stack_top.inner.push(AttrTokenTree::AttrsTarget(target))
            }
            FlatToken::Empty => {}
        }
    }

    if break_last_token {
        let last_token = stack_top.inner.pop().unwrap();
        if let AttrTokenTree::Token(last_token, spacing) = last_token {
            let unglued_first = last_token.kind.break_two_token_op().unwrap().0;

            // An 'unglued' token is always two ASCII characters
            let mut first_span = last_token.span.shrink_to_lo();
            first_span = first_span.with_hi(first_span.lo() + rustc_span::BytePos(1));

            stack_top
                .inner
                .push(AttrTokenTree::Token(Token::new(unglued_first, first_span), spacing));
        } else {
            panic!("Unexpected last token {last_token:?}")
        }
    }
    AttrTokenStream::new(stack_top.inner)
}

// Some types are used a lot. Make sure they don't unintentionally get bigger.
#[cfg(target_pointer_width = "64")]
mod size_asserts {
    use super::*;
    use rustc_data_structures::static_assert_size;
    // tidy-alphabetical-start
    static_assert_size!(AttrWrapper, 16);
    static_assert_size!(LazyAttrTokenStreamImpl, 96);
    // tidy-alphabetical-end
}
