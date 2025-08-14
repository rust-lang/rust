use std::borrow::Cow;
use std::mem;

use rustc_ast::token::Token;
use rustc_ast::tokenstream::{
    AttrsTarget, LazyAttrTokenStream, NodeRange, ParserRange, Spacing, TokenCursor,
};
use rustc_ast::{self as ast, AttrVec, Attribute, HasAttrs, HasTokens};
use rustc_data_structures::fx::FxHashSet;
use rustc_errors::PResult;
use rustc_session::parse::ParseSess;
use rustc_span::{DUMMY_SP, sym};
use thin_vec::ThinVec;

use super::{Capturing, ForceCollect, Parser, Trailing};

// When collecting tokens, this fully captures the start point. Usually its
// just after outer attributes, but occasionally it's before.
#[derive(Clone, Debug)]
pub(super) struct CollectPos {
    start_token: (Token, Spacing),
    cursor_snapshot: TokenCursor,
    start_pos: u32,
}

pub(super) enum UsePreAttrPos {
    No,
    Yes,
}

/// A wrapper type to ensure that the parser handles outer attributes correctly.
/// When we parse outer attributes, we need to ensure that we capture tokens
/// for the attribute target. This allows us to perform cfg-expansion on
/// a token stream before we invoke a derive proc-macro.
///
/// This wrapper prevents direct access to the underlying `ast::AttrVec`.
/// Parsing code can only get access to the underlying attributes
/// by passing an `AttrWrapper` to `collect_tokens`.
/// This makes it difficult to accidentally construct an AST node
/// (which stores an `ast::AttrVec`) without first collecting tokens.
///
/// This struct has its own module, to ensure that the parser code
/// cannot directly access the `attrs` field.
#[derive(Debug, Clone)]
pub(super) struct AttrWrapper {
    attrs: AttrVec,
    // The start of the outer attributes in the parser's token stream.
    // This lets us create a `NodeReplacement` for the entire attribute
    // target, including outer attributes. `None` if there are no outer
    // attributes.
    start_pos: Option<u32>,
}

impl AttrWrapper {
    pub(super) fn new(attrs: AttrVec, start_pos: u32) -> AttrWrapper {
        AttrWrapper { attrs, start_pos: Some(start_pos) }
    }

    pub(super) fn empty() -> AttrWrapper {
        AttrWrapper { attrs: AttrVec::new(), start_pos: None }
    }

    pub(super) fn take_for_recovery(self, psess: &ParseSess) -> AttrVec {
        psess.dcx().span_delayed_bug(
            self.attrs.get(0).map(|attr| attr.span).unwrap_or(DUMMY_SP),
            "AttrVec is taken for recovery but no error is produced",
        );

        self.attrs
    }

    /// Prepend `self.attrs` to `attrs`.
    // FIXME: require passing an NT to prevent misuse of this method
    pub(super) fn prepend_to_nt_inner(mut self, attrs: &mut AttrVec) {
        mem::swap(attrs, &mut self.attrs);
        attrs.extend(self.attrs);
    }

    pub(super) fn is_empty(&self) -> bool {
        self.attrs.is_empty()
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

impl<'a> Parser<'a> {
    pub(super) fn collect_pos(&self) -> CollectPos {
        CollectPos {
            start_token: (self.token, self.token_spacing),
            cursor_snapshot: self.token_cursor.clone(),
            start_pos: self.num_bump_calls,
        }
    }

    /// Parses code with `f`. If appropriate, it records the tokens (in
    /// `LazyAttrTokenStream` form) that were parsed in the result, accessible
    /// via the `HasTokens` trait. The `Trailing` part of the callback's
    /// result indicates if an extra token should be captured, e.g. a comma or
    /// semicolon. The `UsePreAttrPos` part of the callback's result indicates
    /// if we should use `pre_attr_pos` as the collection start position (only
    /// required in a few cases).
    ///
    /// The `attrs` passed in are in `AttrWrapper` form, which is opaque. The
    /// `AttrVec` within is passed to `f`. See the comment on `AttrWrapper` for
    /// details.
    ///
    /// `pre_attr_pos` is the position before the outer attributes (or the node
    /// itself, if no outer attributes are present). It is only needed if `f`
    /// can return `UsePreAttrPos::Yes`.
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
    pub(super) fn collect_tokens<R: HasAttrs + HasTokens>(
        &mut self,
        pre_attr_pos: Option<CollectPos>,
        attrs: AttrWrapper,
        force_collect: ForceCollect,
        f: impl FnOnce(&mut Self, AttrVec) -> PResult<'a, (R, Trailing, UsePreAttrPos)>,
    ) -> PResult<'a, R> {
        let possible_capture_mode = self.capture_cfg;

        // We must collect if anything could observe the collected tokens, i.e.
        // if any of the following conditions hold.
        // - We are force collecting tokens (because force collection requires
        //   tokens by definition).
        let needs_collection = matches!(force_collect, ForceCollect::Yes)
            // - Any of our outer attributes require tokens.
            || needs_tokens(&attrs.attrs)
            // - Our target supports custom inner attributes (custom
            //   inner attribute invocation might require token capturing).
            || R::SUPPORTS_CUSTOM_INNER_ATTRS
            // - We are in "possible capture mode" (which requires tokens if
            //   the parsed node has `#[cfg]` or `#[cfg_attr]` attributes).
            || possible_capture_mode;
        if !needs_collection {
            return Ok(f(self, attrs.attrs)?.0);
        }

        let mut collect_pos = self.collect_pos();
        let has_outer_attrs = !attrs.attrs.is_empty();
        let parser_replacements_start = self.capture_state.parser_replacements.len();

        // We set and restore `Capturing::Yes` on either side of the call to
        // `f`, so we can distinguish the outermost call to `collect_tokens`
        // (e.g. parsing `m` in the example above) from any inner (indirectly
        // recursive) calls (e.g. parsing `g` in the example above). This
        // distinction is used below and in `Parser::parse_inner_attributes`.
        let (mut ret, capture_trailing, use_pre_attr_pos) = {
            let prev_capturing = mem::replace(&mut self.capture_state.capturing, Capturing::Yes);
            let res = f(self, attrs.attrs);
            self.capture_state.capturing = prev_capturing;
            res?
        };

        // - `None`: Our target doesn't support tokens at all (e.g. `NtIdent`).
        // - `Some(None)`: Our target supports tokens and has none.
        // - `Some(Some(_))`: Our target already has tokens set (e.g. we've
        //   parsed something like `#[my_attr] $item`).
        let ret_can_hold_tokens = matches!(ret.tokens_mut(), Some(None));

        // Ignore any attributes we've previously processed. This happens when
        // an inner call to `collect_tokens` returns an AST node and then an
        // outer call ends up with the same AST node without any additional
        // wrapping layer.
        let mut seen_indices = FxHashSet::default();
        for (i, attr) in ret.attrs().iter().enumerate() {
            let is_unseen = self.capture_state.seen_attrs.insert(attr.id);
            if !is_unseen {
                seen_indices.insert(i);
            }
        }
        let ret_attrs: Cow<'_, [Attribute]> =
            if seen_indices.is_empty() {
                Cow::Borrowed(ret.attrs())
            } else {
                let ret_attrs =
                    ret.attrs()
                        .iter()
                        .enumerate()
                        .filter_map(|(i, attr)| {
                            if seen_indices.contains(&i) { None } else { Some(attr.clone()) }
                        })
                        .collect();
                Cow::Owned(ret_attrs)
            };

        // When we're not in "definite capture mode", then skip collecting and
        // return early if `ret` doesn't support tokens or already has some.
        //
        // Note that this check is independent of `force_collect`. There's no
        // need to collect tokens when we don't support tokens or already have
        // tokens.
        let definite_capture_mode = self.capture_cfg
            && matches!(self.capture_state.capturing, Capturing::Yes)
            && has_cfg_or_cfg_attr(&ret_attrs);
        if !definite_capture_mode && !ret_can_hold_tokens {
            return Ok(ret);
        }

        // This is similar to the `needs_collection` check at the start of this
        // function, but now that we've parsed an AST node we have complete
        // information available. (If we return early here that means the
        // setup, such as cloning the token cursor, was unnecessary. That's
        // hard to avoid.)
        //
        // We must collect if anything could observe the collected tokens, i.e.
        // if any of the following conditions hold.
        // - We are force collecting tokens.
        let needs_collection = matches!(force_collect, ForceCollect::Yes)
            // - Any of our outer *or* inner attributes require tokens.
            //   (`attr.attrs` was just outer attributes, but `ret.attrs()` is
            //   outer and inner attributes. So this check is more precise than
            //   the earlier `needs_tokens` check, and we don't need to
            //   check `R::SUPPORTS_CUSTOM_INNER_ATTRS`.)
            || needs_tokens(&ret_attrs)
            // - We are in "definite capture mode", which requires that there
            //   are `#[cfg]` or `#[cfg_attr]` attributes. (During normal
            //   non-`capture_cfg` parsing, we don't need any special capturing
            //   for those attributes, because they're builtin.)
            || definite_capture_mode;
        if !needs_collection {
            return Ok(ret);
        }

        // Replace the post-attribute collection start position with the
        // pre-attribute position supplied, if `f` indicated it is necessary.
        // (The caller is responsible for providing a non-`None` `pre_attr_pos`
        // if this is a possibility.)
        if matches!(use_pre_attr_pos, UsePreAttrPos::Yes) {
            collect_pos = pre_attr_pos.unwrap();
        }

        let parser_replacements_end = self.capture_state.parser_replacements.len();

        assert!(
            !(self.break_last_token > 0 && matches!(capture_trailing, Trailing::Yes)),
            "Cannot have break_last_token > 0 and have trailing token"
        );
        assert!(self.break_last_token <= 2, "cannot break token more than twice");

        let end_pos = self.num_bump_calls
            + capture_trailing as u32
            // If we "broke" the last token (e.g. breaking a `>>` token once into `>` + `>`, or
            // breaking a `>>=` token twice into `>` + `>` + `=`), then extend the range of
            // captured tokens to include it, because the parser was not actually bumped past it.
            // (Even if we broke twice, it was still just one token originally, hence the `1`.)
            // When the `LazyAttrTokenStream` gets converted into an `AttrTokenStream`, we will
            // rebreak that final token once or twice.
            + if self.break_last_token == 0 { 0 } else { 1 };

        let num_calls = end_pos - collect_pos.start_pos;

        // Take the captured `ParserRange`s for any inner attributes that we parsed in
        // `Parser::parse_inner_attributes`, and pair them in a `ParserReplacement` with `None`,
        // which means the relevant tokens will be removed. (More details below.)
        let mut inner_attr_parser_replacements = Vec::new();
        for attr in ret_attrs.iter() {
            if attr.style == ast::AttrStyle::Inner {
                if let Some(inner_attr_parser_range) =
                    self.capture_state.inner_attr_parser_ranges.remove(&attr.id)
                {
                    inner_attr_parser_replacements.push((inner_attr_parser_range, None));
                } else {
                    self.dcx().span_delayed_bug(attr.span, "Missing token range for attribute");
                }
            }
        }

        // This is hot enough for `deep-vector` that checking the conditions for an empty iterator
        // is measurably faster than actually executing the iterator.
        let node_replacements = if parser_replacements_start == parser_replacements_end
            && inner_attr_parser_replacements.is_empty()
        {
            ThinVec::new()
        } else {
            // Grab any replace ranges that occur *inside* the current AST node. Convert them
            // from `ParserRange` form to `NodeRange` form. We will perform the actual
            // replacement only when we convert the `LazyAttrTokenStream` to an
            // `AttrTokenStream`.
            self.capture_state.parser_replacements
                [parser_replacements_start..parser_replacements_end]
                .iter()
                .cloned()
                .chain(inner_attr_parser_replacements)
                .map(|(parser_range, data)| {
                    (NodeRange::new(parser_range, collect_pos.start_pos), data)
                })
                .collect()
        };

        // What is the status here when parsing the example code at the top of this method?
        //
        // When parsing `g`:
        // - `start_pos..end_pos` is `12..33` (`fn g { ... }`, excluding the outer attr).
        // - `inner_attr_parser_replacements` has one entry (`ParserRange(17..27)`), to
        //   delete the inner attr's tokens.
        //   - This entry is converted to `NodeRange(5..15)` (relative to the `fn`) and put into
        //     the lazy tokens for `g`, i.e. deleting the inner attr from those tokens (if they get
        //     evaluated).
        //   - Those lazy tokens are also put into an `AttrsTarget` that is appended to `self`'s
        //     replace ranges at the bottom of this function, for processing when parsing `m`.
        // - `parser_replacements_start..parser_replacements_end` is empty.
        //
        // When parsing `m`:
        // - `start_pos..end_pos` is `0..34` (`mod m`, excluding the `#[cfg_eval]` attribute).
        // - `inner_attr_parser_replacements` is empty.
        // - `parser_replacements_start..parser_replacements_end` has one entry.
        //   - One `AttrsTarget` (added below when parsing `g`) to replace all of `g` (`3..33`,
        //     including its outer attribute), with:
        //     - `attrs`: includes the outer and the inner attr.
        //     - `tokens`: lazy tokens for `g` (with its inner attr deleted).

        let tokens = LazyAttrTokenStream::new_pending(
            collect_pos.start_token,
            collect_pos.cursor_snapshot,
            num_calls,
            self.break_last_token,
            node_replacements,
        );
        let mut tokens_used = false;

        // If in "definite capture mode" we need to register a replace range
        // for the `#[cfg]` and/or `#[cfg_attr]` attrs. This allows us to run
        // eager cfg-expansion on the captured token stream.
        if definite_capture_mode {
            assert!(self.break_last_token == 0, "Should not have unglued last token with cfg attr");

            // What is the status here when parsing the example code at the top of this method?
            //
            // When parsing `g`, we add one entry:
            // - The pushed entry (`ParserRange(3..33)`) has a new `AttrsTarget` with:
            //   - `attrs`: includes the outer and the inner attr.
            //   - `tokens`: lazy tokens for `g` (with its inner attr deleted).
            //
            // When parsing `m`, we do nothing here.

            // Set things up so that the entire AST node that we just parsed, including attributes,
            // will be replaced with `target` in the lazy token stream. This will allow us to
            // cfg-expand this AST node.
            let start_pos =
                if has_outer_attrs { attrs.start_pos.unwrap() } else { collect_pos.start_pos };
            let target =
                AttrsTarget { attrs: ret_attrs.iter().cloned().collect(), tokens: tokens.clone() };
            tokens_used = true;
            self.capture_state
                .parser_replacements
                .push((ParserRange(start_pos..end_pos), Some(target)));
        } else if matches!(self.capture_state.capturing, Capturing::No) {
            // Only clear the ranges once we've finished capturing entirely, i.e. we've finished
            // the outermost call to this method.
            self.capture_state.parser_replacements.clear();
            self.capture_state.inner_attr_parser_ranges.clear();
            self.capture_state.seen_attrs.clear();
        }

        // If we support tokens and don't already have them, store the newly captured tokens.
        if let Some(target_tokens @ None) = ret.tokens_mut() {
            tokens_used = true;
            *target_tokens = Some(tokens);
        }

        assert!(tokens_used); // check we didn't create `tokens` unnecessarily
        Ok(ret)
    }
}

/// Tokens are needed if:
/// - any non-single-segment attributes (other than doc comments) are present,
///   e.g. `rustfmt::skip`; or
/// - any `cfg_attr` attributes are present; or
/// - any single-segment, non-builtin attributes are present, e.g. `derive`,
///   `test`, `global_allocator`.
fn needs_tokens(attrs: &[ast::Attribute]) -> bool {
    attrs.iter().any(|attr| match attr.ident() {
        None => !attr.is_doc_comment(),
        Some(ident) => {
            ident.name == sym::cfg_attr || !rustc_feature::is_builtin_attr_name(ident.name)
        }
    })
}
