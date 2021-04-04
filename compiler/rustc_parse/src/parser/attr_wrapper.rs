use super::attr;
use super::{ForceCollect, Parser, TokenCursor, TrailingToken};
use rustc_ast::token::{self, Token, TokenKind};
use rustc_ast::tokenstream::{CreateTokenStream, TokenStream, TokenTree, TreeAndSpacing};
use rustc_ast::tokenstream::{DelimSpan, LazyTokenStream, Spacing};
use rustc_ast::AstLike;
use rustc_ast::{self as ast};
use rustc_errors::PResult;
use rustc_span::{Span, DUMMY_SP};

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
    attrs: Vec<ast::Attribute>,
}

impl AttrWrapper {
    pub fn empty() -> AttrWrapper {
        AttrWrapper { attrs: vec![] }
    }
    pub fn new(attrs: Vec<ast::Attribute>) -> AttrWrapper {
        AttrWrapper { attrs }
    }
    // FIXME: Delay span bug here?
    pub(crate) fn take_for_recovery(self) -> Vec<ast::Attribute> {
        self.attrs
    }
    pub fn is_empty(&self) -> bool {
        self.attrs.is_empty()
    }
}

impl<'a> Parser<'a> {
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
        if matches!(force_collect, ForceCollect::No) && !attr::maybe_needs_tokens(&attrs.attrs) {
            return Ok(f(self, attrs.attrs)?.0);
        }
        let start_token = (self.token.clone(), self.token_spacing);
        let cursor_snapshot = self.token_cursor.clone();

        let (mut ret, trailing_token) = f(self, attrs.attrs)?;
        let tokens = match ret.tokens_mut() {
            Some(tokens) if tokens.is_none() => tokens,
            _ => return Ok(ret),
        };

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
            num_calls: usize,
            desugar_doc_comments: bool,
            append_unglued_token: Option<TreeAndSpacing>,
        }
        impl CreateTokenStream for LazyTokenStreamImpl {
            fn create_token_stream(&self) -> TokenStream {
                if self.num_calls == 0 {
                    return TokenStream::new(vec![]);
                }

                let mut cursor_snapshot = self.cursor_snapshot.clone();
                // Don't skip `None` delimiters, since we want to pass them to
                // proc macros. Normally, we'll end up capturing `TokenKind::Interpolated`,
                // which gets converted to a `None`-delimited group when we invoke
                // a proc-macro. However, it's possible to already have a `None`-delimited
                // group in the stream (such as when parsing the output of a proc-macro,
                // or in certain unusual cases with cross-crate `macro_rules!` macros).
                cursor_snapshot.skip_none_delims = false;

                // The token produced by the final call to `next` or `next_desugared`
                // was not actually consumed by the callback.
                let num_calls = self.num_calls - 1;
                let mut i = 0;
                let tokens =
                    std::iter::once(self.start_token.clone()).chain(std::iter::from_fn(|| {
                        if i >= num_calls {
                            return None;
                        }

                        let token = if self.desugar_doc_comments {
                            cursor_snapshot.next_desugared()
                        } else {
                            cursor_snapshot.next()
                        };

                        // When the `LazyTokenStreamImpl` was original produced, we did *not*
                        // include `NoDelim` tokens in `num_calls`, since they are normally ignored
                        // by the parser. Therefore, we only increment our counter for other types of tokens.
                        if !matches!(
                            token.0.kind,
                            token::OpenDelim(token::NoDelim) | token::CloseDelim(token::NoDelim)
                        ) {
                            i += 1;
                        }
                        Some(token)
                    }));

                make_token_stream(tokens, self.append_unglued_token.clone())
            }
        }

        let mut num_calls = self.token_cursor.num_next_calls - cursor_snapshot.num_next_calls;
        match trailing_token {
            TrailingToken::None => {}
            TrailingToken::Semi => {
                assert_eq!(self.token.kind, token::Semi);
                num_calls += 1;
            }
            TrailingToken::MaybeComma => {
                if self.token.kind == token::Comma {
                    num_calls += 1;
                }
            }
        }

        *tokens = Some(LazyTokenStream::new(LazyTokenStreamImpl {
            start_token,
            num_calls,
            cursor_snapshot,
            desugar_doc_comments: self.desugar_doc_comments,
            append_unglued_token: self.token_cursor.append_unglued_token.clone(),
        }));

        Ok(ret)
    }
}

/// Converts a flattened iterator of tokens (including open and close delimiter tokens)
/// into a `TokenStream`, creating a `TokenTree::Delimited` for each matching pair
/// of open and close delims.
fn make_token_stream(
    tokens: impl Iterator<Item = (Token, Spacing)>,
    append_unglued_token: Option<TreeAndSpacing>,
) -> TokenStream {
    #[derive(Debug)]
    struct FrameData {
        open: Span,
        inner: Vec<(TokenTree, Spacing)>,
    }
    let mut stack = vec![FrameData { open: DUMMY_SP, inner: vec![] }];
    for (token, spacing) in tokens {
        match token {
            Token { kind: TokenKind::OpenDelim(_), span } => {
                stack.push(FrameData { open: span, inner: vec![] });
            }
            Token { kind: TokenKind::CloseDelim(delim), span } => {
                let frame_data = stack.pop().expect("Token stack was empty!");
                let dspan = DelimSpan::from_pair(frame_data.open, span);
                let stream = TokenStream::new(frame_data.inner);
                let delimited = TokenTree::Delimited(dspan, delim, stream);
                stack
                    .last_mut()
                    .unwrap_or_else(|| panic!("Bottom token frame is missing for tokens!"))
                    .inner
                    .push((delimited, Spacing::Alone));
            }
            token => {
                stack
                    .last_mut()
                    .expect("Bottom token frame is missing!")
                    .inner
                    .push((TokenTree::Token(token), spacing));
            }
        }
    }
    let mut final_buf = stack.pop().expect("Missing final buf!");
    final_buf.inner.extend(append_unglued_token);
    assert!(stack.is_empty(), "Stack should be empty: final_buf={:?} stack={:?}", final_buf, stack);
    TokenStream::new(final_buf.inner)
}
