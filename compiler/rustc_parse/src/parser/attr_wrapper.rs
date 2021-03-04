use super::attr;
use super::{ForceCollect, IncludeNoDelim, Parser, TrailingToken};
use rustc_ast::token::{self, Token};
use rustc_ast::tokenstream::{CreateTokenStream, TokenStream};
use rustc_ast::tokenstream::{LazyTokenStream, Spacing};
use rustc_ast::AstLike;
use rustc_ast::{self as ast};
use rustc_data_structures::sync::Lrc;
use rustc_errors::PResult;

use std::ops::Range;

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

        let start_index = self.token_cursor.index - 1;
        let (mut ret, trailing_token) = f(self, attrs.attrs)?;
        let mut end_index = self.token_cursor.index - 1;

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
            tokens: Lrc<Vec<(Token, Spacing)>>,
            range: Range<usize>,
            append_unglued_token: Option<(Token, Spacing)>,
        }
        impl CreateTokenStream for LazyTokenStreamImpl {
            fn create_token_stream(&self) -> TokenStream {
                super::make_token_stream(
                    &self.tokens[self.range.clone()],
                    IncludeNoDelim::No,
                    self.append_unglued_token.clone(),
                )
            }
        }

        match trailing_token {
            TrailingToken::None => {}
            TrailingToken::Semi => {
                assert_eq!(self.token.kind, token::Semi);
                end_index += 1;
            }
            TrailingToken::MaybeComma => {
                if self.token.kind == token::Comma {
                    end_index += 1;
                }
            }
        }

        let lazy_impl = LazyTokenStreamImpl {
            tokens: self.token_cursor.tokens.clone(),
            range: (start_index..end_index),
            append_unglued_token: self.token_cursor.append_unglued_token.clone(),
        };
        ret.finalize_tokens(LazyTokenStream::new(lazy_impl));
        Ok(ret)
    }
}
