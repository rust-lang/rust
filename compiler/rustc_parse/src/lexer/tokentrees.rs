use super::{StringReader, UnmatchedBrace};
use rustc_ast::token::{self, Delimiter, Token};
use rustc_ast::tokenstream::{DelimSpan, Spacing, TokenStream, TokenTree};
use rustc_ast_pretty::pprust::token_to_string;
use rustc_data_structures::fx::FxHashMap;
use rustc_errors::{PErr, PResult};
use rustc_span::Span;

pub(super) struct TokenTreesReader<'a> {
    string_reader: StringReader<'a>,
    /// The "next" token, which has been obtained from the `StringReader` but
    /// not yet handled by the `TokenTreesReader`.
    token: Token,
    /// Stack of open delimiters and their spans. Used for error message.
    open_braces: Vec<(Delimiter, Span)>,
    unmatched_braces: Vec<UnmatchedBrace>,
    /// The type and spans for all braces
    ///
    /// Used only for error recovery when arriving to EOF with mismatched braces.
    matching_delim_spans: Vec<(Delimiter, Span, Span)>,
    last_unclosed_found_span: Option<Span>,
    /// Collect empty block spans that might have been auto-inserted by editors.
    last_delim_empty_block_spans: FxHashMap<Delimiter, Span>,
    /// Collect the spans of braces (Open, Close). Used only
    /// for detecting if blocks are empty and only braces.
    matching_block_spans: Vec<(Span, Span)>,
}

impl<'a> TokenTreesReader<'a> {
    pub(super) fn parse_token_trees(
        string_reader: StringReader<'a>,
    ) -> (PResult<'a, TokenStream>, Vec<UnmatchedBrace>) {
        let mut tt_reader = TokenTreesReader {
            string_reader,
            token: Token::dummy(),
            open_braces: Vec::new(),
            unmatched_braces: Vec::new(),
            matching_delim_spans: Vec::new(),
            last_unclosed_found_span: None,
            last_delim_empty_block_spans: FxHashMap::default(),
            matching_block_spans: Vec::new(),
        };
        let res = tt_reader.parse_all_token_trees();
        (res, tt_reader.unmatched_braces)
    }

    // Parse a stream of tokens into a list of `TokenTree`s, up to an `Eof`.
    fn parse_all_token_trees(&mut self) -> PResult<'a, TokenStream> {
        self.token = self.string_reader.next_token().0;
        let mut buf = TokenStreamBuilder::default();
        loop {
            match self.token.kind {
                token::OpenDelim(delim) => buf.push(self.parse_token_tree_open_delim(delim)),
                token::CloseDelim(delim) => return Err(self.close_delim_err(delim)),
                token::Eof => return Ok(buf.into_token_stream()),
                _ => buf.push(self.parse_token_tree_non_delim_non_eof()),
            }
        }
    }

    // Parse a stream of tokens into a list of `TokenTree`s, up to a `CloseDelim`.
    fn parse_token_trees_until_close_delim(&mut self) -> TokenStream {
        let mut buf = TokenStreamBuilder::default();
        loop {
            match self.token.kind {
                token::OpenDelim(delim) => buf.push(self.parse_token_tree_open_delim(delim)),
                token::CloseDelim(..) => return buf.into_token_stream(),
                token::Eof => {
                    self.eof_err().emit();
                    return buf.into_token_stream();
                }
                _ => buf.push(self.parse_token_tree_non_delim_non_eof()),
            }
        }
    }

    fn eof_err(&mut self) -> PErr<'a> {
        let msg = "this file contains an unclosed delimiter";
        let mut err = self.string_reader.sess.span_diagnostic.struct_span_err(self.token.span, msg);
        for &(_, sp) in &self.open_braces {
            err.span_label(sp, "unclosed delimiter");
            self.unmatched_braces.push(UnmatchedBrace {
                expected_delim: Delimiter::Brace,
                found_delim: None,
                found_span: self.token.span,
                unclosed_span: Some(sp),
                candidate_span: None,
            });
        }

        if let Some((delim, _)) = self.open_braces.last() {
            if let Some((_, open_sp, close_sp)) =
                self.matching_delim_spans.iter().find(|(d, open_sp, close_sp)| {
                    let sm = self.string_reader.sess.source_map();
                    if let Some(close_padding) = sm.span_to_margin(*close_sp) {
                        if let Some(open_padding) = sm.span_to_margin(*open_sp) {
                            return delim == d && close_padding != open_padding;
                        }
                    }
                    false
                })
            // these are in reverse order as they get inserted on close, but
            {
                // we want the last open/first close
                err.span_label(*open_sp, "this delimiter might not be properly closed...");
                err.span_label(*close_sp, "...as it matches this but it has different indentation");
            }
        }
        err
    }

    fn parse_token_tree_open_delim(&mut self, open_delim: Delimiter) -> TokenTree {
        // The span for beginning of the delimited section
        let pre_span = self.token.span;

        // Move past the open delimiter.
        self.open_braces.push((open_delim, self.token.span));
        self.token = self.string_reader.next_token().0;

        // Parse the token trees within the delimiters.
        // We stop at any delimiter so we can try to recover if the user
        // uses an incorrect delimiter.
        let tts = self.parse_token_trees_until_close_delim();

        // Expand to cover the entire delimited token tree
        let delim_span = DelimSpan::from_pair(pre_span, self.token.span);

        match self.token.kind {
            // Correct delimiter.
            token::CloseDelim(close_delim) if close_delim == open_delim => {
                let (open_brace, open_brace_span) = self.open_braces.pop().unwrap();
                let close_brace_span = self.token.span;

                if tts.is_empty() {
                    let empty_block_span = open_brace_span.to(close_brace_span);
                    let sm = self.string_reader.sess.source_map();
                    if !sm.is_multiline(empty_block_span) {
                        // Only track if the block is in the form of `{}`, otherwise it is
                        // likely that it was written on purpose.
                        self.last_delim_empty_block_spans.insert(open_delim, empty_block_span);
                    }
                }

                //only add braces
                if let (Delimiter::Brace, Delimiter::Brace) = (open_brace, open_delim) {
                    self.matching_block_spans.push((open_brace_span, close_brace_span));
                }

                if self.open_braces.is_empty() {
                    // Clear up these spans to avoid suggesting them as we've found
                    // properly matched delimiters so far for an entire block.
                    self.matching_delim_spans.clear();
                } else {
                    self.matching_delim_spans.push((open_brace, open_brace_span, close_brace_span));
                }
                // Move past the closing delimiter.
                self.token = self.string_reader.next_token().0;
            }
            // Incorrect delimiter.
            token::CloseDelim(close_delim) => {
                let mut unclosed_delimiter = None;
                let mut candidate = None;

                if self.last_unclosed_found_span != Some(self.token.span) {
                    // do not complain about the same unclosed delimiter multiple times
                    self.last_unclosed_found_span = Some(self.token.span);
                    // This is a conservative error: only report the last unclosed
                    // delimiter. The previous unclosed delimiters could actually be
                    // closed! The parser just hasn't gotten to them yet.
                    if let Some(&(_, sp)) = self.open_braces.last() {
                        unclosed_delimiter = Some(sp);
                    };
                    let sm = self.string_reader.sess.source_map();
                    if let Some(current_padding) = sm.span_to_margin(self.token.span) {
                        for (brace, brace_span) in &self.open_braces {
                            if let Some(padding) = sm.span_to_margin(*brace_span) {
                                // high likelihood of these two corresponding
                                if current_padding == padding && brace == &close_delim {
                                    candidate = Some(*brace_span);
                                }
                            }
                        }
                    }
                    let (tok, _) = self.open_braces.pop().unwrap();
                    self.unmatched_braces.push(UnmatchedBrace {
                        expected_delim: tok,
                        found_delim: Some(close_delim),
                        found_span: self.token.span,
                        unclosed_span: unclosed_delimiter,
                        candidate_span: candidate,
                    });
                } else {
                    self.open_braces.pop();
                }

                // If the incorrect delimiter matches an earlier opening
                // delimiter, then don't consume it (it can be used to
                // close the earlier one). Otherwise, consume it.
                // E.g., we try to recover from:
                // fn foo() {
                //     bar(baz(
                // }  // Incorrect delimiter but matches the earlier `{`
                if !self.open_braces.iter().any(|&(b, _)| b == close_delim) {
                    self.token = self.string_reader.next_token().0;
                }
            }
            token::Eof => {
                // Silently recover, the EOF token will be seen again
                // and an error emitted then. Thus we don't pop from
                // self.open_braces here.
            }
            _ => unreachable!(),
        }

        TokenTree::Delimited(delim_span, open_delim, tts)
    }

    fn close_delim_err(&mut self, delim: Delimiter) -> PErr<'a> {
        // An unexpected closing delimiter (i.e., there is no
        // matching opening delimiter).
        let token_str = token_to_string(&self.token);
        let msg = format!("unexpected closing delimiter: `{}`", token_str);
        let mut err =
            self.string_reader.sess.span_diagnostic.struct_span_err(self.token.span, &msg);

        // Braces are added at the end, so the last element is the biggest block
        if let Some(parent) = self.matching_block_spans.last() {
            if let Some(span) = self.last_delim_empty_block_spans.remove(&delim) {
                // Check if the (empty block) is in the last properly closed block
                if (parent.0.to(parent.1)).contains(span) {
                    err.span_label(span, "block is empty, you might have not meant to close it");
                } else {
                    err.span_label(parent.0, "this opening brace...");
                    err.span_label(parent.1, "...matches this closing brace");
                }
            } else {
                err.span_label(parent.0, "this opening brace...");
                err.span_label(parent.1, "...matches this closing brace");
            }
        }

        err.span_label(self.token.span, "unexpected closing delimiter");
        err
    }

    #[inline]
    fn parse_token_tree_non_delim_non_eof(&mut self) -> TokenTree {
        // `this_spacing` for the returned token refers to whether the token is
        // immediately followed by another op token. It is determined by the
        // next token: its kind and its `preceded_by_whitespace` status.
        let (next_tok, is_next_tok_preceded_by_whitespace) = self.string_reader.next_token();
        let this_spacing = if is_next_tok_preceded_by_whitespace || !next_tok.is_op() {
            Spacing::Alone
        } else {
            Spacing::Joint
        };
        let this_tok = std::mem::replace(&mut self.token, next_tok);
        TokenTree::Token(this_tok, this_spacing)
    }
}

#[derive(Default)]
struct TokenStreamBuilder {
    buf: Vec<TokenTree>,
}

impl TokenStreamBuilder {
    #[inline(always)]
    fn push(&mut self, tree: TokenTree) {
        if let Some(TokenTree::Token(prev_token, Spacing::Joint)) = self.buf.last()
            && let TokenTree::Token(token, joint) = &tree
            && let Some(glued) = prev_token.glue(token)
        {
            self.buf.pop();
            self.buf.push(TokenTree::Token(glued, *joint));
        } else {
            self.buf.push(tree)
        }
    }

    fn into_token_stream(self) -> TokenStream {
        TokenStream::new(self.buf)
    }
}
