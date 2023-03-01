use super::diagnostics::report_suspicious_mismatch_block;
use super::diagnostics::same_identation_level;
use super::diagnostics::TokenTreeDiagInfo;
use super::{StringReader, UnmatchedDelim};
use rustc_ast::token::{self, Delimiter, Token};
use rustc_ast::tokenstream::{DelimSpan, Spacing, TokenStream, TokenTree};
use rustc_ast_pretty::pprust::token_to_string;
use rustc_errors::{PErr, PResult};

pub(super) struct TokenTreesReader<'a> {
    string_reader: StringReader<'a>,
    /// The "next" token, which has been obtained from the `StringReader` but
    /// not yet handled by the `TokenTreesReader`.
    token: Token,
    diag_info: TokenTreeDiagInfo,
}

impl<'a> TokenTreesReader<'a> {
    pub(super) fn parse_all_token_trees(
        string_reader: StringReader<'a>,
    ) -> (PResult<'a, TokenStream>, Vec<UnmatchedDelim>) {
        let mut tt_reader = TokenTreesReader {
            string_reader,
            token: Token::dummy(),
            diag_info: TokenTreeDiagInfo::default(),
        };
        let res = tt_reader.parse_token_trees(/* is_delimited */ false);
        (res, tt_reader.diag_info.unmatched_delims)
    }

    // Parse a stream of tokens into a list of `TokenTree`s.
    fn parse_token_trees(&mut self, is_delimited: bool) -> PResult<'a, TokenStream> {
        self.token = self.string_reader.next_token().0;
        let mut buf = Vec::new();
        loop {
            match self.token.kind {
                token::OpenDelim(delim) => buf.push(self.parse_token_tree_open_delim(delim)?),
                token::CloseDelim(delim) => {
                    return if is_delimited {
                        Ok(TokenStream::new(buf))
                    } else {
                        Err(self.close_delim_err(delim))
                    };
                }
                token::Eof => {
                    return if is_delimited {
                        Err(self.eof_err())
                    } else {
                        Ok(TokenStream::new(buf))
                    };
                }
                _ => {
                    // Get the next normal token. This might require getting multiple adjacent
                    // single-char tokens and joining them together.
                    let (this_spacing, next_tok) = loop {
                        let (next_tok, is_next_tok_preceded_by_whitespace) =
                            self.string_reader.next_token();
                        if !is_next_tok_preceded_by_whitespace {
                            if let Some(glued) = self.token.glue(&next_tok) {
                                self.token = glued;
                            } else {
                                let this_spacing =
                                    if next_tok.is_op() { Spacing::Joint } else { Spacing::Alone };
                                break (this_spacing, next_tok);
                            }
                        } else {
                            break (Spacing::Alone, next_tok);
                        }
                    };
                    let this_tok = std::mem::replace(&mut self.token, next_tok);
                    buf.push(TokenTree::Token(this_tok, this_spacing));
                }
            }
        }
    }

    fn eof_err(&mut self) -> PErr<'a> {
        let msg = "this file contains an unclosed delimiter";
        let mut err = self.string_reader.sess.span_diagnostic.struct_span_err(self.token.span, msg);
        for &(_, sp) in &self.diag_info.open_braces {
            err.span_label(sp, "unclosed delimiter");
            self.diag_info.unmatched_delims.push(UnmatchedDelim {
                expected_delim: Delimiter::Brace,
                found_delim: None,
                found_span: self.token.span,
                unclosed_span: Some(sp),
                candidate_span: None,
            });
        }

        if let Some((delim, _)) = self.diag_info.open_braces.last() {
            report_suspicious_mismatch_block(
                &mut err,
                &self.diag_info,
                &self.string_reader.sess.source_map(),
                *delim,
            )
        }
        err
    }

    fn parse_token_tree_open_delim(&mut self, open_delim: Delimiter) -> PResult<'a, TokenTree> {
        // The span for beginning of the delimited section
        let pre_span = self.token.span;

        self.diag_info.open_braces.push((open_delim, self.token.span));

        // Parse the token trees within the delimiters.
        // We stop at any delimiter so we can try to recover if the user
        // uses an incorrect delimiter.
        let tts = self.parse_token_trees(/* is_delimited */ true)?;

        // Expand to cover the entire delimited token tree
        let delim_span = DelimSpan::from_pair(pre_span, self.token.span);
        let sm = self.string_reader.sess.source_map();

        match self.token.kind {
            // Correct delimiter.
            token::CloseDelim(close_delim) if close_delim == open_delim => {
                let (open_brace, open_brace_span) = self.diag_info.open_braces.pop().unwrap();
                let close_brace_span = self.token.span;

                if tts.is_empty() && close_delim == Delimiter::Brace {
                    let empty_block_span = open_brace_span.to(close_brace_span);
                    if !sm.is_multiline(empty_block_span) {
                        // Only track if the block is in the form of `{}`, otherwise it is
                        // likely that it was written on purpose.
                        self.diag_info.empty_block_spans.push(empty_block_span);
                    }
                }

                // only add braces
                if let (Delimiter::Brace, Delimiter::Brace) = (open_brace, open_delim) {
                    // Add all the matching spans, we will sort by span later
                    self.diag_info.matching_block_spans.push((open_brace_span, close_brace_span));
                }

                // Move past the closing delimiter.
                self.token = self.string_reader.next_token().0;
            }
            // Incorrect delimiter.
            token::CloseDelim(close_delim) => {
                let mut unclosed_delimiter = None;
                let mut candidate = None;

                if self.diag_info.last_unclosed_found_span != Some(self.token.span) {
                    // do not complain about the same unclosed delimiter multiple times
                    self.diag_info.last_unclosed_found_span = Some(self.token.span);
                    // This is a conservative error: only report the last unclosed
                    // delimiter. The previous unclosed delimiters could actually be
                    // closed! The parser just hasn't gotten to them yet.
                    if let Some(&(_, sp)) = self.diag_info.open_braces.last() {
                        unclosed_delimiter = Some(sp);
                    };
                    for (brace, brace_span) in &self.diag_info.open_braces {
                        if same_identation_level(&sm, self.token.span, *brace_span)
                            && brace == &close_delim
                        {
                            // high likelihood of these two corresponding
                            candidate = Some(*brace_span);
                        }
                    }
                    let (tok, _) = self.diag_info.open_braces.pop().unwrap();
                    self.diag_info.unmatched_delims.push(UnmatchedDelim {
                        expected_delim: tok,
                        found_delim: Some(close_delim),
                        found_span: self.token.span,
                        unclosed_span: unclosed_delimiter,
                        candidate_span: candidate,
                    });
                } else {
                    self.diag_info.open_braces.pop();
                }

                // If the incorrect delimiter matches an earlier opening
                // delimiter, then don't consume it (it can be used to
                // close the earlier one). Otherwise, consume it.
                // E.g., we try to recover from:
                // fn foo() {
                //     bar(baz(
                // }  // Incorrect delimiter but matches the earlier `{`
                if !self.diag_info.open_braces.iter().any(|&(b, _)| b == close_delim) {
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

        Ok(TokenTree::Delimited(delim_span, open_delim, tts))
    }

    fn close_delim_err(&mut self, delim: Delimiter) -> PErr<'a> {
        // An unexpected closing delimiter (i.e., there is no
        // matching opening delimiter).
        let token_str = token_to_string(&self.token);
        let msg = format!("unexpected closing delimiter: `{}`", token_str);
        let mut err =
            self.string_reader.sess.span_diagnostic.struct_span_err(self.token.span, &msg);

        report_suspicious_mismatch_block(
            &mut err,
            &self.diag_info,
            &self.string_reader.sess.source_map(),
            delim,
        );
        err.span_label(self.token.span, "unexpected closing delimiter");
        err
    }
}
