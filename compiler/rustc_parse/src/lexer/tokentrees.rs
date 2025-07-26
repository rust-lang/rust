use rustc_ast::token::{self, Delimiter, Token};
use rustc_ast::tokenstream::{DelimSpacing, DelimSpan, Spacing, TokenStream, TokenTree};
use rustc_ast_pretty::pprust::token_to_string;
use rustc_errors::Diag;

use super::diagnostics::{
    report_missing_open_delim, report_suspicious_mismatch_block, same_indentation_level,
};
use super::{Lexer, UnmatchedDelim};

impl<'psess, 'src> Lexer<'psess, 'src> {
    // Lex into a token stream. The `Spacing` in the result is that of the
    // opening delimiter.
    pub(super) fn lex_token_trees(
        &mut self,
        is_delimited: bool,
    ) -> Result<(Spacing, TokenStream), Vec<Diag<'psess>>> {
        // Move past the opening delimiter.
        let open_spacing = self.bump_minimal();

        let mut buf = Vec::new();
        loop {
            if let Some(delim) = self.token.kind.open_delim() {
                // Invisible delimiters cannot occur here because `TokenTreesReader` parses
                // code directly from strings, with no macro expansion involved.
                debug_assert!(!matches!(delim, Delimiter::Invisible(_)));
                buf.push(match self.lex_token_tree_open_delim(delim) {
                    Ok(val) => val,
                    Err(errs) => return Err(errs),
                })
            } else if let Some(delim) = self.token.kind.close_delim() {
                // Invisible delimiters cannot occur here because `TokenTreesReader` parses
                // code directly from strings, with no macro expansion involved.
                debug_assert!(!matches!(delim, Delimiter::Invisible(_)));
                return if is_delimited {
                    Ok((open_spacing, TokenStream::new(buf)))
                } else {
                    Err(vec![self.close_delim_err(delim)])
                };
            } else if self.token.kind == token::Eof {
                return if is_delimited {
                    Err(vec![self.eof_err()])
                } else {
                    Ok((open_spacing, TokenStream::new(buf)))
                };
            } else {
                // Get the next normal token.
                let (this_tok, this_spacing) = self.bump();
                buf.push(TokenTree::Token(this_tok, this_spacing));
            }
        }
    }

    fn lex_token_tree_open_delim(
        &mut self,
        open_delim: Delimiter,
    ) -> Result<TokenTree, Vec<Diag<'psess>>> {
        // The span for beginning of the delimited section.
        let pre_span = self.token.span;

        self.diag_info.open_delimiters.push((open_delim, self.token.span));

        // Lex the token trees within the delimiters.
        // We stop at any delimiter so we can try to recover if the user
        // uses an incorrect delimiter.
        let (open_spacing, tts) = self.lex_token_trees(/* is_delimited */ true)?;

        // Expand to cover the entire delimited token tree.
        let delim_span = DelimSpan::from_pair(pre_span, self.token.span);
        let sm = self.psess.source_map();

        let close_spacing = if let Some(close_delim) = self.token.kind.close_delim() {
            if close_delim == open_delim {
                // Correct delimiter.
                let (open_delimiter, open_delimiter_span) =
                    self.diag_info.open_delimiters.pop().unwrap();
                let close_delimiter_span = self.token.span;

                if tts.is_empty() && close_delim == Delimiter::Brace {
                    let empty_block_span = open_delimiter_span.to(close_delimiter_span);
                    if !sm.is_multiline(empty_block_span) {
                        // Only track if the block is in the form of `{}`, otherwise it is
                        // likely that it was written on purpose.
                        self.diag_info.empty_block_spans.push(empty_block_span);
                    }
                }

                // only add braces
                if let (Delimiter::Brace, Delimiter::Brace) = (open_delimiter, open_delim) {
                    // Add all the matching spans, we will sort by span later
                    self.diag_info
                        .matching_block_spans
                        .push((open_delimiter_span, close_delimiter_span));
                }

                // Move past the closing delimiter.
                self.bump_minimal()
            } else {
                // Incorrect delimiter.
                let mut unclosed_delimiter = None;
                let mut candidate = None;

                if self.diag_info.last_unclosed_found_span != Some(self.token.span) {
                    // do not complain about the same unclosed delimiter multiple times
                    self.diag_info.last_unclosed_found_span = Some(self.token.span);
                    // This is a conservative error: only report the last unclosed
                    // delimiter. The previous unclosed delimiters could actually be
                    // closed! The lexer just hasn't gotten to them yet.
                    if let Some(&(_, sp)) = self.diag_info.open_delimiters.last() {
                        unclosed_delimiter = Some(sp);
                    };
                    for (delimiter, delimiter_span) in &self.diag_info.open_delimiters {
                        if same_indentation_level(sm, self.token.span, *delimiter_span)
                            && delimiter == &close_delim
                        {
                            // high likelihood of these two corresponding
                            candidate = Some(*delimiter_span);
                        }
                    }
                    let (_, _) = self.diag_info.open_delimiters.pop().unwrap();
                    self.diag_info.unmatched_delims.push(UnmatchedDelim {
                        found_delim: Some(close_delim),
                        found_span: self.token.span,
                        unclosed_span: unclosed_delimiter,
                        candidate_span: candidate,
                    });
                } else {
                    self.diag_info.open_delimiters.pop();
                }

                // If the incorrect delimiter matches an earlier opening
                // delimiter, then don't consume it (it can be used to
                // close the earlier one). Otherwise, consume it.
                // E.g., we try to recover from:
                // fn foo() {
                //     bar(baz(
                // }  // Incorrect delimiter but matches the earlier `{`
                if !self.diag_info.open_delimiters.iter().any(|&(d, _)| d == close_delim) {
                    self.bump_minimal()
                } else {
                    // The choice of value here doesn't matter.
                    Spacing::Alone
                }
            }
        } else {
            assert_eq!(self.token.kind, token::Eof);
            // Silently recover, the EOF token will be seen again
            // and an error emitted then. Thus we don't pop from
            // self.open_delimiters here. The choice of spacing value here
            // doesn't matter.
            Spacing::Alone
        };

        let spacing = DelimSpacing::new(open_spacing, close_spacing);

        Ok(TokenTree::Delimited(delim_span, spacing, open_delim, tts))
    }

    // Move on to the next token, returning the current token and its spacing.
    // Will glue adjacent single-char tokens together.
    fn bump(&mut self) -> (Token, Spacing) {
        let (this_spacing, next_tok) = loop {
            let (next_tok, is_next_tok_preceded_by_whitespace) = self.next_token_from_cursor();

            if is_next_tok_preceded_by_whitespace {
                break (Spacing::Alone, next_tok);
            } else if let Some(glued) = self.token.glue(&next_tok) {
                self.token = glued;
            } else {
                let this_spacing = self.calculate_spacing(&next_tok);
                break (this_spacing, next_tok);
            }
        };
        let this_tok = std::mem::replace(&mut self.token, next_tok);
        (this_tok, this_spacing)
    }

    // Cut-down version of `bump` used when the token kind is known in advance.
    fn bump_minimal(&mut self) -> Spacing {
        let (next_tok, is_next_tok_preceded_by_whitespace) = self.next_token_from_cursor();
        let this_spacing = if is_next_tok_preceded_by_whitespace {
            Spacing::Alone
        } else {
            self.calculate_spacing(&next_tok)
        };
        self.token = next_tok;
        this_spacing
    }

    fn calculate_spacing(&self, next_tok: &Token) -> Spacing {
        if next_tok.is_punct() {
            Spacing::Joint
        } else if *next_tok == token::Eof {
            Spacing::Alone
        } else {
            Spacing::JointHidden
        }
    }

    fn eof_err(&mut self) -> Diag<'psess> {
        const UNCLOSED_DELIMITER_SHOW_LIMIT: usize = 5;
        let msg = "this file contains an unclosed delimiter";
        let mut err = self.dcx().struct_span_err(self.token.span, msg);

        let len = usize::min(UNCLOSED_DELIMITER_SHOW_LIMIT, self.diag_info.open_delimiters.len());
        for &(_, span) in &self.diag_info.open_delimiters[..len] {
            err.span_label(span, "unclosed delimiter");
            self.diag_info.unmatched_delims.push(UnmatchedDelim {
                found_delim: None,
                found_span: self.token.span,
                unclosed_span: Some(span),
                candidate_span: None,
            });
        }

        if let Some((_, span)) = self.diag_info.open_delimiters.get(UNCLOSED_DELIMITER_SHOW_LIMIT)
            && self.diag_info.open_delimiters.len() >= UNCLOSED_DELIMITER_SHOW_LIMIT + 2
        {
            err.span_label(
                *span,
                format!(
                    "another {} unclosed delimiters begin from here",
                    self.diag_info.open_delimiters.len() - UNCLOSED_DELIMITER_SHOW_LIMIT
                ),
            );
        }

        if let Some((delim, _)) = self.diag_info.open_delimiters.last() {
            report_suspicious_mismatch_block(
                &mut err,
                &self.diag_info,
                self.psess.source_map(),
                *delim,
            )
        }
        err
    }

    fn close_delim_err(&mut self, delim: Delimiter) -> Diag<'psess> {
        // An unexpected closing delimiter (i.e., there is no matching opening delimiter).
        let token_str = token_to_string(&self.token);
        let msg = format!("unexpected closing delimiter: `{token_str}`");
        let mut err = self.dcx().struct_span_err(self.token.span, msg);

        // if there is no missing open delim, report suspicious mismatch block
        if !report_missing_open_delim(&mut err, &mut self.diag_info.unmatched_delims) {
            report_suspicious_mismatch_block(
                &mut err,
                &self.diag_info,
                self.psess.source_map(),
                delim,
            );
        }

        err.span_label(self.token.span, "unexpected closing delimiter");
        err
    }
}
