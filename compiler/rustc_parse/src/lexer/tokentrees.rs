use rustc_ast::token::{self, Delimiter, Token};
use rustc_ast::tokenstream::{DelimSpacing, DelimSpan, Spacing, TokenStream, TokenTree};
use rustc_ast_pretty::pprust::token_to_string;
use rustc_errors::Diag;

use super::diagnostics::{report_suspicious_mismatch_block, same_indentation_level};
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
            match self.token.kind {
                token::OpenDelim(delim) => {
                    // Invisible delimiters cannot occur here because `TokenTreesReader` parses
                    // code directly from strings, with no macro expansion involved.
                    debug_assert!(!matches!(delim, Delimiter::Invisible(_)));
                    buf.push(match self.lex_token_tree_open_delim(delim) {
                        Ok(val) => val,
                        Err(errs) => return Err(errs),
                    })
                }
                token::CloseDelim(delim) => {
                    // Invisible delimiters cannot occur here because `TokenTreesReader` parses
                    // code directly from strings, with no macro expansion involved.
                    debug_assert!(!matches!(delim, Delimiter::Invisible(_)));
                    return if is_delimited {
                        Ok((open_spacing, TokenStream::new(buf)))
                    } else {
                        Err(vec![self.close_delim_err(delim)])
                    };
                }
                token::Eof => {
                    return if is_delimited {
                        Err(vec![self.eof_err()])
                    } else {
                        Ok((open_spacing, TokenStream::new(buf)))
                    };
                }
                _ => {
                    // Get the next normal token.
                    let (this_tok, this_spacing) = self.bump();
                    buf.push(TokenTree::Token(this_tok, this_spacing));
                }
            }
        }
    }

    fn eof_err(&mut self) -> Diag<'psess> {
        let msg = "this file contains an unclosed delimiter";
        let mut err = self.dcx().struct_span_err(self.token.span, msg);

        let unclosed_delimiter_show_limit = 5;
        let len = usize::min(unclosed_delimiter_show_limit, self.diag_info.open_braces.len());
        for &(_, span) in &self.diag_info.open_braces[..len] {
            err.span_label(span, "unclosed delimiter");
            self.diag_info.unmatched_delims.push(UnmatchedDelim {
                found_delim: None,
                found_span: self.token.span,
                unclosed_span: Some(span),
                candidate_span: None,
            });
        }

        if let Some((_, span)) = self.diag_info.open_braces.get(unclosed_delimiter_show_limit)
            && self.diag_info.open_braces.len() >= unclosed_delimiter_show_limit + 2
        {
            err.span_label(
                *span,
                format!(
                    "another {} unclosed delimiters begin from here",
                    self.diag_info.open_braces.len() - unclosed_delimiter_show_limit
                ),
            );
        }

        if let Some((delim, _)) = self.diag_info.open_braces.last() {
            report_suspicious_mismatch_block(
                &mut err,
                &self.diag_info,
                self.psess.source_map(),
                *delim,
            )
        }
        err
    }

    fn lex_token_tree_open_delim(
        &mut self,
        open_delim: Delimiter,
    ) -> Result<TokenTree, Vec<Diag<'psess>>> {
        // The span for beginning of the delimited section.
        let pre_span = self.token.span;

        self.diag_info.open_braces.push((open_delim, self.token.span));

        // Lex the token trees within the delimiters.
        // We stop at any delimiter so we can try to recover if the user
        // uses an incorrect delimiter.
        let (open_spacing, tts) = self.lex_token_trees(/* is_delimited */ true)?;

        // Expand to cover the entire delimited token tree.
        let delim_span = DelimSpan::from_pair(pre_span, self.token.span);
        let sm = self.psess.source_map();

        let close_spacing = match self.token.kind {
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
                self.bump_minimal()
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
                    // closed! The lexer just hasn't gotten to them yet.
                    if let Some(&(_, sp)) = self.diag_info.open_braces.last() {
                        unclosed_delimiter = Some(sp);
                    };
                    for (brace, brace_span) in &self.diag_info.open_braces {
                        if same_indentation_level(sm, self.token.span, *brace_span)
                            && brace == &close_delim
                        {
                            // high likelihood of these two corresponding
                            candidate = Some(*brace_span);
                        }
                    }
                    let (_, _) = self.diag_info.open_braces.pop().unwrap();
                    self.diag_info.unmatched_delims.push(UnmatchedDelim {
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
                    self.bump_minimal()
                } else {
                    // The choice of value here doesn't matter.
                    Spacing::Alone
                }
            }
            token::Eof => {
                // Silently recover, the EOF token will be seen again
                // and an error emitted then. Thus we don't pop from
                // self.open_braces here. The choice of spacing value here
                // doesn't matter.
                Spacing::Alone
            }
            _ => unreachable!(),
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
                let this_spacing = if next_tok.is_punct() {
                    Spacing::Joint
                } else if next_tok == token::Eof {
                    Spacing::Alone
                } else {
                    Spacing::JointHidden
                };
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
            if next_tok.is_punct() {
                Spacing::Joint
            } else if next_tok == token::Eof {
                Spacing::Alone
            } else {
                Spacing::JointHidden
            }
        };

        self.token = next_tok;
        this_spacing
    }

    fn close_delim_err(&mut self, delim: Delimiter) -> Diag<'psess> {
        // An unexpected closing delimiter (i.e., there is no matching opening delimiter).
        let token_str = token_to_string(&self.token);
        let msg = format!("unexpected closing delimiter: `{token_str}`");
        let mut err = self.dcx().struct_span_err(self.token.span, msg);

        report_suspicious_mismatch_block(&mut err, &self.diag_info, self.psess.source_map(), delim);
        err.span_label(self.token.span, "unexpected closing delimiter");
        err
    }
}
