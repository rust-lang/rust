use super::diagnostics::report_suspicious_mismatch_block;
use super::diagnostics::same_indentation_level;
use super::diagnostics::TokenTreeDiagInfo;
use super::{StringReader, UnmatchedDelim};
use rustc_ast::token::{self, Delimiter, Token};
use rustc_ast::tokenstream::{DelimSpacing, DelimSpan, Spacing, TokenStream, TokenTree};
use rustc_ast_pretty::pprust::token_to_string;
use rustc_errors::{Applicability, PErr};
use rustc_span::symbol::kw;

pub(super) struct TokenTreesReader<'psess, 'src> {
    string_reader: StringReader<'psess, 'src>,
    /// The "next" token, which has been obtained from the `StringReader` but
    /// not yet handled by the `TokenTreesReader`.
    token: Token,
    diag_info: TokenTreeDiagInfo,
}

impl<'psess, 'src> TokenTreesReader<'psess, 'src> {
    pub(super) fn parse_all_token_trees(
        string_reader: StringReader<'psess, 'src>,
    ) -> (TokenStream, Result<(), Vec<PErr<'psess>>>, Vec<UnmatchedDelim>) {
        let mut tt_reader = TokenTreesReader {
            string_reader,
            token: Token::dummy(),
            diag_info: TokenTreeDiagInfo::default(),
        };
        let (_open_spacing, stream, res) =
            tt_reader.parse_token_trees(/* is_delimited */ false);
        (stream, res, tt_reader.diag_info.unmatched_delims)
    }

    // Parse a stream of tokens into a list of `TokenTree`s. The `Spacing` in
    // the result is that of the opening delimiter.
    fn parse_token_trees(
        &mut self,
        is_delimited: bool,
    ) -> (Spacing, TokenStream, Result<(), Vec<PErr<'psess>>>) {
        // Move past the opening delimiter.
        let (_, open_spacing) = self.bump(false);

        let mut buf = Vec::new();
        loop {
            match self.token.kind {
                token::OpenDelim(delim) => {
                    buf.push(match self.parse_token_tree_open_delim(delim) {
                        Ok(val) => val,
                        Err(errs) => return (open_spacing, TokenStream::new(buf), Err(errs)),
                    })
                }
                token::CloseDelim(delim) => {
                    return (
                        open_spacing,
                        TokenStream::new(buf),
                        if is_delimited { Ok(()) } else { Err(vec![self.close_delim_err(delim)]) },
                    );
                }
                token::Eof => {
                    return (
                        open_spacing,
                        TokenStream::new(buf),
                        if is_delimited { Err(vec![self.eof_err()]) } else { Ok(()) },
                    );
                }
                _ => {
                    // Get the next normal token.
                    let (this_tok, this_spacing) = self.bump(true);
                    buf.push(TokenTree::Token(this_tok, this_spacing));
                }
            }
        }
    }

    fn eof_err(&mut self) -> PErr<'psess> {
        let msg = "this file contains an unclosed delimiter";
        let mut err = self.string_reader.psess.dcx.struct_span_err(self.token.span, msg);
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
                self.string_reader.psess.source_map(),
                *delim,
            )
        }
        err
    }

    fn parse_token_tree_open_delim(
        &mut self,
        open_delim: Delimiter,
    ) -> Result<TokenTree, Vec<PErr<'psess>>> {
        // The span for beginning of the delimited section
        let pre_span = self.token.span;

        self.diag_info.open_braces.push((open_delim, self.token.span));

        // Parse the token trees within the delimiters.
        // We stop at any delimiter so we can try to recover if the user
        // uses an incorrect delimiter.
        let (open_spacing, tts, res) = self.parse_token_trees(/* is_delimited */ true);
        if let Err(errs) = res {
            return Err(self.unclosed_delim_err(tts, errs));
        }

        // Expand to cover the entire delimited token tree
        let delim_span = DelimSpan::from_pair(pre_span, self.token.span);
        let sm = self.string_reader.psess.source_map();

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
                self.bump(false).1
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
                        if same_indentation_level(sm, self.token.span, *brace_span)
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
                    self.bump(false).1
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
    // Will glue adjacent single-char tokens together if `glue` is set.
    fn bump(&mut self, glue: bool) -> (Token, Spacing) {
        let (this_spacing, next_tok) = loop {
            let (next_tok, is_next_tok_preceded_by_whitespace) = self.string_reader.next_token();

            if is_next_tok_preceded_by_whitespace {
                break (Spacing::Alone, next_tok);
            } else if glue && let Some(glued) = self.token.glue(&next_tok) {
                self.token = glued;
            } else {
                let this_spacing = if next_tok.is_punct() {
                    Spacing::Joint
                } else if next_tok.kind == token::Eof {
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

    fn unclosed_delim_err(
        &mut self,
        tts: TokenStream,
        mut errs: Vec<PErr<'psess>>,
    ) -> Vec<PErr<'psess>> {
        // If there are unclosed delims, see if there are diff markers and if so, point them
        // out instead of complaining about the unclosed delims.
        let mut parser = crate::stream_to_parser(self.string_reader.psess, tts, None);
        let mut diff_errs = vec![];
        // Suggest removing a `{` we think appears in an `if`/`while` condition
        // We want to suggest removing a `{` only if we think we're in an `if`/`while` condition, but
        // we have no way of tracking this in the lexer itself, so we piggyback on the parser
        let mut in_cond = false;
        while parser.token != token::Eof {
            if let Err(diff_err) = parser.err_diff_marker() {
                diff_errs.push(diff_err);
            } else if parser.is_keyword_ahead(0, &[kw::If, kw::While]) {
                in_cond = true;
            } else if matches!(
                parser.token.kind,
                token::CloseDelim(Delimiter::Brace) | token::FatArrow
            ) {
                // end of the `if`/`while` body, or the end of a `match` guard
                in_cond = false;
            } else if in_cond && parser.token == token::OpenDelim(Delimiter::Brace) {
                // Store the `&&` and `let` to use their spans later when creating the diagnostic
                let maybe_andand = parser.look_ahead(1, |t| t.clone());
                let maybe_let = parser.look_ahead(2, |t| t.clone());
                if maybe_andand == token::OpenDelim(Delimiter::Brace) {
                    // This might be the beginning of the `if`/`while` body (i.e., the end of the condition)
                    in_cond = false;
                } else if maybe_andand == token::AndAnd && maybe_let.is_keyword(kw::Let) {
                    let mut err = parser.dcx().struct_span_err(
                        parser.token.span,
                        "found a `{` in the middle of a let-chain",
                    );
                    err.span_suggestion(
                        parser.token.span,
                        "consider removing this brace to parse the `let` as part of the same chain",
                        "",
                        Applicability::MachineApplicable,
                    );
                    err.span_label(
                        maybe_andand.span.to(maybe_let.span),
                        "you might have meant to continue the let-chain here",
                    );
                    errs.push(err);
                }
            }
            parser.bump();
        }
        if !diff_errs.is_empty() {
            for err in errs {
                err.cancel();
            }
            return diff_errs;
        }
        return errs;
    }

    fn close_delim_err(&mut self, delim: Delimiter) -> PErr<'psess> {
        // An unexpected closing delimiter (i.e., there is no
        // matching opening delimiter).
        let token_str = token_to_string(&self.token);
        let msg = format!("unexpected closing delimiter: `{token_str}`");
        let mut err = self.string_reader.psess.dcx.struct_span_err(self.token.span, msg);

        report_suspicious_mismatch_block(
            &mut err,
            &self.diag_info,
            self.string_reader.psess.source_map(),
            delim,
        );
        err.span_label(self.token.span, "unexpected closing delimiter");
        err
    }
}
