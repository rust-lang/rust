use syntax_pos::Span;

use crate::print::pprust::token_to_string;
use crate::parse::lexer::{StringReader, UnmatchedBrace};
use crate::parse::token::{self, Token};
use crate::parse::PResult;
use crate::tokenstream::{DelimSpan, IsJoint::*, TokenStream, TokenTree, TreeAndJoint};

impl<'a> StringReader<'a> {
    crate fn into_token_trees(self) -> (PResult<'a, TokenStream>, Vec<UnmatchedBrace>) {
        let mut tt_reader = TokenTreesReader {
            string_reader: self,
            token: Token::dummy(),
            open_braces: Vec::new(),
            unmatched_braces: Vec::new(),
            matching_delim_spans: Vec::new(),
            last_unclosed_found_span: None,
        };
        let res = tt_reader.parse_all_token_trees();
        (res, tt_reader.unmatched_braces)
    }
}

struct TokenTreesReader<'a> {
    string_reader: StringReader<'a>,
    token: Token,
    /// Stack of open delimiters and their spans. Used for error message.
    open_braces: Vec<(token::DelimToken, Span)>,
    unmatched_braces: Vec<UnmatchedBrace>,
    /// The type and spans for all braces
    ///
    /// Used only for error recovery when arriving to EOF with mismatched braces.
    matching_delim_spans: Vec<(token::DelimToken, Span, Span)>,
    last_unclosed_found_span: Option<Span>,
}

impl<'a> TokenTreesReader<'a> {
    // Parse a stream of tokens into a list of `TokenTree`s, up to an `Eof`.
    fn parse_all_token_trees(&mut self) -> PResult<'a, TokenStream> {
        let mut tts = Vec::new();

        self.real_token();
        while self.token != token::Eof {
            tts.push(self.parse_token_tree()?);
        }

        Ok(TokenStream::new(tts))
    }

    // Parse a stream of tokens into a list of `TokenTree`s, up to a `CloseDelim`.
    fn parse_token_trees_until_close_delim(&mut self) -> TokenStream {
        let mut tts = vec![];
        loop {
            if let token::CloseDelim(..) = self.token.kind {
                return TokenStream::new(tts);
            }

            match self.parse_token_tree() {
                Ok(tree) => tts.push(tree),
                Err(mut e) => {
                    e.emit();
                    return TokenStream::new(tts);
                }
            }
        }
    }

    fn parse_token_tree(&mut self) -> PResult<'a, TreeAndJoint> {
        let sm = self.string_reader.sess.source_map();
        match self.token.kind {
            token::Eof => {
                let msg = "this file contains an un-closed delimiter";
                let mut err = self.string_reader.sess.span_diagnostic
                    .struct_span_err(self.token.span, msg);
                for &(_, sp) in &self.open_braces {
                    err.span_label(sp, "un-closed delimiter");
                }

                if let Some((delim, _)) = self.open_braces.last() {
                    if let Some((_, open_sp, close_sp)) = self.matching_delim_spans.iter()
                        .filter(|(d, open_sp, close_sp)| {
                            if let Some(close_padding) = sm.span_to_margin(*close_sp) {
                                if let Some(open_padding) = sm.span_to_margin(*open_sp) {
                                    return delim == d && close_padding != open_padding;
                                }
                            }
                            false
                        }).next()  // these are in reverse order as they get inserted on close, but
                    {              // we want the last open/first close
                        err.span_label(
                            *open_sp,
                            "this delimiter might not be properly closed...",
                        );
                        err.span_label(
                            *close_sp,
                            "...as it matches this but it has different indentation",
                        );
                    }
                }
                Err(err)
            },
            token::OpenDelim(delim) => {
                // The span for beginning of the delimited section
                let pre_span = self.token.span;

                // Parse the open delimiter.
                self.open_braces.push((delim, self.token.span));
                self.real_token();

                // Parse the token trees within the delimiters.
                // We stop at any delimiter so we can try to recover if the user
                // uses an incorrect delimiter.
                let tts = self.parse_token_trees_until_close_delim();

                // Expand to cover the entire delimited token tree
                let delim_span = DelimSpan::from_pair(pre_span, self.token.span);

                match self.token.kind {
                    // Correct delimiter.
                    token::CloseDelim(d) if d == delim => {
                        let (open_brace, open_brace_span) = self.open_braces.pop().unwrap();
                        if self.open_braces.len() == 0 {
                            // Clear up these spans to avoid suggesting them as we've found
                            // properly matched delimiters so far for an entire block.
                            self.matching_delim_spans.clear();
                        } else {
                            self.matching_delim_spans.push(
                                (open_brace, open_brace_span, self.token.span),
                            );
                        }
                        // Parse the close delimiter.
                        self.real_token();
                    }
                    // Incorrect delimiter.
                    token::CloseDelim(other) => {
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
                            if let Some(current_padding) = sm.span_to_margin(self.token.span) {
                                for (brace, brace_span) in &self.open_braces {
                                    if let Some(padding) = sm.span_to_margin(*brace_span) {
                                        // high likelihood of these two corresponding
                                        if current_padding == padding && brace == &other {
                                            candidate = Some(*brace_span);
                                        }
                                    }
                                }
                            }
                            let (tok, _) = self.open_braces.pop().unwrap();
                            self.unmatched_braces.push(UnmatchedBrace {
                                expected_delim: tok,
                                found_delim: other,
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
                        if !self.open_braces.iter().any(|&(b, _)| b == other) {
                            self.real_token();
                        }
                    }
                    token::Eof => {
                        // Silently recover, the EOF token will be seen again
                        // and an error emitted then. Thus we don't pop from
                        // self.open_braces here.
                    },
                    _ => {}
                }

                Ok(TokenTree::Delimited(
                    delim_span,
                    delim,
                    tts.into()
                ).into())
            },
            token::CloseDelim(_) => {
                // An unexpected closing delimiter (i.e., there is no
                // matching opening delimiter).
                let token_str = token_to_string(&self.token);
                let msg = format!("unexpected close delimiter: `{}`", token_str);
                let mut err = self.string_reader.sess.span_diagnostic
                    .struct_span_err(self.token.span, &msg);
                err.span_label(self.token.span, "unexpected close delimiter");
                Err(err)
            },
            _ => {
                let tt = TokenTree::Token(self.token.take());
                // Note that testing for joint-ness here is done via the raw
                // source span as the joint-ness is a property of the raw source
                // rather than wanting to take `override_span` into account.
                // Additionally, we actually check if the *next* pair of tokens
                // is joint, but this is equivalent to checking the current pair.
                let raw = self.string_reader.peek_span_src_raw;
                self.real_token();
                let is_joint = raw.hi() == self.string_reader.peek_span_src_raw.lo()
                    && self.token.is_op();
                Ok((tt, if is_joint { Joint } else { NonJoint }))
            }
        }
    }

    fn real_token(&mut self) {
        self.token = self.string_reader.real_token();
    }
}
