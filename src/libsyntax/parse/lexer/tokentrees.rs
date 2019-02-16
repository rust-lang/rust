use crate::print::pprust::token_to_string;
use crate::parse::lexer::{StringReader, UnmatchedBrace};
use crate::parse::{token, PResult};
use crate::tokenstream::{DelimSpan, IsJoint::*, TokenStream, TokenTree, TreeAndJoint};

impl<'a> StringReader<'a> {
    // Parse a stream of tokens into a list of `TokenTree`s, up to an `Eof`.
    crate fn parse_all_token_trees(&mut self) -> PResult<'a, TokenStream> {
        let mut tts = Vec::new();

        while self.token != token::Eof {
            tts.push(self.parse_token_tree()?);
        }

        Ok(TokenStream::new(tts))
    }

    // Parse a stream of tokens into a list of `TokenTree`s, up to a `CloseDelim`.
    fn parse_token_trees_until_close_delim(&mut self) -> TokenStream {
        let mut tts = vec![];
        loop {
            if let token::CloseDelim(..) = self.token {
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
        let sm = self.sess.source_map();
        match self.token {
            token::Eof => {
                let msg = "this file contains an un-closed delimiter";
                let mut err = self.sess.span_diagnostic.struct_span_err(self.span, msg);
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
                let pre_span = self.span;

                // Parse the open delimiter.
                self.open_braces.push((delim, self.span));
                self.real_token();

                // Parse the token trees within the delimiters.
                // We stop at any delimiter so we can try to recover if the user
                // uses an incorrect delimiter.
                let tts = self.parse_token_trees_until_close_delim();

                // Expand to cover the entire delimited token tree
                let delim_span = DelimSpan::from_pair(pre_span, self.span);

                match self.token {
                    // Correct delimiter.
                    token::CloseDelim(d) if d == delim => {
                        let (open_brace, open_brace_span) = self.open_braces.pop().unwrap();
                        if self.open_braces.len() == 0 {
                            // Clear up these spans to avoid suggesting them as we've found
                            // properly matched delimiters so far for an entire block.
                            self.matching_delim_spans.clear();
                        } else {
                            self.matching_delim_spans.push(
                                (open_brace, open_brace_span, self.span),
                            );
                        }
                        // Parse the close delimiter.
                        self.real_token();
                    }
                    // Incorrect delimiter.
                    token::CloseDelim(other) => {
                        let mut unclosed_delimiter = None;
                        let mut candidate = None;
                        if self.last_unclosed_found_span != Some(self.span) {
                            // do not complain about the same unclosed delimiter multiple times
                            self.last_unclosed_found_span = Some(self.span);
                            // This is a conservative error: only report the last unclosed
                            // delimiter. The previous unclosed delimiters could actually be
                            // closed! The parser just hasn't gotten to them yet.
                            if let Some(&(_, sp)) = self.open_braces.last() {
                                unclosed_delimiter = Some(sp);
                            };
                            if let Some(current_padding) = sm.span_to_margin(self.span) {
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
                                found_span: self.span,
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
                let mut err = self.sess.span_diagnostic.struct_span_err(self.span, &msg);
                err.span_label(self.span, "unexpected close delimiter");
                Err(err)
            },
            _ => {
                let tt = TokenTree::Token(self.span, self.token.clone());
                // Note that testing for joint-ness here is done via the raw
                // source span as the joint-ness is a property of the raw source
                // rather than wanting to take `override_span` into account.
                let raw = self.span_src_raw;
                self.real_token();
                let is_joint = raw.hi() == self.span_src_raw.lo() && token::is_op(&self.token);
                Ok((tt, if is_joint { Joint } else { NonJoint }))
            }
        }
    }
}
