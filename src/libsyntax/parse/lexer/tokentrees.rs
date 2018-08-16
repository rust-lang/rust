// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use print::pprust::token_to_string;
use parse::lexer::StringReader;
use parse::{token, PResult};
use tokenstream::{Delimited, TokenStream, TokenTree};

impl<'a> StringReader<'a> {
    // Parse a stream of tokens into a list of `TokenTree`s, up to an `Eof`.
    crate fn parse_all_token_trees(&mut self) -> PResult<'a, TokenStream> {
        let mut tts = Vec::new();

        while self.token != token::Eof {
            tts.push(self.parse_token_tree()?);
        }

        Ok(TokenStream::concat(tts))
    }

    // Parse a stream of tokens into a list of `TokenTree`s, up to a `CloseDelim`.
    fn parse_token_trees_until_close_delim(&mut self) -> TokenStream {
        let mut tts = vec![];
        loop {
            if let token::CloseDelim(..) = self.token {
                return TokenStream::concat(tts);
            }

            match self.parse_token_tree() {
                Ok(tree) => tts.push(tree),
                Err(mut e) => {
                    e.emit();
                    return TokenStream::concat(tts);
                }
            }
        }
    }

    fn parse_token_tree(&mut self) -> PResult<'a, TokenStream> {
        match self.token {
            token::Eof => {
                let msg = "this file contains an un-closed delimiter";
                let mut err = self.sess.span_diagnostic.struct_span_err(self.span, msg);
                for &(_, sp) in &self.open_braces {
                    err.span_help(sp, "did you mean to close this delimiter?");
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
                let span = pre_span.with_hi(self.span.hi());

                match self.token {
                    // Correct delimiter.
                    token::CloseDelim(d) if d == delim => {
                        self.open_braces.pop().unwrap();

                        // Parse the close delimiter.
                        self.real_token();
                    }
                    // Incorrect delimiter.
                    token::CloseDelim(other) => {
                        let token_str = token_to_string(&self.token);
                        if self.last_unclosed_found_span != Some(self.span) {
                            // do not complain about the same unclosed delimiter multiple times
                            self.last_unclosed_found_span = Some(self.span);
                            let msg = format!("incorrect close delimiter: `{}`", token_str);
                            let mut err = self.sess.span_diagnostic.struct_span_err(
                                self.span,
                                &msg,
                            );
                            err.span_label(self.span, "incorrect close delimiter");
                            // This is a conservative error: only report the last unclosed
                            // delimiter. The previous unclosed delimiters could actually be
                            // closed! The parser just hasn't gotten to them yet.
                            if let Some(&(_, sp)) = self.open_braces.last() {
                                err.span_label(sp, "unclosed delimiter");
                            };
                            err.emit();
                        }
                        self.open_braces.pop().unwrap();

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

                Ok(TokenTree::Delimited(span, Delimited {
                    delim,
                    tts: tts.into(),
                }).into())
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

                Ok(if is_joint { tt.joint() } else { tt.into() })
            }
        }
    }
}
