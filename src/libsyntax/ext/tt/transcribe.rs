// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use ast::Ident;
use ext::base::ExtCtxt;
use ext::expand::Marker;
use ext::tt::macro_parser::{NamedMatch, MatchedSeq, MatchedNonterminal};
use ext::tt::quoted;
use fold::noop_fold_tt;
use parse::token::{self, Token, NtTT};
use OneVector;
use syntax_pos::{Span, DUMMY_SP};
use tokenstream::{TokenStream, TokenTree, Delimited};

use std::rc::Rc;
use rustc_data_structures::sync::Lrc;
use std::mem;
use std::ops::Add;
use std::collections::HashMap;

// An iterator over the token trees in a delimited token tree (`{ ... }`) or a sequence (`$(...)`).
enum Frame {
    Delimited {
        forest: Lrc<quoted::Delimited>,
        idx: usize,
        span: Span,
    },
    Sequence {
        forest: Lrc<quoted::SequenceRepetition>,
        idx: usize,
        sep: Option<Token>,
    },
}

impl Frame {
    fn new(tts: Vec<quoted::TokenTree>) -> Frame {
        let forest = Lrc::new(quoted::Delimited { delim: token::NoDelim, tts: tts });
        Frame::Delimited { forest: forest, idx: 0, span: DUMMY_SP }
    }
}

impl Iterator for Frame {
    type Item = quoted::TokenTree;

    fn next(&mut self) -> Option<quoted::TokenTree> {
        match *self {
            Frame::Delimited { ref forest, ref mut idx, .. } => {
                *idx += 1;
                forest.tts.get(*idx - 1).cloned()
            }
            Frame::Sequence { ref forest, ref mut idx, .. } => {
                *idx += 1;
                forest.tts.get(*idx - 1).cloned()
            }
        }
    }
}

/// This can do Macro-By-Example transcription. On the other hand, if
/// `src` contains no `TokenTree::{Sequence, MetaVar, MetaVarDecl}`s, `interp` can
/// (and should) be None.
pub fn transcribe(cx: &ExtCtxt,
                  interp: Option<HashMap<Ident, Rc<NamedMatch>>>,
                  src: Vec<quoted::TokenTree>)
                  -> TokenStream {
    let mut stack: OneVector<Frame> = smallvec![Frame::new(src)];
    let interpolations = interp.unwrap_or_else(HashMap::new); /* just a convenience */
    let mut repeats = Vec::new();
    let mut result: Vec<TokenStream> = Vec::new();
    let mut result_stack = Vec::new();

    loop {
        let tree = if let Some(tree) = stack.last_mut().unwrap().next() {
            tree
        } else {
            if let Frame::Sequence { ref mut idx, ref sep, .. } = *stack.last_mut().unwrap() {
                let (ref mut repeat_idx, repeat_len) = *repeats.last_mut().unwrap();
                *repeat_idx += 1;
                if *repeat_idx < repeat_len {
                    *idx = 0;
                    if let Some(sep) = sep.clone() {
                        // repeat same span, I guess
                        let prev_span = match result.last() {
                            Some(stream) => stream.trees().next().unwrap().span(),
                            None => DUMMY_SP,
                        };
                        result.push(TokenTree::Token(prev_span, sep).into());
                    }
                    continue
                }
            }

            match stack.pop().unwrap() {
                Frame::Sequence { .. } => {
                    repeats.pop();
                }
                Frame::Delimited { forest, span, .. } => {
                    if result_stack.is_empty() {
                        return TokenStream::concat(result);
                    }
                    let tree = TokenTree::Delimited(span, Delimited {
                        delim: forest.delim,
                        tts: TokenStream::concat(result).into(),
                    });
                    result = result_stack.pop().unwrap();
                    result.push(tree.into());
                }
            }
            continue
        };

        match tree {
            quoted::TokenTree::Sequence(sp, seq) => {
                // FIXME(pcwalton): Bad copy.
                match lockstep_iter_size(&quoted::TokenTree::Sequence(sp, seq.clone()),
                                         &interpolations,
                                         &repeats) {
                    LockstepIterSize::Unconstrained => {
                        cx.span_fatal(sp, /* blame macro writer */
                            "attempted to repeat an expression \
                             containing no syntax \
                             variables matched as repeating at this depth");
                    }
                    LockstepIterSize::Contradiction(ref msg) => {
                        // FIXME #2887 blame macro invoker instead
                        cx.span_fatal(sp, &msg[..]);
                    }
                    LockstepIterSize::Constraint(len, _) => {
                        if len == 0 {
                            if seq.op == quoted::KleeneOp::OneOrMore {
                                // FIXME #2887 blame invoker
                                cx.span_fatal(sp, "this must repeat at least once");
                            }
                        } else {
                            repeats.push((0, len));
                            stack.push(Frame::Sequence {
                                idx: 0,
                                sep: seq.separator.clone(),
                                forest: seq,
                            });
                        }
                    }
                }
            }
            // FIXME #2887: think about span stuff here
            quoted::TokenTree::MetaVar(mut sp, ident) => {
                if let Some(cur_matched) = lookup_cur_matched(ident, &interpolations, &repeats) {
                    if let MatchedNonterminal(ref nt) = *cur_matched {
                        if let NtTT(ref tt) = **nt {
                            result.push(tt.clone().into());
                        } else {
                            sp = sp.apply_mark(cx.current_expansion.mark);
                            let token = TokenTree::Token(sp, Token::interpolated((**nt).clone()));
                            result.push(token.into());
                        }
                    } else {
                        cx.span_fatal(sp, /* blame the macro writer */
                            &format!("variable '{}' is still repeating at this depth", ident));
                    }
                } else {
                    let ident =
                        Ident::new(ident.name, ident.span.apply_mark(cx.current_expansion.mark));
                    sp = sp.apply_mark(cx.current_expansion.mark);
                    result.push(TokenTree::Token(sp, token::Dollar).into());
                    result.push(TokenTree::Token(sp, token::Token::from_ast_ident(ident)).into());
                }
            }
            quoted::TokenTree::Delimited(mut span, delimited) => {
                span = span.apply_mark(cx.current_expansion.mark);
                stack.push(Frame::Delimited { forest: delimited, idx: 0, span: span });
                result_stack.push(mem::replace(&mut result, Vec::new()));
            }
            quoted::TokenTree::Token(sp, tok) => {
                let mut marker = Marker(cx.current_expansion.mark);
                result.push(noop_fold_tt(TokenTree::Token(sp, tok), &mut marker).into())
            }
            quoted::TokenTree::MetaVarDecl(..) => panic!("unexpected `TokenTree::MetaVarDecl"),
        }
    }
}

fn lookup_cur_matched(ident: Ident,
                      interpolations: &HashMap<Ident, Rc<NamedMatch>>,
                      repeats: &[(usize, usize)])
                      -> Option<Rc<NamedMatch>> {
    interpolations.get(&ident).map(|matched| {
        let mut matched = matched.clone();
        for &(idx, _) in repeats {
            let m = matched.clone();
            match *m {
                MatchedNonterminal(_) => break,
                MatchedSeq(ref ads, _) => matched = Rc::new(ads[idx].clone()),
            }
        }

        matched
    })
}

#[derive(Clone)]
enum LockstepIterSize {
    Unconstrained,
    Constraint(usize, Ident),
    Contradiction(String),
}

impl Add for LockstepIterSize {
    type Output = LockstepIterSize;

    fn add(self, other: LockstepIterSize) -> LockstepIterSize {
        match self {
            LockstepIterSize::Unconstrained => other,
            LockstepIterSize::Contradiction(_) => self,
            LockstepIterSize::Constraint(l_len, ref l_id) => match other {
                LockstepIterSize::Unconstrained => self.clone(),
                LockstepIterSize::Contradiction(_) => other,
                LockstepIterSize::Constraint(r_len, _) if l_len == r_len => self.clone(),
                LockstepIterSize::Constraint(r_len, r_id) => {
                    let msg = format!("inconsistent lockstep iteration: \
                                       '{}' has {} items, but '{}' has {}",
                                      l_id, l_len, r_id, r_len);
                    LockstepIterSize::Contradiction(msg)
                }
            },
        }
    }
}

fn lockstep_iter_size(tree: &quoted::TokenTree,
                      interpolations: &HashMap<Ident, Rc<NamedMatch>>,
                      repeats: &[(usize, usize)])
                      -> LockstepIterSize {
    use self::quoted::TokenTree;
    match *tree {
        TokenTree::Delimited(_, ref delimed) => {
            delimed.tts.iter().fold(LockstepIterSize::Unconstrained, |size, tt| {
                size + lockstep_iter_size(tt, interpolations, repeats)
            })
        },
        TokenTree::Sequence(_, ref seq) => {
            seq.tts.iter().fold(LockstepIterSize::Unconstrained, |size, tt| {
                size + lockstep_iter_size(tt, interpolations, repeats)
            })
        },
        TokenTree::MetaVar(_, name) | TokenTree::MetaVarDecl(_, name, _) =>
            match lookup_cur_matched(name, interpolations, repeats) {
                Some(matched) => match *matched {
                    MatchedNonterminal(_) => LockstepIterSize::Unconstrained,
                    MatchedSeq(ref ads, _) => LockstepIterSize::Constraint(ads.len(), name),
                },
                _ => LockstepIterSize::Unconstrained
            },
        TokenTree::Token(..) => LockstepIterSize::Unconstrained,
    }
}
