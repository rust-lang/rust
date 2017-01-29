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
use errors::Handler;
use ext::tt::macro_parser::{NamedMatch, MatchedSeq, MatchedNonterminal};
use ext::tt::quoted;
use parse::token::{self, MatchNt, SubstNt, Token, NtIdent, NtTT};
use syntax_pos::{Span, DUMMY_SP};
use tokenstream::{TokenTree, Delimited};
use util::small_vector::SmallVector;

use std::rc::Rc;
use std::mem;
use std::ops::Add;
use std::collections::HashMap;

// An iterator over the token trees in a delimited token tree (`{ ... }`) or a sequence (`$(...)`).
enum Frame {
    Delimited {
        forest: Rc<quoted::Delimited>,
        idx: usize,
        span: Span,
    },
    Sequence {
        forest: Rc<quoted::SequenceRepetition>,
        idx: usize,
        sep: Option<Token>,
    },
}

impl Frame {
    fn new(tts: Vec<quoted::TokenTree>) -> Frame {
        let forest = Rc::new(quoted::Delimited { delim: token::NoDelim, tts: tts });
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
/// `src` contains no `TokenTree::Sequence`s, `MatchNt`s or `SubstNt`s, `interp` can
/// (and should) be None.
pub fn transcribe(sp_diag: &Handler,
                  interp: Option<HashMap<Ident, Rc<NamedMatch>>>,
                  src: Vec<quoted::TokenTree>)
                  -> Vec<TokenTree> {
    let mut stack = SmallVector::one(Frame::new(src));
    let interpolations = interp.unwrap_or_else(HashMap::new); /* just a convenience */
    let mut repeat_idx = Vec::new();
    let mut repeat_len = Vec::new();
    let mut result = Vec::new();
    let mut result_stack = Vec::new();

    loop {
        let tree = if let Some(tree) = stack.last_mut().unwrap().next() {
            tree
        } else {
            if let Frame::Sequence { ref mut idx, ref sep, .. } = *stack.last_mut().unwrap() {
                if *repeat_idx.last().unwrap() < *repeat_len.last().unwrap() - 1 {
                    *repeat_idx.last_mut().unwrap() += 1;
                    *idx = 0;
                    if let Some(sep) = sep.clone() {
                        // repeat same span, I guess
                        let prev_span = result.last().map(TokenTree::span).unwrap_or(DUMMY_SP);
                        result.push(TokenTree::Token(prev_span, sep));
                    }
                    continue
                }
            }

            match stack.pop().unwrap() {
                Frame::Sequence { .. } => {
                    repeat_idx.pop();
                    repeat_len.pop();
                }
                Frame::Delimited { forest, span, .. } => {
                    if result_stack.is_empty() {
                        return result;
                    }
                    let tree = TokenTree::Delimited(span, Rc::new(Delimited {
                        delim: forest.delim,
                        tts: result,
                    }));
                    result = result_stack.pop().unwrap();
                    result.push(tree);
                }
            }
            continue
        };

        match tree {
            quoted::TokenTree::Sequence(sp, seq) => {
                // FIXME(pcwalton): Bad copy.
                match lockstep_iter_size(&quoted::TokenTree::Sequence(sp, seq.clone()),
                                         &interpolations,
                                         &repeat_idx) {
                    LockstepIterSize::Unconstrained => {
                        panic!(sp_diag.span_fatal(
                            sp.clone(), /* blame macro writer */
                            "attempted to repeat an expression \
                             containing no syntax \
                             variables matched as repeating at this depth"));
                    }
                    LockstepIterSize::Contradiction(ref msg) => {
                        // FIXME #2887 blame macro invoker instead
                        panic!(sp_diag.span_fatal(sp.clone(), &msg[..]));
                    }
                    LockstepIterSize::Constraint(len, _) => {
                        if len == 0 {
                            if seq.op == quoted::KleeneOp::OneOrMore {
                                // FIXME #2887 blame invoker
                                panic!(sp_diag.span_fatal(sp.clone(),
                                                          "this must repeat at least once"));
                            }
                        } else {
                            repeat_len.push(len);
                            repeat_idx.push(0);
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
            quoted::TokenTree::Token(sp, SubstNt(ident)) => {
                match lookup_cur_matched(ident, &interpolations, &repeat_idx) {
                    None => result.push(TokenTree::Token(sp, SubstNt(ident))),
                    Some(cur_matched) => if let MatchedNonterminal(ref nt) = *cur_matched {
                        match **nt {
                            // sidestep the interpolation tricks for ident because
                            // (a) idents can be in lots of places, so it'd be a pain
                            // (b) we actually can, since it's a token.
                            NtIdent(ref sn) => {
                                result.push(TokenTree::Token(sn.span, token::Ident(sn.node)));
                            }
                            NtTT(ref tt) => result.push(tt.clone()),
                            _ => {
                                // FIXME(pcwalton): Bad copy
                                result.push(TokenTree::Token(sp, token::Interpolated(nt.clone())));
                            }
                        }
                    } else {
                        panic!(sp_diag.span_fatal(
                            sp, /* blame the macro writer */
                            &format!("variable '{}' is still repeating at this depth", ident)));
                    }
                }
            }
            quoted::TokenTree::Delimited(span, delimited) => {
                stack.push(Frame::Delimited { forest: delimited, idx: 0, span: span });
                result_stack.push(mem::replace(&mut result, Vec::new()));
            }
            quoted::TokenTree::Token(span, tok) => result.push(TokenTree::Token(span, tok)),
        }
    }
}

fn lookup_cur_matched(ident: Ident,
                      interpolations: &HashMap<Ident, Rc<NamedMatch>>,
                      repeat_idx: &[usize])
                      -> Option<Rc<NamedMatch>> {
    interpolations.get(&ident).map(|matched| {
        repeat_idx.iter().fold(matched.clone(), |ad, idx| {
            match *ad {
                MatchedNonterminal(_) => {
                    // end of the line; duplicate henceforth
                    ad.clone()
                }
                MatchedSeq(ref ads, _) => ads[*idx].clone()
            }
        })
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
                      repeat_idx: &[usize])
                      -> LockstepIterSize {
    use self::quoted::TokenTree;
    match *tree {
        TokenTree::Delimited(_, ref delimed) => {
            delimed.tts.iter().fold(LockstepIterSize::Unconstrained, |size, tt| {
                size + lockstep_iter_size(tt, interpolations, repeat_idx)
            })
        },
        TokenTree::Sequence(_, ref seq) => {
            seq.tts.iter().fold(LockstepIterSize::Unconstrained, |size, tt| {
                size + lockstep_iter_size(tt, interpolations, repeat_idx)
            })
        },
        TokenTree::Token(_, SubstNt(name)) | TokenTree::Token(_, MatchNt(name, _)) =>
            match lookup_cur_matched(name, interpolations, repeat_idx) {
                Some(matched) => match *matched {
                    MatchedNonterminal(_) => LockstepIterSize::Unconstrained,
                    MatchedSeq(ref ads, _) => LockstepIterSize::Constraint(ads.len(), name),
                },
                _ => LockstepIterSize::Unconstrained
            },
        TokenTree::Token(..) => LockstepIterSize::Unconstrained,
    }
}
