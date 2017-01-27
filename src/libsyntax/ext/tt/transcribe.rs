// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
use self::LockstepIterSize::*;

use ast::Ident;
use errors::Handler;
use ext::tt::macro_parser::{NamedMatch, MatchedSeq, MatchedNonterminal};
use parse::token::{self, MatchNt, SubstNt, Token, NtIdent, NtTT};
use syntax_pos::{Span, DUMMY_SP};
use tokenstream::{self, TokenTree, Delimited, SequenceRepetition};
use util::small_vector::SmallVector;

use std::rc::Rc;
use std::ops::Add;
use std::collections::HashMap;

// An iterator over the token trees in a delimited token tree (`{ ... }`) or a sequence (`$(...)`).
enum Frame {
    Delimited {
        forest: Rc<Delimited>,
        idx: usize,
        span: Span,
    },
    MatchNt {
        name: Ident,
        kind: Ident,
        idx: usize,
        span: Span,
    },
    Sequence {
        forest: Rc<SequenceRepetition>,
        idx: usize,
        sep: Option<Token>,
    },
}

impl Iterator for Frame {
    type Item = TokenTree;

    fn next(&mut self) -> Option<TokenTree> {
        match *self {
            Frame::Delimited { ref forest, ref mut idx, span } => {
                *idx += 1;
                if *idx == forest.delim.len() {
                    Some(forest.open_tt(span))
                } else if let Some(tree) = forest.tts.get(*idx - forest.delim.len() - 1) {
                    Some(tree.clone())
                } else if *idx == forest.tts.len() + 2 * forest.delim.len() {
                    Some(forest.close_tt(span))
                } else {
                    None
                }
            }
            Frame::Sequence { ref forest, ref mut idx, .. } => {
                *idx += 1;
                forest.tts.get(*idx - 1).cloned()
            }
            Frame::MatchNt { ref mut idx, name, kind, span } => {
                *idx += 1;
                match *idx {
                    1 => Some(TokenTree::Token(span, token::SubstNt(name))),
                    2 => Some(TokenTree::Token(span, token::Colon)),
                    3 => Some(TokenTree::Token(span, token::Ident(kind))),
                    _ => None,
                }
            }
        }
    }
}

struct TtReader<'a> {
    sp_diag: &'a Handler,
    /// the unzipped tree:
    stack: SmallVector<Frame>,
    /* for MBE-style macro transcription */
    interpolations: HashMap<Ident, Rc<NamedMatch>>,

    repeat_idx: Vec<usize>,
    repeat_len: Vec<usize>,
}

/// This can do Macro-By-Example transcription. On the other hand, if
/// `src` contains no `TokenTree::Sequence`s, `MatchNt`s or `SubstNt`s, `interp` can
/// (and should) be None.
pub fn transcribe(sp_diag: &Handler,
                  interp: Option<HashMap<Ident, Rc<NamedMatch>>>,
                  src: Vec<tokenstream::TokenTree>)
                  -> Vec<TokenTree> {
    let mut r = TtReader {
        sp_diag: sp_diag,
        stack: SmallVector::one(Frame::Delimited {
            forest: Rc::new(tokenstream::Delimited { delim: token::NoDelim, tts: src }),
            idx: 0,
            span: DUMMY_SP,
        }),
        interpolations: match interp { /* just a convenience */
            None => HashMap::new(),
            Some(x) => x,
        },
        repeat_idx: Vec::new(),
        repeat_len: Vec::new(),
    };

    let mut tts = Vec::new();
    let mut prev_span = DUMMY_SP;
    while let Some(tt) = tt_next_token(&mut r, prev_span) {
        prev_span = tt.span();
        tts.push(tt);
    }
    tts
}

fn lookup_cur_matched_by_matched(r: &TtReader, start: Rc<NamedMatch>) -> Rc<NamedMatch> {
    r.repeat_idx.iter().fold(start, |ad, idx| {
        match *ad {
            MatchedNonterminal(_) => {
                // end of the line; duplicate henceforth
                ad.clone()
            }
            MatchedSeq(ref ads, _) => ads[*idx].clone()
        }
    })
}

fn lookup_cur_matched(r: &TtReader, name: Ident) -> Option<Rc<NamedMatch>> {
    let matched_opt = r.interpolations.get(&name).cloned();
    matched_opt.map(|s| lookup_cur_matched_by_matched(r, s))
}

#[derive(Clone)]
enum LockstepIterSize {
    LisUnconstrained,
    LisConstraint(usize, Ident),
    LisContradiction(String),
}

impl Add for LockstepIterSize {
    type Output = LockstepIterSize;

    fn add(self, other: LockstepIterSize) -> LockstepIterSize {
        match self {
            LisUnconstrained => other,
            LisContradiction(_) => self,
            LisConstraint(l_len, ref l_id) => match other {
                LisUnconstrained => self.clone(),
                LisContradiction(_) => other,
                LisConstraint(r_len, _) if l_len == r_len => self.clone(),
                LisConstraint(r_len, r_id) => {
                    LisContradiction(format!("inconsistent lockstep iteration: \
                                              '{}' has {} items, but '{}' has {}",
                                              l_id, l_len, r_id, r_len))
                }
            },
        }
    }
}

fn lockstep_iter_size(t: &TokenTree, r: &TtReader) -> LockstepIterSize {
    match *t {
        TokenTree::Delimited(_, ref delimed) => {
            delimed.tts.iter().fold(LisUnconstrained, |size, tt| {
                size + lockstep_iter_size(tt, r)
            })
        },
        TokenTree::Sequence(_, ref seq) => {
            seq.tts.iter().fold(LisUnconstrained, |size, tt| {
                size + lockstep_iter_size(tt, r)
            })
        },
        TokenTree::Token(_, SubstNt(name)) | TokenTree::Token(_, MatchNt(name, _)) =>
            match lookup_cur_matched(r, name) {
                Some(matched) => match *matched {
                    MatchedNonterminal(_) => LisUnconstrained,
                    MatchedSeq(ref ads, _) => LisConstraint(ads.len(), name),
                },
                _ => LisUnconstrained
            },
        TokenTree::Token(..) => LisUnconstrained,
    }
}

/// Return the next token from the TtReader.
/// EFFECT: advances the reader's token field
fn tt_next_token(r: &mut TtReader, prev_span: Span) -> Option<TokenTree> {
    loop {
        let tree = match r.stack.last_mut() {
            Some(frame) => frame.next(),
            None => return None,
        };

        let tree = if let Some(tree) = tree {
            tree
        } else {
            if let Frame::Sequence { ref mut idx, ref sep, .. } = *r.stack.last_mut().unwrap() {
                if *r.repeat_idx.last().unwrap() < *r.repeat_len.last().unwrap() - 1 {
                    *r.repeat_idx.last_mut().unwrap() += 1;
                    *idx = 0;
                    if let Some(sep) = sep.clone() {
                        return Some(TokenTree::Token(prev_span, sep)); // repeat same span, I guess
                    }
                    continue
                }
            }

            if let Frame::Sequence { .. } = r.stack.pop().unwrap() {
                r.repeat_idx.pop();
                r.repeat_len.pop();
            }
            continue
        };

        match tree {
            TokenTree::Sequence(sp, seq) => {
                // FIXME(pcwalton): Bad copy.
                match lockstep_iter_size(&TokenTree::Sequence(sp, seq.clone()),
                                         r) {
                    LisUnconstrained => {
                        panic!(r.sp_diag.span_fatal(
                            sp.clone(), /* blame macro writer */
                            "attempted to repeat an expression \
                             containing no syntax \
                             variables matched as repeating at this depth"));
                    }
                    LisContradiction(ref msg) => {
                        // FIXME #2887 blame macro invoker instead
                        panic!(r.sp_diag.span_fatal(sp.clone(), &msg[..]));
                    }
                    LisConstraint(len, _) => {
                        if len == 0 {
                            if seq.op == tokenstream::KleeneOp::OneOrMore {
                                // FIXME #2887 blame invoker
                                panic!(r.sp_diag.span_fatal(sp.clone(),
                                                     "this must repeat at least once"));
                            }

                            return tt_next_token(r, prev_span);
                        }
                        r.repeat_len.push(len);
                        r.repeat_idx.push(0);
                        r.stack.push(Frame::Sequence {
                            idx: 0,
                            sep: seq.separator.clone(),
                            forest: seq,
                        });
                    }
                }
            }
            // FIXME #2887: think about span stuff here
            TokenTree::Token(sp, SubstNt(ident)) => {
                match lookup_cur_matched(r, ident) {
                    None => {
                        return Some(TokenTree::Token(sp, SubstNt(ident)));
                        // this can't be 0 length, just like TokenTree::Delimited
                    }
                    Some(cur_matched) => if let MatchedNonterminal(ref nt) = *cur_matched {
                        match **nt {
                            // sidestep the interpolation tricks for ident because
                            // (a) idents can be in lots of places, so it'd be a pain
                            // (b) we actually can, since it's a token.
                            NtIdent(ref sn) => {
                                return Some(TokenTree::Token(sn.span, token::Ident(sn.node)));
                            }
                            NtTT(ref tt) => return Some(tt.clone()),
                            _ => {
                                // FIXME(pcwalton): Bad copy
                                return Some(TokenTree::Token(sp, token::Interpolated(nt.clone())));
                            }
                        }
                    } else {
                        panic!(r.sp_diag.span_fatal(
                            sp, /* blame the macro writer */
                            &format!("variable '{}' is still repeating at this depth", ident)));
                    }
                }
            }
            TokenTree::Delimited(span, delimited) => {
                r.stack.push(Frame::Delimited { forest: delimited, idx: 0, span: span });
            }
            TokenTree::Token(span, MatchNt(name, kind)) => {
                r.stack.push(Frame::MatchNt { name: name, kind: kind, idx: 0, span: span });
            }
            tt @ TokenTree::Token(..) => return Some(tt),
        }
    }
}
