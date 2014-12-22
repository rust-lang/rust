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

use ast;
use ast::{TokenTree, TtDelimited, TtToken, TtSequence, Ident};
use codemap::{Span, DUMMY_SP};
use diagnostic::SpanHandler;
use ext::tt::macro_parser::{NamedMatch, MatchedSeq, MatchedNonterminal};
use parse::token::{Eof, DocComment, Interpolated, MatchNt, SubstNt};
use parse::token::{Token, NtIdent};
use parse::token;
use parse::lexer::TokenAndSpan;

use std::rc::Rc;
use std::ops::Add;
use std::collections::HashMap;

///an unzipping of `TokenTree`s
#[deriving(Clone)]
struct TtFrame {
    forest: TokenTree,
    idx: uint,
    dotdotdoted: bool,
    sep: Option<Token>,
}

#[deriving(Clone)]
pub struct TtReader<'a> {
    pub sp_diag: &'a SpanHandler,
    /// the unzipped tree:
    stack: Vec<TtFrame>,
    /* for MBE-style macro transcription */
    interpolations: HashMap<Ident, Rc<NamedMatch>>,
    repeat_idx: Vec<uint>,
    repeat_len: Vec<uint>,
    /* cached: */
    pub cur_tok: Token,
    pub cur_span: Span,
    /// Transform doc comments. Only useful in macro invocations
    pub desugar_doc_comments: bool,
}

/// This can do Macro-By-Example transcription. On the other hand, if
/// `src` contains no `TtSequence`s and `TtNonterminal`s, `interp` can (and
/// should) be none.
pub fn new_tt_reader<'a>(sp_diag: &'a SpanHandler,
                         interp: Option<HashMap<Ident, Rc<NamedMatch>>>,
                         src: Vec<ast::TokenTree> )
                         -> TtReader<'a> {
    let mut r = TtReader {
        sp_diag: sp_diag,
        stack: vec!(TtFrame {
            forest: TtSequence(DUMMY_SP, Rc::new(ast::SequenceRepetition {
                tts: src,
                // doesn't matter. This merely holds the root unzipping.
                separator: None, op: ast::ZeroOrMore, num_captures: 0
            })),
            idx: 0,
            dotdotdoted: false,
            sep: None,
        }),
        interpolations: match interp { /* just a convenience */
            None => HashMap::new(),
            Some(x) => x,
        },
        repeat_idx: Vec::new(),
        repeat_len: Vec::new(),
        desugar_doc_comments: false,
        /* dummy values, never read: */
        cur_tok: token::Eof,
        cur_span: DUMMY_SP,
    };
    tt_next_token(&mut r); /* get cur_tok and cur_span set up */
    r
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

#[deriving(Clone)]
enum LockstepIterSize {
    LisUnconstrained,
    LisConstraint(uint, Ident),
    LisContradiction(String),
}

// NOTE(stage0): Remove impl after a snapshot
#[cfg(stage0)]
impl Add<LockstepIterSize, LockstepIterSize> for LockstepIterSize {
    fn add(&self, other: &LockstepIterSize) -> LockstepIterSize {
        match *self {
            LisUnconstrained => other.clone(),
            LisContradiction(_) => self.clone(),
            LisConstraint(l_len, l_id) => match *other {
                LisUnconstrained => self.clone(),
                LisContradiction(_) => other.clone(),
                LisConstraint(r_len, _) if l_len == r_len => self.clone(),
                LisConstraint(r_len, r_id) => {
                    let l_n = token::get_ident(l_id);
                    let r_n = token::get_ident(r_id);
                    LisContradiction(format!("inconsistent lockstep iteration: \
                                              '{}' has {} items, but '{}' has {}",
                                              l_n, l_len, r_n, r_len).to_string())
                }
            },
        }
    }
}

#[cfg(not(stage0))]  // NOTE(stage0): Remove cfg after a snapshot
impl Add<LockstepIterSize, LockstepIterSize> for LockstepIterSize {
    fn add(self, other: LockstepIterSize) -> LockstepIterSize {
        match self {
            LisUnconstrained => other,
            LisContradiction(_) => self,
            LisConstraint(l_len, ref l_id) => match other {
                LisUnconstrained => self.clone(),
                LisContradiction(_) => other,
                LisConstraint(r_len, _) if l_len == r_len => self.clone(),
                LisConstraint(r_len, r_id) => {
                    let l_n = token::get_ident(l_id.clone());
                    let r_n = token::get_ident(r_id);
                    LisContradiction(format!("inconsistent lockstep iteration: \
                                              '{}' has {} items, but '{}' has {}",
                                              l_n, l_len, r_n, r_len).to_string())
                }
            },
        }
    }
}

fn lockstep_iter_size(t: &TokenTree, r: &TtReader) -> LockstepIterSize {
    match *t {
        TtDelimited(_, ref delimed) => {
            delimed.tts.iter().fold(LisUnconstrained, |size, tt| {
                size + lockstep_iter_size(tt, r)
            })
        },
        TtSequence(_, ref seq) => {
            seq.tts.iter().fold(LisUnconstrained, |size, tt| {
                size + lockstep_iter_size(tt, r)
            })
        },
        TtToken(_, SubstNt(name, _)) | TtToken(_, MatchNt(name, _, _, _)) =>
            match lookup_cur_matched(r, name) {
                Some(matched) => match *matched {
                    MatchedNonterminal(_) => LisUnconstrained,
                    MatchedSeq(ref ads, _) => LisConstraint(ads.len(), name),
                },
                _ => LisUnconstrained
            },
        TtToken(..) => LisUnconstrained,
    }
}

/// Return the next token from the TtReader.
/// EFFECT: advances the reader's token field
pub fn tt_next_token(r: &mut TtReader) -> TokenAndSpan {
    // FIXME(pcwalton): Bad copy?
    let ret_val = TokenAndSpan {
        tok: r.cur_tok.clone(),
        sp: r.cur_span.clone(),
    };
    loop {
        let should_pop = match r.stack.last() {
            None => {
                assert_eq!(ret_val.tok, token::Eof);
                return ret_val;
            }
            Some(frame) => {
                if frame.idx < frame.forest.len() {
                    break;
                }
                !frame.dotdotdoted ||
                    *r.repeat_idx.last().unwrap() == *r.repeat_len.last().unwrap() - 1
            }
        };

        /* done with this set; pop or repeat? */
        if should_pop {
            let prev = r.stack.pop().unwrap();
            match r.stack.last_mut() {
                None => {
                    r.cur_tok = token::Eof;
                    return ret_val;
                }
                Some(frame) => {
                    frame.idx += 1;
                }
            }
            if prev.dotdotdoted {
                r.repeat_idx.pop();
                r.repeat_len.pop();
            }
        } else { /* repeat */
            *r.repeat_idx.last_mut().unwrap() += 1u;
            r.stack.last_mut().unwrap().idx = 0;
            match r.stack.last().unwrap().sep.clone() {
                Some(tk) => {
                    r.cur_tok = tk; /* repeat same span, I guess */
                    return ret_val;
                }
                None => {}
            }
        }
    }
    loop { /* because it's easiest, this handles `TtDelimited` not starting
              with a `TtToken`, even though it won't happen */
        let t = {
            let frame = r.stack.last().unwrap();
            // FIXME(pcwalton): Bad copy.
            frame.forest.get_tt(frame.idx)
        };
        match t {
            TtSequence(sp, seq) => {
                // FIXME(pcwalton): Bad copy.
                match lockstep_iter_size(&TtSequence(sp, seq.clone()),
                                         r) {
                    LisUnconstrained => {
                        r.sp_diag.span_fatal(
                            sp.clone(), /* blame macro writer */
                            "attempted to repeat an expression \
                             containing no syntax \
                             variables matched as repeating at this depth");
                    }
                    LisContradiction(ref msg) => {
                        // FIXME #2887 blame macro invoker instead
                        r.sp_diag.span_fatal(sp.clone(), msg.as_slice());
                    }
                    LisConstraint(len, _) => {
                        if len == 0 {
                            if seq.op == ast::OneOrMore {
                                // FIXME #2887 blame invoker
                                r.sp_diag.span_fatal(sp.clone(),
                                                     "this must repeat at least once");
                            }

                            r.stack.last_mut().unwrap().idx += 1;
                            return tt_next_token(r);
                        }
                        r.repeat_len.push(len);
                        r.repeat_idx.push(0);
                        r.stack.push(TtFrame {
                            idx: 0,
                            dotdotdoted: true,
                            sep: seq.separator.clone(),
                            forest: TtSequence(sp, seq),
                        });
                    }
                }
            }
            // FIXME #2887: think about span stuff here
            TtToken(sp, SubstNt(ident, namep)) => {
                match lookup_cur_matched(r, ident) {
                    None => {
                        r.stack.push(TtFrame {
                            forest: TtToken(sp, SubstNt(ident, namep)),
                            idx: 0,
                            dotdotdoted: false,
                            sep: None
                        });
                        // this can't be 0 length, just like TtDelimited
                    }
                    Some(cur_matched) => {
                        r.stack.last_mut().unwrap().idx += 1;
                        match *cur_matched {
                            // sidestep the interpolation tricks for ident because
                            // (a) idents can be in lots of places, so it'd be a pain
                            // (b) we actually can, since it's a token.
                            MatchedNonterminal(NtIdent(box sn, b)) => {
                                r.cur_span = sp;
                                r.cur_tok = token::Ident(sn, b);
                                return ret_val;
                            }
                            MatchedNonterminal(ref other_whole_nt) => {
                                // FIXME(pcwalton): Bad copy.
                                r.cur_span = sp;
                                r.cur_tok = token::Interpolated((*other_whole_nt).clone());
                                return ret_val;
                            }
                            MatchedSeq(..) => {
                                r.sp_diag.span_fatal(
                                    r.cur_span, /* blame the macro writer */
                                    format!("variable '{}' is still repeating at this depth",
                                            token::get_ident(ident)).as_slice());
                            }
                        }
                    }
                }
            }
            // TtDelimited or any token that can be unzipped
            seq @ TtDelimited(..) | seq @ TtToken(_, MatchNt(..)) => {
                // do not advance the idx yet
                r.stack.push(TtFrame {
                   forest: seq,
                   idx: 0,
                   dotdotdoted: false,
                   sep: None
                });
                // if this could be 0-length, we'd need to potentially recur here
            }
            TtToken(sp, DocComment(name)) if r.desugar_doc_comments => {
                r.stack.push(TtFrame {
                   forest: TtToken(sp, DocComment(name)),
                   idx: 0,
                   dotdotdoted: false,
                   sep: None
                });
            }
            TtToken(sp, tok) => {
                r.cur_span = sp;
                r.cur_tok = tok;
                r.stack.last_mut().unwrap().idx += 1;
                return ret_val;
            }
        }
    }
}
