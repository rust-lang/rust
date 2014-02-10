// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use ast;
use ast::{TokenTree, TTDelim, TTTok, TTSeq, TTNonterminal, Ident};
use codemap::{Span, DUMMY_SP};
use diagnostic::SpanHandler;
use ext::tt::macro_parser::{NamedMatch, MatchedSeq, MatchedNonterminal};
use parse::token::{EOF, INTERPOLATED, IDENT, Token, NtIdent};
use parse::token;
use parse::lexer::TokenAndSpan;

use std::cell::{Cell, RefCell};
use std::hashmap::HashMap;
use std::option;

///an unzipping of `TokenTree`s
struct TtFrame {
    forest: @~[ast::TokenTree],
    idx: Cell<uint>,
    dotdotdoted: bool,
    sep: Option<Token>,
    up: Option<@TtFrame>,
}

pub struct TtReader {
    sp_diag: @SpanHandler,
    // the unzipped tree:
    priv stack: RefCell<@TtFrame>,
    /* for MBE-style macro transcription */
    priv interpolations: RefCell<HashMap<Ident, @NamedMatch>>,
    priv repeat_idx: RefCell<~[uint]>,
    priv repeat_len: RefCell<~[uint]>,
    /* cached: */
    cur_tok: RefCell<Token>,
    cur_span: RefCell<Span>,
}

/** This can do Macro-By-Example transcription. On the other hand, if
 *  `src` contains no `TTSeq`s and `TTNonterminal`s, `interp` can (and
 *  should) be none. */
pub fn new_tt_reader(sp_diag: @SpanHandler,
                     interp: Option<HashMap<Ident, @NamedMatch>>,
                     src: ~[ast::TokenTree])
                     -> TtReader {
    let r = TtReader {
        sp_diag: sp_diag,
        stack: RefCell::new(@TtFrame {
            forest: @src,
            idx: Cell::new(0u),
            dotdotdoted: false,
            sep: None,
            up: option::None
        }),
        interpolations: match interp { /* just a convienience */
            None => RefCell::new(HashMap::new()),
            Some(x) => RefCell::new(x),
        },
        repeat_idx: RefCell::new(~[]),
        repeat_len: RefCell::new(~[]),
        /* dummy values, never read: */
        cur_tok: RefCell::new(EOF),
        cur_span: RefCell::new(DUMMY_SP),
    };
    tt_next_token(&r); /* get cur_tok and cur_span set up */
    return r;
}

fn dup_tt_frame(f: @TtFrame) -> @TtFrame {
    @TtFrame {
        forest: @(*f.forest).clone(),
        idx: f.idx.clone(),
        dotdotdoted: f.dotdotdoted,
        sep: f.sep.clone(),
        up: match f.up {
            Some(up_frame) => Some(dup_tt_frame(up_frame)),
            None => None
        }
    }
}

pub fn dup_tt_reader(r: &TtReader) -> TtReader {
    TtReader {
        sp_diag: r.sp_diag,
        stack: RefCell::new(dup_tt_frame(r.stack.get())),
        repeat_idx: r.repeat_idx.clone(),
        repeat_len: r.repeat_len.clone(),
        cur_tok: r.cur_tok.clone(),
        cur_span: r.cur_span.clone(),
        interpolations: r.interpolations.clone(),
    }
}


fn lookup_cur_matched_by_matched(r: &TtReader, start: @NamedMatch)
                                 -> @NamedMatch {
    fn red(ad: @NamedMatch, idx: &uint) -> @NamedMatch {
        match *ad {
            MatchedNonterminal(_) => {
                // end of the line; duplicate henceforth
                ad
            }
            MatchedSeq(ref ads, _) => ads[*idx]
        }
    }
    let repeat_idx = r.repeat_idx.borrow();
    repeat_idx.get().iter().fold(start, red)
}

fn lookup_cur_matched(r: &TtReader, name: Ident) -> @NamedMatch {
    let matched_opt = {
        let interpolations = r.interpolations.borrow();
        interpolations.get().find_copy(&name)
    };
    match matched_opt {
        Some(s) => lookup_cur_matched_by_matched(r, s),
        None => {
            let name_string = token::get_ident(name.name);
            r.sp_diag.span_fatal(r.cur_span.get(),
                                 format!("unknown macro variable `{}`",
                                         name_string.get()));
        }
    }
}

#[deriving(Clone)]
enum LockstepIterSize {
    LisUnconstrained,
    LisConstraint(uint, Ident),
    LisContradiction(~str),
}

fn lis_merge(lhs: LockstepIterSize, rhs: LockstepIterSize) -> LockstepIterSize {
    match lhs {
        LisUnconstrained => rhs.clone(),
        LisContradiction(_) => lhs.clone(),
        LisConstraint(l_len, ref l_id) => match rhs {
            LisUnconstrained => lhs.clone(),
            LisContradiction(_) => rhs.clone(),
            LisConstraint(r_len, _) if l_len == r_len => lhs.clone(),
            LisConstraint(r_len, ref r_id) => {
                let l_n = token::get_ident(l_id.name);
                let r_n = token::get_ident(r_id.name);
                LisContradiction(format!("inconsistent lockstep iteration: \
                                          '{}' has {} items, but '{}' has {}",
                                          l_n.get(), l_len, r_n.get(), r_len))
            }
        }
    }
}

fn lockstep_iter_size(t: &TokenTree, r: &TtReader) -> LockstepIterSize {
    match *t {
        TTDelim(ref tts) | TTSeq(_, ref tts, _, _) => {
            tts.iter().fold(LisUnconstrained, |lis, tt| {
                lis_merge(lis, lockstep_iter_size(tt, r))
            })
        }
        TTTok(..) => LisUnconstrained,
        TTNonterminal(_, name) => match *lookup_cur_matched(r, name) {
            MatchedNonterminal(_) => LisUnconstrained,
            MatchedSeq(ref ads, _) => LisConstraint(ads.len(), name)
        }
    }
}

// return the next token from the TtReader.
// EFFECT: advances the reader's token field
pub fn tt_next_token(r: &TtReader) -> TokenAndSpan {
    // FIXME(pcwalton): Bad copy?
    let ret_val = TokenAndSpan {
        tok: r.cur_tok.get(),
        sp: r.cur_span.get(),
    };
    loop {
        {
            let mut stack = r.stack.borrow_mut();
            if stack.get().idx.get() < stack.get().forest.len() {
                break;
            }
        }

        /* done with this set; pop or repeat? */
        if !r.stack.get().dotdotdoted || {
                let repeat_idx = r.repeat_idx.borrow();
                let repeat_len = r.repeat_len.borrow();
                *repeat_idx.get().last().unwrap() ==
                *repeat_len.get().last().unwrap() - 1
            } {

            match r.stack.get().up {
              None => {
                r.cur_tok.set(EOF);
                return ret_val;
              }
              Some(tt_f) => {
                if r.stack.get().dotdotdoted {
                    {
                        let mut repeat_idx = r.repeat_idx.borrow_mut();
                        let mut repeat_len = r.repeat_len.borrow_mut();
                        repeat_idx.get().pop().unwrap();
                        repeat_len.get().pop().unwrap();
                    }
                }

                r.stack.set(tt_f);
                r.stack.get().idx.set(r.stack.get().idx.get() + 1u);
              }
            }

        } else { /* repeat */
            r.stack.get().idx.set(0u);
            {
                let mut repeat_idx = r.repeat_idx.borrow_mut();
                repeat_idx.get()[repeat_idx.get().len() - 1u] += 1u;
            }
            match r.stack.get().sep.clone() {
              Some(tk) => {
                r.cur_tok.set(tk); /* repeat same span, I guess */
                return ret_val;
              }
              None => ()
            }
        }
    }
    loop { /* because it's easiest, this handles `TTDelim` not starting
    with a `TTTok`, even though it won't happen */
        // FIXME(pcwalton): Bad copy.
        match r.stack.get().forest[r.stack.get().idx.get()].clone() {
          TTDelim(tts) => {
            r.stack.set(@TtFrame {
                forest: tts,
                idx: Cell::new(0u),
                dotdotdoted: false,
                sep: None,
                up: option::Some(r.stack.get())
            });
            // if this could be 0-length, we'd need to potentially recur here
          }
          TTTok(sp, tok) => {
            r.cur_span.set(sp);
            r.cur_tok.set(tok);
            r.stack.get().idx.set(r.stack.get().idx.get() + 1u);
            return ret_val;
          }
          TTSeq(sp, tts, sep, zerok) => {
            // FIXME(pcwalton): Bad copy.
            let t = TTSeq(sp, tts, sep.clone(), zerok);
            match lockstep_iter_size(&t, r) {
              LisUnconstrained => {
                r.sp_diag.span_fatal(
                    sp, /* blame macro writer */
                      "attempted to repeat an expression \
                       containing no syntax \
                       variables matched as repeating at this depth");
                  }
                  LisContradiction(ref msg) => {
                      /* FIXME #2887 blame macro invoker instead*/
                      r.sp_diag.span_fatal(sp, (*msg));
                  }
                  LisConstraint(len, _) => {
                    if len == 0 {
                      if !zerok {
                        r.sp_diag.span_fatal(sp, /* FIXME #2887 blame invoker
                        */
                                             "this must repeat at least \
                                              once");
                          }

                    r.stack.get().idx.set(r.stack.get().idx.get() + 1u);
                    return tt_next_token(r);
                } else {
                    {
                        let mut repeat_idx = r.repeat_idx.borrow_mut();
                        let mut repeat_len = r.repeat_len.borrow_mut();
                        repeat_len.get().push(len);
                        repeat_idx.get().push(0u);
                        r.stack.set(@TtFrame {
                            forest: tts,
                            idx: Cell::new(0u),
                            dotdotdoted: true,
                            sep: sep,
                            up: Some(r.stack.get())
                        });
                    }
                }
              }
            }
          }
          // FIXME #2887: think about span stuff here
          TTNonterminal(sp, ident) => {
            match *lookup_cur_matched(r, ident) {
              /* sidestep the interpolation tricks for ident because
              (a) idents can be in lots of places, so it'd be a pain
              (b) we actually can, since it's a token. */
              MatchedNonterminal(NtIdent(~sn,b)) => {
                r.cur_span.set(sp);
                r.cur_tok.set(IDENT(sn,b));
                r.stack.get().idx.set(r.stack.get().idx.get() + 1u);
                return ret_val;
              }
              MatchedNonterminal(ref other_whole_nt) => {
                // FIXME(pcwalton): Bad copy.
                r.cur_span.set(sp);
                r.cur_tok.set(INTERPOLATED((*other_whole_nt).clone()));
                r.stack.get().idx.set(r.stack.get().idx.get() + 1u);
                return ret_val;
              }
              MatchedSeq(..) => {
                let string = token::get_ident(ident.name);
                r.sp_diag.span_fatal(
                    r.cur_span.get(), /* blame the macro writer */
                    format!("variable '{}' is still repeating at this depth",
                            string.get()));
              }
            }
          }
        }
    }

}
