// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Earley-like parser for macros.

use ast;
use ast::{Matcher, MatchTok, MatchSeq, MatchNonterminal, Ident};
use codemap::{BytePos, mk_sp};
use codemap;
use parse::lexer::*; //resolve bug?
use parse::ParseSess;
use parse::attr::ParserAttr;
use parse::parser::{LifetimeAndTypesWithoutColons, Parser};
use parse::token::{Token, EOF, Nonterminal};
use parse::token;

use std::rc::Rc;
use std::gc::GC;
use std::collections::HashMap;

/* This is an Earley-like parser, without support for in-grammar nonterminals,
only by calling out to the main rust parser for named nonterminals (which it
commits to fully when it hits one in a grammar). This means that there are no
completer or predictor rules, and therefore no need to store one column per
token: instead, there's a set of current Earley items and a set of next
ones. Instead of NTs, we have a special case for Kleene star. The big-O, in
pathological cases, is worse than traditional Earley parsing, but it's an
easier fit for Macro-by-Example-style rules, and I think the overhead is
lower. (In order to prevent the pathological case, we'd need to lazily
construct the resulting `NamedMatch`es at the very end. It'd be a pain,
and require more memory to keep around old items, but it would also save
overhead)*/

/* Quick intro to how the parser works:

A 'position' is a dot in the middle of a matcher, usually represented as a
dot. For example `· a $( a )* a b` is a position, as is `a $( · a )* a b`.

The parser walks through the input a character at a time, maintaining a list
of items consistent with the current position in the input string: `cur_eis`.

As it processes them, it fills up `eof_eis` with items that would be valid if
the macro invocation is now over, `bb_eis` with items that are waiting on
a Rust nonterminal like `$e:expr`, and `next_eis` with items that are waiting
on the a particular token. Most of the logic concerns moving the · through the
repetitions indicated by Kleene stars. It only advances or calls out to the
real Rust parser when no `cur_eis` items remain

Example: Start parsing `a a a a b` against [· a $( a )* a b].

Remaining input: `a a a a b`
next_eis: [· a $( a )* a b]

- - - Advance over an `a`. - - -

Remaining input: `a a a b`
cur: [a · $( a )* a b]
Descend/Skip (first item).
next: [a $( · a )* a b]  [a $( a )* · a b].

- - - Advance over an `a`. - - -

Remaining input: `a a b`
cur: [a $( a · )* a b]  next: [a $( a )* a · b]
Finish/Repeat (first item)
next: [a $( a )* · a b]  [a $( · a )* a b]  [a $( a )* a · b]

- - - Advance over an `a`. - - - (this looks exactly like the last step)

Remaining input: `a b`
cur: [a $( a · )* a b]  next: [a $( a )* a · b]
Finish/Repeat (first item)
next: [a $( a )* · a b]  [a $( · a )* a b]  [a $( a )* a · b]

- - - Advance over an `a`. - - - (this looks exactly like the last step)

Remaining input: `b`
cur: [a $( a · )* a b]  next: [a $( a )* a · b]
Finish/Repeat (first item)
next: [a $( a )* · a b]  [a $( · a )* a b]

- - - Advance over a `b`. - - -

Remaining input: ``
eof: [a $( a )* a b ·]

 */


/* to avoid costly uniqueness checks, we require that `MatchSeq` always has a
nonempty body. */


#[deriving(Clone)]
pub struct MatcherPos {
    elts: Vec<ast::Matcher> , // maybe should be <'>? Need to understand regions.
    sep: Option<Token>,
    idx: uint,
    up: Option<Box<MatcherPos>>,
    matches: Vec<Vec<Rc<NamedMatch>>>,
    match_lo: uint, match_hi: uint,
    sp_lo: BytePos,
}

pub fn count_names(ms: &[Matcher]) -> uint {
    ms.iter().fold(0, |ct, m| {
        ct + match m.node {
            MatchTok(_) => 0u,
            MatchSeq(ref more_ms, _, _, _, _) => {
                count_names(more_ms.as_slice())
            }
            MatchNonterminal(_, _, _) => 1u
        }})
}

pub fn initial_matcher_pos(ms: Vec<Matcher> , sep: Option<Token>, lo: BytePos)
                           -> Box<MatcherPos> {
    let mut match_idx_hi = 0u;
    for elt in ms.iter() {
        match elt.node {
            MatchTok(_) => (),
            MatchSeq(_,_,_,_,hi) => {
                match_idx_hi = hi;       // it is monotonic...
            }
            MatchNonterminal(_,_,pos) => {
                match_idx_hi = pos+1u;  // ...so latest is highest
            }
        }
    }
    let matches = Vec::from_fn(count_names(ms.as_slice()), |_i| Vec::new());
    box MatcherPos {
        elts: ms,
        sep: sep,
        idx: 0u,
        up: None,
        matches: matches,
        match_lo: 0u,
        match_hi: match_idx_hi,
        sp_lo: lo
    }
}

// NamedMatch is a pattern-match result for a single ast::MatchNonterminal:
// so it is associated with a single ident in a parse, and all
// MatchedNonterminal's in the NamedMatch have the same nonterminal type
// (expr, item, etc). All the leaves in a single NamedMatch correspond to a
// single matcher_nonterminal in the ast::Matcher that produced it.
//
// It should probably be renamed, it has more or less exact correspondence to
// ast::match nodes, and the in-memory structure of a particular NamedMatch
// represents the match that occurred when a particular subset of an
// ast::match -- those ast::Matcher nodes leading to a single
// MatchNonterminal -- was applied to a particular token tree.
//
// The width of each MatchedSeq in the NamedMatch, and the identity of the
// MatchedNonterminal's, will depend on the token tree it was applied to: each
// MatchedSeq corresponds to a single MatchSeq in the originating
// ast::Matcher. The depth of the NamedMatch structure will therefore depend
// only on the nesting depth of ast::MatchSeq's in the originating
// ast::Matcher it was derived from.

pub enum NamedMatch {
    MatchedSeq(Vec<Rc<NamedMatch>>, codemap::Span),
    MatchedNonterminal(Nonterminal)
}

pub fn nameize(p_s: &ParseSess, ms: &[Matcher], res: &[Rc<NamedMatch>])
            -> HashMap<Ident, Rc<NamedMatch>> {
    fn n_rec(p_s: &ParseSess, m: &Matcher, res: &[Rc<NamedMatch>],
             ret_val: &mut HashMap<Ident, Rc<NamedMatch>>) {
        match *m {
          codemap::Spanned {node: MatchTok(_), .. } => (),
          codemap::Spanned {node: MatchSeq(ref more_ms, _, _, _, _), .. } => {
            for next_m in more_ms.iter() {
                n_rec(p_s, next_m, res, ret_val)
            };
          }
          codemap::Spanned {
                node: MatchNonterminal(bind_name, _, idx),
                span
          } => {
            if ret_val.contains_key(&bind_name) {
                let string = token::get_ident(bind_name);
                p_s.span_diagnostic
                   .span_fatal(span,
                               format!("duplicated bind name: {}",
                                       string.get()).as_slice())
            }
            ret_val.insert(bind_name, res[idx].clone());
          }
        }
    }
    let mut ret_val = HashMap::new();
    for m in ms.iter() { n_rec(p_s, m, res, &mut ret_val) }
    ret_val
}

pub enum ParseResult {
    Success(HashMap<Ident, Rc<NamedMatch>>),
    Failure(codemap::Span, String),
    Error(codemap::Span, String)
}

pub fn parse_or_else(sess: &ParseSess,
                     cfg: ast::CrateConfig,
                     rdr: TtReader,
                     ms: Vec<Matcher> )
                     -> HashMap<Ident, Rc<NamedMatch>> {
    match parse(sess, cfg, rdr, ms.as_slice()) {
        Success(m) => m,
        Failure(sp, str) => {
            sess.span_diagnostic.span_fatal(sp, str.as_slice())
        }
        Error(sp, str) => {
            sess.span_diagnostic.span_fatal(sp, str.as_slice())
        }
    }
}

// perform a token equality check, ignoring syntax context (that is, an unhygienic comparison)
pub fn token_name_eq(t1 : &Token, t2 : &Token) -> bool {
    match (t1,t2) {
        (&token::IDENT(id1,_),&token::IDENT(id2,_))
        | (&token::LIFETIME(id1),&token::LIFETIME(id2)) =>
            id1.name == id2.name,
        _ => *t1 == *t2
    }
}

pub fn parse(sess: &ParseSess,
             cfg: ast::CrateConfig,
             mut rdr: TtReader,
             ms: &[Matcher])
             -> ParseResult {
    let mut cur_eis = Vec::new();
    cur_eis.push(initial_matcher_pos(ms.iter()
                                       .map(|x| (*x).clone())
                                       .collect(),
                                     None,
                                     rdr.peek().sp.lo));

    loop {
        let mut bb_eis = Vec::new(); // black-box parsed by parser.rs
        let mut next_eis = Vec::new(); // or proceed normally
        let mut eof_eis = Vec::new();

        let TokenAndSpan {tok: tok, sp: sp} = rdr.peek();

        /* we append new items to this while we go */
        loop {
            let ei = match cur_eis.pop() {
                None => break, /* for each Earley Item */
                Some(ei) => ei,
            };

            let idx = ei.idx;
            let len = ei.elts.len();

            /* at end of sequence */
            if idx >= len {
                // can't move out of `match`es, so:
                if ei.up.is_some() {
                    // hack: a matcher sequence is repeating iff it has a
                    // parent (the top level is just a container)


                    // disregard separator, try to go up
                    // (remove this condition to make trailing seps ok)
                    if idx == len {
                        // pop from the matcher position

                        let mut new_pos = ei.up.clone().unwrap();

                        // update matches (the MBE "parse tree") by appending
                        // each tree as a subtree.

                        // I bet this is a perf problem: we're preemptively
                        // doing a lot of array work that will get thrown away
                        // most of the time.

                        // Only touch the binders we have actually bound
                        for idx in range(ei.match_lo, ei.match_hi) {
                            let sub = (*ei.matches.get(idx)).clone();
                            new_pos.matches
                                   .get_mut(idx)
                                   .push(Rc::new(MatchedSeq(sub, mk_sp(ei.sp_lo,
                                                                       sp.hi))));
                        }

                        new_pos.idx += 1;
                        cur_eis.push(new_pos);
                    }

                    // can we go around again?

                    // the *_t vars are workarounds for the lack of unary move
                    match ei.sep {
                      Some(ref t) if idx == len => { // we need a separator
                        // i'm conflicted about whether this should be hygienic....
                        // though in this case, if the separators are never legal
                        // idents, it shouldn't matter.
                        if token_name_eq(&tok, t) { //pass the separator
                            let mut ei_t = ei.clone();
                            ei_t.idx += 1;
                            next_eis.push(ei_t);
                        }
                      }
                      _ => { // we don't need a separator
                        let mut ei_t = ei;
                        ei_t.idx = 0;
                        cur_eis.push(ei_t);
                      }
                    }
                } else {
                    eof_eis.push(ei);
                }
            } else {
                match ei.elts.get(idx).node.clone() {
                  /* need to descend into sequence */
                  MatchSeq(ref matchers, ref sep, zero_ok,
                           match_idx_lo, match_idx_hi) => {
                    if zero_ok {
                        let mut new_ei = ei.clone();
                        new_ei.idx += 1u;
                        //we specifically matched zero repeats.
                        for idx in range(match_idx_lo, match_idx_hi) {
                            new_ei.matches
                                  .get_mut(idx)
                                  .push(Rc::new(MatchedSeq(Vec::new(), sp)));
                        }

                        cur_eis.push(new_ei);
                    }

                    let matches = Vec::from_elem(ei.matches.len(), Vec::new());
                    let ei_t = ei;
                    cur_eis.push(box MatcherPos {
                        elts: (*matchers).clone(),
                        sep: (*sep).clone(),
                        idx: 0u,
                        up: Some(ei_t),
                        matches: matches,
                        match_lo: match_idx_lo, match_hi: match_idx_hi,
                        sp_lo: sp.lo
                    });
                  }
                  MatchNonterminal(_,_,_) => { bb_eis.push(ei) }
                  MatchTok(ref t) => {
                    let mut ei_t = ei.clone();
                    if token_name_eq(t,&tok) {
                        ei_t.idx += 1;
                        next_eis.push(ei_t);
                    }
                  }
                }
            }
        }

        /* error messages here could be improved with links to orig. rules */
        if token_name_eq(&tok, &EOF) {
            if eof_eis.len() == 1u {
                let mut v = Vec::new();
                for dv in eof_eis.get_mut(0).matches.mut_iter() {
                    v.push(dv.pop().unwrap());
                }
                return Success(nameize(sess, ms, v.as_slice()));
            } else if eof_eis.len() > 1u {
                return Error(sp, "ambiguity: multiple successful parses".to_string());
            } else {
                return Failure(sp, "unexpected end of macro invocation".to_string());
            }
        } else {
            if (bb_eis.len() > 0u && next_eis.len() > 0u)
                || bb_eis.len() > 1u {
                let nts = bb_eis.iter().map(|ei| {
                    match ei.elts.get(ei.idx).node {
                      MatchNonterminal(bind, name, _) => {
                        (format!("{} ('{}')",
                                token::get_ident(name),
                                token::get_ident(bind))).to_string()
                      }
                      _ => fail!()
                    } }).collect::<Vec<String>>().connect(" or ");
                return Error(sp, format!(
                    "local ambiguity: multiple parsing options: \
                     built-in NTs {} or {} other options.",
                    nts, next_eis.len()).to_string());
            } else if bb_eis.len() == 0u && next_eis.len() == 0u {
                return Failure(sp, format!("no rules expected the token `{}`",
                            token::to_string(&tok)).to_string());
            } else if next_eis.len() > 0u {
                /* Now process the next token */
                while next_eis.len() > 0u {
                    cur_eis.push(next_eis.pop().unwrap());
                }
                rdr.next_token();
            } else /* bb_eis.len() == 1 */ {
                let mut rust_parser = Parser::new(sess, cfg.clone(), box rdr.clone());

                let mut ei = bb_eis.pop().unwrap();
                match ei.elts.get(ei.idx).node {
                  MatchNonterminal(_, name, idx) => {
                    let name_string = token::get_ident(name);
                    ei.matches.get_mut(idx).push(Rc::new(MatchedNonterminal(
                        parse_nt(&mut rust_parser, name_string.get()))));
                    ei.idx += 1u;
                  }
                  _ => fail!()
                }
                cur_eis.push(ei);

                for _ in range(0, rust_parser.tokens_consumed) {
                    let _ = rdr.next_token();
                }
            }
        }

        assert!(cur_eis.len() > 0u);
    }
}

pub fn parse_nt(p: &mut Parser, name: &str) -> Nonterminal {
    match name {
      "item" => match p.parse_item(Vec::new()) {
        Some(i) => token::NtItem(i),
        None => p.fatal("expected an item keyword")
      },
      "block" => token::NtBlock(p.parse_block()),
      "stmt" => token::NtStmt(p.parse_stmt(Vec::new())),
      "pat" => token::NtPat(p.parse_pat()),
      "expr" => token::NtExpr(p.parse_expr()),
      "ty" => token::NtTy(p.parse_ty(false /* no need to disambiguate*/)),
      // this could be handled like a token, since it is one
      "ident" => match p.token {
        token::IDENT(sn,b) => { p.bump(); token::NtIdent(box sn,b) }
        _ => {
            let token_str = token::to_string(&p.token);
            p.fatal((format!("expected ident, found {}",
                             token_str.as_slice())).as_slice())
        }
      },
      "path" => {
        token::NtPath(box p.parse_path(LifetimeAndTypesWithoutColons).path)
      }
      "meta" => token::NtMeta(p.parse_meta_item()),
      "tt" => {
        p.quote_depth += 1u; //but in theory, non-quoted tts might be useful
        let res = token::NtTT(box(GC) p.parse_token_tree());
        p.quote_depth -= 1u;
        res
      }
      "matchers" => token::NtMatchers(p.parse_matchers()),
      _ => {
          p.fatal(format!("unsupported builtin nonterminal parser: {}",
                          name).as_slice())
      }
    }
}
