// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! This is an Earley-like parser, without support for in-grammar nonterminals,
//! only by calling out to the main rust parser for named nonterminals (which it
//! commits to fully when it hits one in a grammar). This means that there are no
//! completer or predictor rules, and therefore no need to store one column per
//! token: instead, there's a set of current Earley items and a set of next
//! ones. Instead of NTs, we have a special case for Kleene star. The big-O, in
//! pathological cases, is worse than traditional Earley parsing, but it's an
//! easier fit for Macro-by-Example-style rules, and I think the overhead is
//! lower. (In order to prevent the pathological case, we'd need to lazily
//! construct the resulting `NamedMatch`es at the very end. It'd be a pain,
//! and require more memory to keep around old items, but it would also save
//! overhead)
//!
//! Quick intro to how the parser works:
//!
//! A 'position' is a dot in the middle of a matcher, usually represented as a
//! dot. For example `· a $( a )* a b` is a position, as is `a $( · a )* a b`.
//!
//! The parser walks through the input a character at a time, maintaining a list
//! of items consistent with the current position in the input string: `cur_eis`.
//!
//! As it processes them, it fills up `eof_eis` with items that would be valid if
//! the macro invocation is now over, `bb_eis` with items that are waiting on
//! a Rust nonterminal like `$e:expr`, and `next_eis` with items that are waiting
//! on a particular token. Most of the logic concerns moving the · through the
//! repetitions indicated by Kleene stars. It only advances or calls out to the
//! real Rust parser when no `cur_eis` items remain
//!
//! Example: Start parsing `a a a a b` against [· a $( a )* a b].
//!
//! Remaining input: `a a a a b`
//! next_eis: [· a $( a )* a b]
//!
//! - - - Advance over an `a`. - - -
//!
//! Remaining input: `a a a b`
//! cur: [a · $( a )* a b]
//! Descend/Skip (first item).
//! next: [a $( · a )* a b]  [a $( a )* · a b].
//!
//! - - - Advance over an `a`. - - -
//!
//! Remaining input: `a a b`
//! cur: [a $( a · )* a b]  next: [a $( a )* a · b]
//! Finish/Repeat (first item)
//! next: [a $( a )* · a b]  [a $( · a )* a b]  [a $( a )* a · b]
//!
//! - - - Advance over an `a`. - - - (this looks exactly like the last step)
//!
//! Remaining input: `a b`
//! cur: [a $( a · )* a b]  next: [a $( a )* a · b]
//! Finish/Repeat (first item)
//! next: [a $( a )* · a b]  [a $( · a )* a b]  [a $( a )* a · b]
//!
//! - - - Advance over an `a`. - - - (this looks exactly like the last step)
//!
//! Remaining input: `b`
//! cur: [a $( a · )* a b]  next: [a $( a )* a · b]
//! Finish/Repeat (first item)
//! next: [a $( a )* · a b]  [a $( · a )* a b]
//!
//! - - - Advance over a `b`. - - -
//!
//! Remaining input: ``
//! eof: [a $( a )* a b ·]

pub use self::NamedMatch::*;
pub use self::ParseResult::*;
use self::TokenTreeOrTokenTreeVec::*;

use ast;
use ast::{TokenTree, Name, Ident};
use codemap::{BytePos, mk_sp, Span, Spanned};
use codemap;
use errors::FatalError;
use parse::lexer::*; //resolve bug?
use parse::ParseSess;
use parse::parser::{LifetimeAndTypesWithoutColons, Parser};
use parse::token::{DocComment, MatchNt, SubstNt};
use parse::token::{Token, Nonterminal};
use parse::token;
use print::pprust;
use ptr::P;

use std::mem;
use std::rc::Rc;
use std::collections::HashMap;
use std::collections::hash_map::Entry::{Vacant, Occupied};

// To avoid costly uniqueness checks, we require that `MatchSeq` always has
// a nonempty body.

#[derive(Clone)]
enum TokenTreeOrTokenTreeVec {
    Tt(ast::TokenTree),
    TtSeq(Rc<Vec<ast::TokenTree>>),
}

impl TokenTreeOrTokenTreeVec {
    fn len(&self) -> usize {
        match *self {
            TtSeq(ref v) => v.len(),
            Tt(ref tt) => tt.len(),
        }
    }

    fn get_tt(&self, index: usize) -> TokenTree {
        match *self {
            TtSeq(ref v) => v[index].clone(),
            Tt(ref tt) => tt.get_tt(index),
        }
    }
}

/// an unzipping of `TokenTree`s
#[derive(Clone)]
struct MatcherTtFrame {
    elts: TokenTreeOrTokenTreeVec,
    idx: usize,
}

#[derive(Clone)]
pub struct MatcherPos {
    stack: Vec<MatcherTtFrame>,
    top_elts: TokenTreeOrTokenTreeVec,
    sep: Option<Token>,
    idx: usize,
    up: Option<Box<MatcherPos>>,
    matches: Vec<Vec<Rc<NamedMatch>>>,
    match_lo: usize,
    match_cur: usize,
    match_hi: usize,
    sp_lo: BytePos,
}

pub fn count_names(ms: &[TokenTree]) -> usize {
    ms.iter().fold(0, |count, elt| {
        count + match *elt {
            TokenTree::Sequence(_, ref seq) => {
                seq.num_captures
            }
            TokenTree::Delimited(_, ref delim) => {
                count_names(&delim.tts)
            }
            TokenTree::Token(_, MatchNt(..)) => {
                1
            }
            TokenTree::Token(_, _) => 0,
        }
    })
}

pub fn initial_matcher_pos(ms: Rc<Vec<TokenTree>>, sep: Option<Token>, lo: BytePos)
                           -> Box<MatcherPos> {
    let match_idx_hi = count_names(&ms[..]);
    let matches: Vec<_> = (0..match_idx_hi).map(|_| Vec::new()).collect();
    Box::new(MatcherPos {
        stack: vec![],
        top_elts: TtSeq(ms),
        sep: sep,
        idx: 0,
        up: None,
        matches: matches,
        match_lo: 0,
        match_cur: 0,
        match_hi: match_idx_hi,
        sp_lo: lo
    })
}

/// NamedMatch is a pattern-match result for a single token::MATCH_NONTERMINAL:
/// so it is associated with a single ident in a parse, and all
/// `MatchedNonterminal`s in the NamedMatch have the same nonterminal type
/// (expr, item, etc). Each leaf in a single NamedMatch corresponds to a
/// single token::MATCH_NONTERMINAL in the TokenTree that produced it.
///
/// The in-memory structure of a particular NamedMatch represents the match
/// that occurred when a particular subset of a matcher was applied to a
/// particular token tree.
///
/// The width of each MatchedSeq in the NamedMatch, and the identity of the
/// `MatchedNonterminal`s, will depend on the token tree it was applied to:
/// each MatchedSeq corresponds to a single TTSeq in the originating
/// token tree. The depth of the NamedMatch structure will therefore depend
/// only on the nesting depth of `ast::TTSeq`s in the originating
/// token tree it was derived from.

pub enum NamedMatch {
    MatchedSeq(Vec<Rc<NamedMatch>>, codemap::Span),
    MatchedNonterminal(Nonterminal)
}

pub fn nameize(p_s: &ParseSess, ms: &[TokenTree], res: &[Rc<NamedMatch>])
            -> ParseResult<HashMap<Name, Rc<NamedMatch>>> {
    fn n_rec(p_s: &ParseSess, m: &TokenTree, res: &[Rc<NamedMatch>],
             ret_val: &mut HashMap<Name, Rc<NamedMatch>>, idx: &mut usize)
             -> Result<(), (codemap::Span, String)> {
        match *m {
            TokenTree::Sequence(_, ref seq) => {
                for next_m in &seq.tts {
                    try!(n_rec(p_s, next_m, res, ret_val, idx))
                }
            }
            TokenTree::Delimited(_, ref delim) => {
                for next_m in &delim.tts {
                    try!(n_rec(p_s, next_m, res, ret_val, idx));
                }
            }
            TokenTree::Token(sp, MatchNt(bind_name, _, _, _)) => {
                match ret_val.entry(bind_name.name) {
                    Vacant(spot) => {
                        spot.insert(res[*idx].clone());
                        *idx += 1;
                    }
                    Occupied(..) => {
                        return Err((sp, format!("duplicated bind name: {}", bind_name)))
                    }
                }
            }
            TokenTree::Token(sp, SubstNt(..)) => {
                return Err((sp, "missing fragment specifier".to_string()))
            }
            TokenTree::Token(_, _) => (),
        }

        Ok(())
    }

    let mut ret_val = HashMap::new();
    let mut idx = 0;
    for m in ms {
        match n_rec(p_s, m, res, &mut ret_val, &mut idx) {
            Ok(_) => {},
            Err((sp, msg)) => return Error(sp, msg),
        }
    }

    Success(ret_val)
}

pub enum ParseResult<T> {
    Success(T),
    /// Arm failed to match
    Failure(codemap::Span, String),
    /// Fatal error (malformed macro?). Abort compilation.
    Error(codemap::Span, String)
}

pub type NamedParseResult = ParseResult<HashMap<Name, Rc<NamedMatch>>>;
pub type PositionalParseResult = ParseResult<Vec<Rc<NamedMatch>>>;

/// Perform a token equality check, ignoring syntax context (that is, an
/// unhygienic comparison)
pub fn token_name_eq(t1 : &Token, t2 : &Token) -> bool {
    match (t1,t2) {
        (&token::Ident(id1,_),&token::Ident(id2,_))
        | (&token::Lifetime(id1),&token::Lifetime(id2)) =>
            id1.name == id2.name,
        _ => *t1 == *t2
    }
}

pub fn parse(sess: &ParseSess,
             cfg: ast::CrateConfig,
             mut rdr: TtReader,
             ms: &[TokenTree])
             -> NamedParseResult {
    let mut cur_eis = Vec::new();
    cur_eis.push(initial_matcher_pos(Rc::new(ms.iter()
                                                .cloned()
                                                .collect()),
                                     None,
                                     rdr.peek().sp.lo));

    loop {
        let mut bb_eis = Vec::new(); // black-box parsed by parser.rs
        let mut next_eis = Vec::new(); // or proceed normally
        let mut eof_eis = Vec::new();

        let TokenAndSpan { tok, sp } = rdr.peek();

        /* we append new items to this while we go */
        loop {
            let mut ei = match cur_eis.pop() {
                None => break, /* for each Earley Item */
                Some(ei) => ei,
            };

            // When unzipped trees end, remove them
            while ei.idx >= ei.top_elts.len() {
                match ei.stack.pop() {
                    Some(MatcherTtFrame { elts, idx }) => {
                        ei.top_elts = elts;
                        ei.idx = idx + 1;
                    }
                    None => break
                }
            }

            let idx = ei.idx;
            let len = ei.top_elts.len();

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
                        for idx in ei.match_lo..ei.match_hi {
                            let sub = (ei.matches[idx]).clone();
                            (&mut new_pos.matches[idx])
                                   .push(Rc::new(MatchedSeq(sub, mk_sp(ei.sp_lo,
                                                                       sp.hi))));
                        }

                        new_pos.match_cur = ei.match_hi;
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
                                // ei_t.match_cur = ei_t.match_lo;
                                ei_t.idx += 1;
                                next_eis.push(ei_t);
                            }
                        }
                        _ => { // we don't need a separator
                            let mut ei_t = ei;
                            ei_t.match_cur = ei_t.match_lo;
                            ei_t.idx = 0;
                            cur_eis.push(ei_t);
                        }
                    }
                } else {
                    eof_eis.push(ei);
                }
            } else {
                match ei.top_elts.get_tt(idx) {
                    /* need to descend into sequence */
                    TokenTree::Sequence(sp, seq) => {
                        if seq.op == ast::ZeroOrMore {
                            let mut new_ei = ei.clone();
                            new_ei.match_cur += seq.num_captures;
                            new_ei.idx += 1;
                            //we specifically matched zero repeats.
                            for idx in ei.match_cur..ei.match_cur + seq.num_captures {
                                (&mut new_ei.matches[idx]).push(Rc::new(MatchedSeq(vec![], sp)));
                            }

                            cur_eis.push(new_ei);
                        }

                        let matches: Vec<_> = (0..ei.matches.len())
                            .map(|_| Vec::new()).collect();
                        let ei_t = ei;
                        cur_eis.push(Box::new(MatcherPos {
                            stack: vec![],
                            sep: seq.separator.clone(),
                            idx: 0,
                            matches: matches,
                            match_lo: ei_t.match_cur,
                            match_cur: ei_t.match_cur,
                            match_hi: ei_t.match_cur + seq.num_captures,
                            up: Some(ei_t),
                            sp_lo: sp.lo,
                            top_elts: Tt(TokenTree::Sequence(sp, seq)),
                        }));
                    }
                    TokenTree::Token(_, MatchNt(..)) => {
                        // Built-in nonterminals never start with these tokens,
                        // so we can eliminate them from consideration.
                        match tok {
                            token::CloseDelim(_) => {},
                            _ => bb_eis.push(ei),
                        }
                    }
                    TokenTree::Token(sp, SubstNt(..)) => {
                        return Error(sp, "missing fragment specifier".to_string())
                    }
                    seq @ TokenTree::Delimited(..) | seq @ TokenTree::Token(_, DocComment(..)) => {
                        let lower_elts = mem::replace(&mut ei.top_elts, Tt(seq));
                        let idx = ei.idx;
                        ei.stack.push(MatcherTtFrame {
                            elts: lower_elts,
                            idx: idx,
                        });
                        ei.idx = 0;
                        cur_eis.push(ei);
                    }
                    TokenTree::Token(_, ref t) => {
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
        if token_name_eq(&tok, &token::Eof) {
            if eof_eis.len() == 1 {
                let mut v = Vec::new();
                for dv in &mut (&mut eof_eis[0]).matches {
                    v.push(dv.pop().unwrap());
                }
                return nameize(sess, ms, &v[..]);
            } else if eof_eis.len() > 1 {
                return Error(sp, "ambiguity: multiple successful parses".to_string());
            } else {
                return Failure(sp, "unexpected end of macro invocation".to_string());
            }
        } else {
            if (!bb_eis.is_empty() && !next_eis.is_empty())
                || bb_eis.len() > 1 {
                let nts = bb_eis.iter().map(|ei| match ei.top_elts.get_tt(ei.idx) {
                    TokenTree::Token(_, MatchNt(bind, name, _, _)) => {
                        format!("{} ('{}')", name, bind)
                    }
                    _ => panic!()
                }).collect::<Vec<String>>().join(" or ");

                return Error(sp, format!(
                    "local ambiguity: multiple parsing options: {}",
                    match next_eis.len() {
                        0 => format!("built-in NTs {}.", nts),
                        1 => format!("built-in NTs {} or 1 other option.", nts),
                        n => format!("built-in NTs {} or {} other options.", nts, n),
                    }
                ))
            } else if bb_eis.is_empty() && next_eis.is_empty() {
                return Failure(sp, format!("no rules expected the token `{}`",
                            pprust::token_to_string(&tok)));
            } else if !next_eis.is_empty() {
                /* Now process the next token */
                while !next_eis.is_empty() {
                    cur_eis.push(next_eis.pop().unwrap());
                }
                rdr.next_token();
            } else /* bb_eis.len() == 1 */ {
                let mut rust_parser = Parser::new(sess, cfg.clone(), Box::new(rdr.clone()));

                let mut ei = bb_eis.pop().unwrap();
                match ei.top_elts.get_tt(ei.idx) {
                    TokenTree::Token(span, MatchNt(_, ident, _, _)) => {
                        let match_cur = ei.match_cur;
                        (&mut ei.matches[match_cur]).push(Rc::new(MatchedNonterminal(
                            parse_nt(&mut rust_parser, span, &ident.name.as_str()))));
                        ei.idx += 1;
                        ei.match_cur += 1;
                    }
                    _ => panic!()
                }
                cur_eis.push(ei);

                for _ in 0..rust_parser.tokens_consumed {
                    let _ = rdr.next_token();
                }
            }
        }

        assert!(!cur_eis.is_empty());
    }
}

pub fn parse_nt<'a>(p: &mut Parser<'a>, sp: Span, name: &str) -> Nonterminal {
    match name {
        "tt" => {
            p.quote_depth += 1; //but in theory, non-quoted tts might be useful
            let res: ::parse::PResult<'a, _> = p.parse_token_tree();
            let res = token::NtTT(P(panictry!(res)));
            p.quote_depth -= 1;
            return res;
        }
        _ => {}
    }
    // check at the beginning and the parser checks after each bump
    panictry!(p.check_unknown_macro_variable());
    match name {
        "item" => match panictry!(p.parse_item()) {
            Some(i) => token::NtItem(i),
            None => {
                p.fatal("expected an item keyword").emit();
                panic!(FatalError);
            }
        },
        "block" => token::NtBlock(panictry!(p.parse_block())),
        "stmt" => match panictry!(p.parse_stmt()) {
            Some(s) => token::NtStmt(s),
            None => {
                p.fatal("expected a statement").emit();
                panic!(FatalError);
            }
        },
        "pat" => token::NtPat(panictry!(p.parse_pat())),
        "expr" => token::NtExpr(panictry!(p.parse_expr())),
        "ty" => token::NtTy(panictry!(p.parse_ty())),
        // this could be handled like a token, since it is one
        "ident" => match p.token {
            token::Ident(sn,b) => {
                panictry!(p.bump());
                token::NtIdent(Box::new(Spanned::<Ident>{node: sn, span: p.span}),b)
            }
            _ => {
                let token_str = pprust::token_to_string(&p.token);
                p.fatal(&format!("expected ident, found {}",
                                 &token_str[..])).emit();
                panic!(FatalError)
            }
        },
        "path" => {
            token::NtPath(Box::new(panictry!(p.parse_path(LifetimeAndTypesWithoutColons))))
        },
        "meta" => token::NtMeta(panictry!(p.parse_meta_item())),
        _ => {
            p.span_fatal_help(sp,
                              &format!("invalid fragment specifier `{}`", name),
                              "valid fragment specifiers are `ident`, `block`, \
                               `stmt`, `expr`, `pat`, `ty`, `path`, `meta`, `tt` \
                               and `item`").emit();
            panic!(FatalError);
        }
    }
}
