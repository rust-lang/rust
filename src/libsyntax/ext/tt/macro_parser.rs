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

use ast::Ident;
use syntax_pos::{self, BytePos, mk_sp, Span};
use codemap::Spanned;
use errors::FatalError;
use parse::{Directory, ParseSess};
use parse::parser::{PathStyle, Parser};
use parse::token::{DocComment, MatchNt, SubstNt};
use parse::token::{Token, Nonterminal};
use parse::token;
use print::pprust;
use tokenstream::{self, TokenTree};
use util::small_vector::SmallVector;

use std::mem;
use std::rc::Rc;
use std::collections::HashMap;
use std::collections::hash_map::Entry::{Vacant, Occupied};

// To avoid costly uniqueness checks, we require that `MatchSeq` always has
// a nonempty body.

#[derive(Clone)]
enum TokenTreeOrTokenTreeVec {
    Tt(tokenstream::TokenTree),
    TtSeq(Vec<tokenstream::TokenTree>),
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
struct MatcherPos {
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

pub type NamedParseResult = ParseResult<HashMap<Ident, Rc<NamedMatch>>>;

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
            TokenTree::Token(..) => 0,
        }
    })
}

fn initial_matcher_pos(ms: Vec<TokenTree>, lo: BytePos) -> Box<MatcherPos> {
    let match_idx_hi = count_names(&ms[..]);
    let matches = create_matches(match_idx_hi);
    Box::new(MatcherPos {
        stack: vec![],
        top_elts: TtSeq(ms),
        sep: None,
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
    MatchedSeq(Vec<Rc<NamedMatch>>, syntax_pos::Span),
    MatchedNonterminal(Rc<Nonterminal>)
}

fn nameize<I: Iterator<Item=Rc<NamedMatch>>>(ms: &[TokenTree], mut res: I) -> NamedParseResult {
    fn n_rec<I: Iterator<Item=Rc<NamedMatch>>>(m: &TokenTree, mut res: &mut I,
             ret_val: &mut HashMap<Ident, Rc<NamedMatch>>)
             -> Result<(), (syntax_pos::Span, String)> {
        match *m {
            TokenTree::Sequence(_, ref seq) => {
                for next_m in &seq.tts {
                    n_rec(next_m, res.by_ref(), ret_val)?
                }
            }
            TokenTree::Delimited(_, ref delim) => {
                for next_m in &delim.tts {
                    n_rec(next_m, res.by_ref(), ret_val)?;
                }
            }
            TokenTree::Token(sp, MatchNt(bind_name, _)) => {
                match ret_val.entry(bind_name) {
                    Vacant(spot) => {
                        spot.insert(res.next().unwrap());
                    }
                    Occupied(..) => {
                        return Err((sp, format!("duplicated bind name: {}", bind_name)))
                    }
                }
            }
            TokenTree::Token(sp, SubstNt(..)) => {
                return Err((sp, "missing fragment specifier".to_string()))
            }
            TokenTree::Token(..) => (),
        }

        Ok(())
    }

    let mut ret_val = HashMap::new();
    for m in ms {
        match n_rec(m, res.by_ref(), &mut ret_val) {
            Ok(_) => {},
            Err((sp, msg)) => return Error(sp, msg),
        }
    }

    Success(ret_val)
}

pub enum ParseResult<T> {
    Success(T),
    /// Arm failed to match. If the second parameter is `token::Eof`, it
    /// indicates an unexpected end of macro invocation. Otherwise, it
    /// indicates that no rules expected the given token.
    Failure(syntax_pos::Span, Token),
    /// Fatal error (malformed macro?). Abort compilation.
    Error(syntax_pos::Span, String)
}

pub fn parse_failure_msg(tok: Token) -> String {
    match tok {
        token::Eof => "unexpected end of macro invocation".to_string(),
        _ => format!("no rules expected the token `{}`", pprust::token_to_string(&tok)),
    }
}

/// Perform a token equality check, ignoring syntax context (that is, an unhygienic comparison)
fn token_name_eq(t1 : &Token, t2 : &Token) -> bool {
    match (t1,t2) {
        (&token::Ident(id1),&token::Ident(id2))
        | (&token::Lifetime(id1),&token::Lifetime(id2)) =>
            id1.name == id2.name,
        _ => *t1 == *t2
    }
}

fn create_matches(len: usize) -> Vec<Vec<Rc<NamedMatch>>> {
    (0..len).into_iter().map(|_| Vec::new()).collect()
}

fn inner_parse_loop(cur_eis: &mut SmallVector<Box<MatcherPos>>,
                    next_eis: &mut Vec<Box<MatcherPos>>,
                    eof_eis: &mut SmallVector<Box<MatcherPos>>,
                    bb_eis: &mut SmallVector<Box<MatcherPos>>,
                    token: &Token, span: &syntax_pos::Span) -> ParseResult<()> {
    while let Some(mut ei) = cur_eis.pop() {
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

        // at end of sequence
        if idx >= len {
            // We are repeating iff there is a parent
            if ei.up.is_some() {
                // Disregarding the separator, add the "up" case to the tokens that should be
                // examined.
                // (remove this condition to make trailing seps ok)
                if idx == len {
                    let mut new_pos = ei.up.clone().unwrap();

                    // update matches (the MBE "parse tree") by appending
                    // each tree as a subtree.

                    // I bet this is a perf problem: we're preemptively
                    // doing a lot of array work that will get thrown away
                    // most of the time.

                    // Only touch the binders we have actually bound
                    for idx in ei.match_lo..ei.match_hi {
                        let sub = ei.matches[idx].clone();
                        new_pos.matches[idx]
                            .push(Rc::new(MatchedSeq(sub, mk_sp(ei.sp_lo,
                                                                span.hi))));
                    }

                    new_pos.match_cur = ei.match_hi;
                    new_pos.idx += 1;
                    cur_eis.push(new_pos);
                }

                // Check if we need a separator
                if idx == len && ei.sep.is_some() {
                    // We have a separator, and it is the current token.
                    if ei.sep.as_ref().map(|ref sep| token_name_eq(&token, sep)).unwrap_or(false) {
                        ei.idx += 1;
                        next_eis.push(ei);
                    }
                } else { // we don't need a separator
                    ei.match_cur = ei.match_lo;
                    ei.idx = 0;
                    cur_eis.push(ei);
                }
            } else {
                // We aren't repeating, so we must be potentially at the end of the input.
                eof_eis.push(ei);
            }
        } else {
            match ei.top_elts.get_tt(idx) {
                /* need to descend into sequence */
                TokenTree::Sequence(sp, seq) => {
                    if seq.op == tokenstream::KleeneOp::ZeroOrMore {
                        // Examine the case where there are 0 matches of this sequence
                        let mut new_ei = ei.clone();
                        new_ei.match_cur += seq.num_captures;
                        new_ei.idx += 1;
                        for idx in ei.match_cur..ei.match_cur + seq.num_captures {
                            new_ei.matches[idx].push(Rc::new(MatchedSeq(vec![], sp)));
                        }
                        cur_eis.push(new_ei);
                    }

                    // Examine the case where there is at least one match of this sequence
                    let matches = create_matches(ei.matches.len());
                    cur_eis.push(Box::new(MatcherPos {
                        stack: vec![],
                        sep: seq.separator.clone(),
                        idx: 0,
                        matches: matches,
                        match_lo: ei.match_cur,
                        match_cur: ei.match_cur,
                        match_hi: ei.match_cur + seq.num_captures,
                        up: Some(ei),
                        sp_lo: sp.lo,
                        top_elts: Tt(TokenTree::Sequence(sp, seq)),
                    }));
                }
                TokenTree::Token(_, MatchNt(..)) => {
                    // Built-in nonterminals never start with these tokens,
                    // so we can eliminate them from consideration.
                    match *token {
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
                    if token_name_eq(t, &token) {
                        ei.idx += 1;
                        next_eis.push(ei);
                    }
                }
            }
        }
    }

    Success(())
}

pub fn parse(sess: &ParseSess, tts: Vec<TokenTree>, ms: &[TokenTree], directory: Option<Directory>)
             -> NamedParseResult {
    let mut parser = Parser::new(sess, tts, directory, true);
    let mut cur_eis = SmallVector::one(initial_matcher_pos(ms.to_owned(), parser.span.lo));
    let mut next_eis = Vec::new(); // or proceed normally

    loop {
        let mut bb_eis = SmallVector::new(); // black-box parsed by parser.rs
        let mut eof_eis = SmallVector::new();
        assert!(next_eis.is_empty());

        match inner_parse_loop(&mut cur_eis, &mut next_eis, &mut eof_eis, &mut bb_eis,
                               &parser.token, &parser.span) {
            Success(_) => {},
            Failure(sp, tok) => return Failure(sp, tok),
            Error(sp, msg) => return Error(sp, msg),
        }

        // inner parse loop handled all cur_eis, so it's empty
        assert!(cur_eis.is_empty());

        /* error messages here could be improved with links to orig. rules */
        if token_name_eq(&parser.token, &token::Eof) {
            if eof_eis.len() == 1 {
                return nameize(ms, eof_eis[0].matches.iter_mut().map(|mut dv| dv.pop().unwrap()));
            } else if eof_eis.len() > 1 {
                return Error(parser.span, "ambiguity: multiple successful parses".to_string());
            } else {
                return Failure(parser.span, token::Eof);
            }
        } else if (!bb_eis.is_empty() && !next_eis.is_empty()) || bb_eis.len() > 1 {
            let nts = bb_eis.iter().map(|ei| match ei.top_elts.get_tt(ei.idx) {
                TokenTree::Token(_, MatchNt(bind, name)) => {
                    format!("{} ('{}')", name, bind)
                }
                _ => panic!()
            }).collect::<Vec<String>>().join(" or ");

            return Error(parser.span, format!(
                "local ambiguity: multiple parsing options: {}",
                match next_eis.len() {
                    0 => format!("built-in NTs {}.", nts),
                    1 => format!("built-in NTs {} or 1 other option.", nts),
                    n => format!("built-in NTs {} or {} other options.", nts, n),
                }
            ));
        } else if bb_eis.is_empty() && next_eis.is_empty() {
            return Failure(parser.span, parser.token);
        } else if !next_eis.is_empty() {
            /* Now process the next token */
            cur_eis.extend(next_eis.drain(..));
            parser.bump();
        } else /* bb_eis.len() == 1 */ {
            let mut ei = bb_eis.pop().unwrap();
            if let TokenTree::Token(span, MatchNt(_, ident)) = ei.top_elts.get_tt(ei.idx) {
                let match_cur = ei.match_cur;
                ei.matches[match_cur].push(Rc::new(MatchedNonterminal(
                            Rc::new(parse_nt(&mut parser, span, &ident.name.as_str())))));
                ei.idx += 1;
                ei.match_cur += 1;
            } else {
                unreachable!()
            }
            cur_eis.push(ei);
        }

        assert!(!cur_eis.is_empty());
    }
}

fn parse_nt<'a>(p: &mut Parser<'a>, sp: Span, name: &str) -> Nonterminal {
    match name {
        "tt" => {
            p.quote_depth += 1; //but in theory, non-quoted tts might be useful
            let tt = panictry!(p.parse_token_tree());
            p.quote_depth -= 1;
            return token::NtTT(tt);
        }
        _ => {}
    }
    // check at the beginning and the parser checks after each bump
    p.check_unknown_macro_variable();
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
        "ty" => token::NtTy(panictry!(p.parse_ty_no_plus())),
        // this could be handled like a token, since it is one
        "ident" => match p.token {
            token::Ident(sn) => {
                p.bump();
                token::NtIdent(Spanned::<Ident>{node: sn, span: p.prev_span})
            }
            _ => {
                let token_str = pprust::token_to_string(&p.token);
                p.fatal(&format!("expected ident, found {}",
                                 &token_str[..])).emit();
                panic!(FatalError)
            }
        },
        "path" => {
            token::NtPath(panictry!(p.parse_path(PathStyle::Type)))
        },
        "meta" => token::NtMeta(panictry!(p.parse_meta_item())),
        // this is not supposed to happen, since it has been checked
        // when compiling the macro.
        _ => p.span_bug(sp, "invalid fragment specifier")
    }
}
