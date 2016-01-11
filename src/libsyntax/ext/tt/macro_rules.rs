// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use ast::{self, TokenTree};
use codemap::{Span, DUMMY_SP};
use ext::base::{DummyResult, ExtCtxt, MacResult, SyntaxExtension};
use ext::base::{NormalTT, TTMacroExpander};
use ext::tt::macro_parser::{Success, Error, Failure};
use ext::tt::macro_parser::{MatchedSeq, MatchedNonterminal};
use ext::tt::macro_parser::parse;
use parse::lexer::new_tt_reader;
use parse::parser::Parser;
use parse::token::{self, special_idents, gensym_ident, NtTT, Token};
use parse::token::Token::*;
use print;
use ptr::P;

use util::small_vector::SmallVector;

use std::cell::RefCell;
use std::collections::{HashMap};
use std::collections::hash_map::{Entry};
use std::rc::Rc;

struct ParserAnyMacro<'a> {
    parser: RefCell<Parser<'a>>,

    /// Span of the expansion site of the macro this parser is for
    site_span: Span,
    /// The ident of the macro we're parsing
    macro_ident: ast::Ident
}

impl<'a> ParserAnyMacro<'a> {
    /// Make sure we don't have any tokens left to parse, so we don't
    /// silently drop anything. `allow_semi` is so that "optional"
    /// semicolons at the end of normal expressions aren't complained
    /// about e.g. the semicolon in `macro_rules! kapow { () => {
    /// panic!(); } }` doesn't get picked up by .parse_expr(), but it's
    /// allowed to be there.
    fn ensure_complete_parse(&self, allow_semi: bool, context: &str) {
        let mut parser = self.parser.borrow_mut();
        if allow_semi && parser.token == token::Semi {
            parser.bump();
        }
        if parser.token != token::Eof {
            let token_str = parser.this_token_to_string();
            let msg = format!("macro expansion ignores token `{}` and any \
                               following",
                              token_str);
            let span = parser.span;
            let mut err = parser.diagnostic().struct_span_err(span, &msg[..]);
            let msg = format!("caused by the macro expansion here; the usage \
                               of `{}!` is likely invalid in {} context",
                               self.macro_ident, context);
            err.span_note(self.site_span, &msg[..])
               .emit();
        }
    }
}

impl<'a> MacResult for ParserAnyMacro<'a> {
    fn make_expr(self: Box<ParserAnyMacro<'a>>) -> Option<P<ast::Expr>> {
        let ret = panictry!(self.parser.borrow_mut().parse_expr());
        self.ensure_complete_parse(true, "expression");
        Some(ret)
    }
    fn make_pat(self: Box<ParserAnyMacro<'a>>) -> Option<P<ast::Pat>> {
        let ret = panictry!(self.parser.borrow_mut().parse_pat());
        self.ensure_complete_parse(false, "pattern");
        Some(ret)
    }
    fn make_items(self: Box<ParserAnyMacro<'a>>) -> Option<SmallVector<P<ast::Item>>> {
        let mut ret = SmallVector::zero();
        while let Some(item) = panictry!(self.parser.borrow_mut().parse_item()) {
            ret.push(item);
        }
        self.ensure_complete_parse(false, "item");
        Some(ret)
    }

    fn make_impl_items(self: Box<ParserAnyMacro<'a>>)
                       -> Option<SmallVector<P<ast::ImplItem>>> {
        let mut ret = SmallVector::zero();
        loop {
            let mut parser = self.parser.borrow_mut();
            match parser.token {
                token::Eof => break,
                _ => ret.push(panictry!(parser.parse_impl_item()))
            }
        }
        self.ensure_complete_parse(false, "item");
        Some(ret)
    }

    fn make_stmts(self: Box<ParserAnyMacro<'a>>)
                 -> Option<SmallVector<P<ast::Stmt>>> {
        let mut ret = SmallVector::zero();
        loop {
            let mut parser = self.parser.borrow_mut();
            match parser.token {
                token::Eof => break,
                _ => match parser.parse_stmt() {
                    Ok(maybe_stmt) => match maybe_stmt {
                        Some(stmt) => ret.push(stmt),
                        None => (),
                    },
                    Err(mut e) => {
                        e.emit();
                        break;
                    }
                }
            }
        }
        self.ensure_complete_parse(false, "statement");
        Some(ret)
    }

    fn make_ty(self: Box<ParserAnyMacro<'a>>) -> Option<P<ast::Ty>> {
        let ret = panictry!(self.parser.borrow_mut().parse_ty());
        self.ensure_complete_parse(false, "type");
        Some(ret)
    }
}

struct MacroRulesMacroExpander {
    name: ast::Ident,
    imported_from: Option<ast::Ident>,
    lhses: Vec<TokenTree>,
    rhses: Vec<TokenTree>,
    valid: bool,
}

impl TTMacroExpander for MacroRulesMacroExpander {
    fn expand<'cx>(&self,
                   cx: &'cx mut ExtCtxt,
                   sp: Span,
                   arg: &[TokenTree])
                   -> Box<MacResult+'cx> {
        if !self.valid {
            return DummyResult::any(sp);
        }
        generic_extension(cx,
                          sp,
                          self.name,
                          self.imported_from,
                          arg,
                          &self.lhses,
                          &self.rhses)
    }
}

/// Given `lhses` and `rhses`, this is the new macro we create
fn generic_extension<'cx>(cx: &'cx ExtCtxt,
                          sp: Span,
                          name: ast::Ident,
                          imported_from: Option<ast::Ident>,
                          arg: &[TokenTree],
                          lhses: &[TokenTree],
                          rhses: &[TokenTree])
                          -> Box<MacResult+'cx> {
    if cx.trace_macros() {
        println!("{}! {{ {} }}",
                 name,
                 print::pprust::tts_to_string(arg));
    }

    // Which arm's failure should we report? (the one furthest along)
    let mut best_fail_spot = DUMMY_SP;
    let mut best_fail_msg = "internal error: ran no matchers".to_string();

    for (i, lhs) in lhses.iter().enumerate() { // try each arm's matchers
        let lhs_tt = match *lhs {
            TokenTree::Delimited(_, ref delim) => &delim.tts[..],
            _ => cx.span_fatal(sp, "malformed macro lhs")
        };

        match TokenTree::parse(cx, lhs_tt, arg) {
            Success(named_matches) => {
                let rhs = match rhses[i] {
                    // ignore delimiters
                    TokenTree::Delimited(_, ref delimed) => delimed.tts.clone(),
                    _ => cx.span_fatal(sp, "malformed macro rhs"),
                };
                // rhs has holes ( `$id` and `$(...)` that need filled)
                let trncbr = new_tt_reader(&cx.parse_sess().span_diagnostic,
                                           Some(named_matches),
                                           imported_from,
                                           rhs);
                let mut p = Parser::new(cx.parse_sess(), cx.cfg(), Box::new(trncbr));
                p.check_unknown_macro_variable();
                // Let the context choose how to interpret the result.
                // Weird, but useful for X-macros.
                return Box::new(ParserAnyMacro {
                    parser: RefCell::new(p),

                    // Pass along the original expansion site and the name of the macro
                    // so we can print a useful error message if the parse of the expanded
                    // macro leaves unparsed tokens.
                    site_span: sp,
                    macro_ident: name
                })
            }
            Failure(sp, ref msg) => if sp.lo >= best_fail_spot.lo {
                best_fail_spot = sp;
                best_fail_msg = (*msg).clone();
            },
            Error(err_sp, ref msg) => {
                cx.span_fatal(err_sp.substitute_dummy(sp), &msg[..])
            }
        }
    }

     cx.span_fatal(best_fail_spot.substitute_dummy(sp), &best_fail_msg[..]);
}

// Note that macro-by-example's input is also matched against a token tree:
//                   $( $lhs:tt => $rhs:tt );+
//
// Holy self-referential!

/// Converts a `macro_rules!` invocation into a syntax extension.
pub fn compile<'cx>(cx: &'cx mut ExtCtxt,
                    def: &ast::MacroDef) -> SyntaxExtension {

    let lhs_nm =  gensym_ident("lhs");
    let rhs_nm =  gensym_ident("rhs");

    // The pattern that macro_rules matches.
    // The grammar for macro_rules! is:
    // $( $lhs:tt => $rhs:tt );+
    // ...quasiquoting this would be nice.
    // These spans won't matter, anyways
    let match_lhs_tok = MatchNt(lhs_nm, special_idents::tt, token::Plain, token::Plain);
    let match_rhs_tok = MatchNt(rhs_nm, special_idents::tt, token::Plain, token::Plain);
    let argument_gram = vec!(
        TokenTree::Sequence(DUMMY_SP,
                   Rc::new(ast::SequenceRepetition {
                       tts: vec![
                           TokenTree::Token(DUMMY_SP, match_lhs_tok),
                           TokenTree::Token(DUMMY_SP, token::FatArrow),
                           TokenTree::Token(DUMMY_SP, match_rhs_tok)],
                       separator: Some(token::Semi),
                       op: ast::OneOrMore,
                       num_captures: 2
                   })),
        //to phase into semicolon-termination instead of
        //semicolon-separation
        TokenTree::Sequence(DUMMY_SP,
                   Rc::new(ast::SequenceRepetition {
                       tts: vec![TokenTree::Token(DUMMY_SP, token::Semi)],
                       separator: None,
                       op: ast::ZeroOrMore,
                       num_captures: 0
                   })));


    // Parse the macro_rules! invocation (`none` is for no interpolations):
    let arg_reader = new_tt_reader(&cx.parse_sess().span_diagnostic,
                                   None,
                                   None,
                                   def.body.clone());

    let argument_map = match parse(cx.parse_sess(),
                                   cx.cfg(),
                                   arg_reader,
                                   &argument_gram) {
        Success(m) => m,
        Failure(sp, str) | Error(sp, str) => {
            panic!(cx.parse_sess().span_diagnostic
                     .span_fatal(sp.substitute_dummy(def.span), &str[..]));
        }
    };

    let mut valid = true;

    // Extract the arguments:
    let lhses = match **argument_map.get(&lhs_nm.name).unwrap() {
        MatchedSeq(ref s, _) => {
            s.iter().map(|m| match **m {
                MatchedNonterminal(NtTT(ref tt)) => (**tt).clone(),
                _ => cx.span_bug(def.span, "wrong-structured lhs")
            }).collect()
        }
        _ => cx.span_bug(def.span, "wrong-structured lhs")
    };

    for lhs in &lhses {
        check_lhs_nt_follows(cx, lhs, def.span);
    }

    let rhses = match **argument_map.get(&rhs_nm.name).unwrap() {
        MatchedSeq(ref s, _) => {
            s.iter().map(|m| match **m {
                MatchedNonterminal(NtTT(ref tt)) => (**tt).clone(),
                _ => cx.span_bug(def.span, "wrong-structured rhs")
            }).collect()
        }
        _ => cx.span_bug(def.span, "wrong-structured rhs")
    };

    for rhs in &rhses {
        valid &= check_rhs(cx, rhs);
    }

    let exp: Box<_> = Box::new(MacroRulesMacroExpander {
        name: def.ident,
        imported_from: def.imported_from,
        lhses: lhses,
        rhses: rhses,
        valid: valid,
    });

    NormalTT(exp, Some(def.span), def.allow_internal_unstable)
}

// why is this here? because of https://github.com/rust-lang/rust/issues/27774
fn ref_slice<A>(s: &A) -> &[A] { use std::slice::from_raw_parts; unsafe { from_raw_parts(s, 1) } }

fn check_lhs_nt_follows(cx: &mut ExtCtxt, lhs: &TokenTree, sp: Span) {
    // lhs is going to be like TokenTree::Delimited(...), where the
    // entire lhs is those tts. Or, it can be a "bare sequence", not wrapped in parens.
    match lhs {
        &TokenTree::Delimited(_, ref tts) => {
            check_matcher(cx, &tts.tts);
        },
        tt @ &TokenTree::Sequence(..) => {
            check_matcher(cx, ref_slice(tt));
        },
        _ => cx.span_err(sp, "invalid macro matcher; matchers must be contained \
                              in balanced delimiters or a repetition indicator")
    };
    // we don't abort on errors on rejection, the driver will do that for us
    // after parsing/expansion. we can report every error in every macro this way.
}

fn check_rhs(cx: &mut ExtCtxt, rhs: &TokenTree) -> bool {
    match *rhs {
        TokenTree::Delimited(..) => return true,
        _ => cx.span_err(rhs.get_span(), "macro rhs must be delimited")
    }
    false
}

// Issue 30450: when we are through a warning cycle, we can just error
// on all failure conditions and remove this struct and enum.

#[derive(Debug)]
struct OnFail {
    saw_failure: bool,
    action: OnFailAction,
}

#[derive(Copy, Clone, Debug)]
enum OnFailAction { Warn, Error, DoNothing }

impl OnFail {
    fn warn() -> OnFail { OnFail { saw_failure: false, action: OnFailAction::Warn } }
    fn error() -> OnFail { OnFail { saw_failure: false, action: OnFailAction::Error } }
    fn do_nothing() -> OnFail { OnFail { saw_failure: false, action: OnFailAction::DoNothing } }
    fn react(&mut self, cx: &mut ExtCtxt, sp: Span, msg: &str) {
        match self.action {
            OnFailAction::DoNothing => {}
            OnFailAction::Error => cx.span_err(sp, msg),
            OnFailAction::Warn => {
                cx.struct_span_warn(sp, msg)
                    .span_note(sp, "The above warning will be a hard error in the next release.")
                    .emit();
            }
        };
        self.saw_failure = true;
    }
}

fn check_matcher(cx: &mut ExtCtxt, matcher: &[TokenTree]) {
    // Issue 30450: when we are through a warning cycle, we can just
    // error on all failure conditions (and remove check_matcher_old).

    // First run the old-pass, but *only* to find out if it would have failed.
    let mut on_fail = OnFail::do_nothing();
    check_matcher_old(cx, matcher.iter(), &Eof, &mut on_fail);
    // Then run the new pass, but merely warn if the old pass accepts and new pass rejects.
    // (Note this silently accepts code if new pass accepts.)
    let mut on_fail = if on_fail.saw_failure {
        OnFail::error()
    } else {
        OnFail::warn()
    };
    check_matcher_new(cx, matcher, &mut on_fail);
}

// returns the last token that was checked, for TokenTree::Sequence.
// return value is used by recursive calls.
fn check_matcher_old<'a, I>(cx: &mut ExtCtxt, matcher: I, follow: &Token, on_fail: &mut OnFail)
-> Option<(Span, Token)> where I: Iterator<Item=&'a TokenTree> {
    use print::pprust::token_to_string;
    use std::iter::once;

    let mut last = None;

    // 2. For each token T in M:
    let mut tokens = matcher.peekable();
    while let Some(token) = tokens.next() {
        last = match *token {
            TokenTree::Token(sp, MatchNt(ref name, ref frag_spec, _, _)) => {
                // ii. If T is a simple NT, look ahead to the next token T' in
                // M. If T' is in the set FOLLOW(NT), continue. Else; reject.
                if can_be_followed_by_any(&frag_spec.name.as_str()) {
                    continue
                } else {
                    let next_token = match tokens.peek() {
                        // If T' closes a complex NT, replace T' with F
                        Some(&&TokenTree::Token(_, CloseDelim(_))) => follow.clone(),
                        Some(&&TokenTree::Token(_, ref tok)) => tok.clone(),
                        Some(&&TokenTree::Sequence(sp, _)) => {
                            // Be conservative around sequences: to be
                            // more specific, we would need to
                            // consider FIRST sets, but also the
                            // possibility that the sequence occurred
                            // zero times (in which case we need to
                            // look at the token that follows the
                            // sequence, which may itself be a sequence,
                            // and so on).
                            on_fail.react(cx, sp,
                                        &format!("`${0}:{1}` is followed by a \
                                                  sequence repetition, which is not \
                                                  allowed for `{1}` fragments",
                                                 name, frag_spec)
                                        );
                            Eof
                        },
                        // die next iteration
                        Some(&&TokenTree::Delimited(_, ref delim)) => delim.close_token(),
                        // else, we're at the end of the macro or sequence
                        None => follow.clone()
                    };

                    let tok = if let TokenTree::Token(_, ref tok) = *token {
                        tok
                    } else {
                        unreachable!()
                    };

                    // If T' is in the set FOLLOW(NT), continue. Else, reject.
                    match (&next_token, is_in_follow(cx, &next_token, &frag_spec.name.as_str())) {
                        (_, Err(msg)) => {
                            on_fail.react(cx, sp, &msg);
                            continue
                        }
                        (&Eof, _) => return Some((sp, tok.clone())),
                        (_, Ok(true)) => continue,
                        (next, Ok(false)) => {
                            on_fail.react(cx, sp, &format!("`${0}:{1}` is followed by `{2}`, which \
                                                      is not allowed for `{1}` fragments",
                                                     name, frag_spec,
                                                     token_to_string(next)));
                            continue
                        },
                    }
                }
            },
            TokenTree::Sequence(sp, ref seq) => {
                // iii. Else, T is a complex NT.
                match seq.separator {
                    // If T has the form $(...)U+ or $(...)U* for some token U,
                    // run the algorithm on the contents with F set to U. If it
                    // accepts, continue, else, reject.
                    Some(ref u) => {
                        let last = check_matcher_old(cx, seq.tts.iter(), u, on_fail);
                        match last {
                            // Since the delimiter isn't required after the last
                            // repetition, make sure that the *next* token is
                            // sane. This doesn't actually compute the FIRST of
                            // the rest of the matcher yet, it only considers
                            // single tokens and simple NTs. This is imprecise,
                            // but conservatively correct.
                            Some((span, tok)) => {
                                let fol = match tokens.peek() {
                                    Some(&&TokenTree::Token(_, ref tok)) => tok.clone(),
                                    Some(&&TokenTree::Delimited(_, ref delim)) =>
                                        delim.close_token(),
                                    Some(_) => {
                                        on_fail.react(cx, sp, "sequence repetition followed by \
                                                another sequence repetition, which is not allowed");
                                        Eof
                                    },
                                    None => Eof
                                };
                                check_matcher_old(cx, once(&TokenTree::Token(span, tok.clone())),
                                                  &fol, on_fail)
                            },
                            None => last,
                        }
                    },
                    // If T has the form $(...)+ or $(...)*, run the algorithm
                    // on the contents with F set to the token following the
                    // sequence. If it accepts, continue, else, reject.
                    None => {
                        let fol = match tokens.peek() {
                            Some(&&TokenTree::Token(_, ref tok)) => tok.clone(),
                            Some(&&TokenTree::Delimited(_, ref delim)) => delim.close_token(),
                            Some(_) => {
                                on_fail.react(cx, sp, "sequence repetition followed by another \
                                             sequence repetition, which is not allowed");
                                Eof
                            },
                            None => Eof
                        };
                        check_matcher_old(cx, seq.tts.iter(), &fol, on_fail)
                    }
                }
            },
            TokenTree::Token(..) => {
                // i. If T is not an NT, continue.
                continue
            },
            TokenTree::Delimited(_, ref tts) => {
                // if we don't pass in that close delimiter, we'll incorrectly consider the matcher
                // `{ $foo:ty }` as having a follow that isn't `RBrace`
                check_matcher_old(cx, tts.tts.iter(), &tts.close_token(), on_fail)
            }
        }
    }
    last
}

fn check_matcher_new(cx: &mut ExtCtxt, matcher: &[TokenTree], on_fail: &mut OnFail) {
    let first_sets = FirstSets::new(matcher);
    let empty_suffix = TokenSet::empty();
    check_matcher_core(cx, &first_sets, matcher, &empty_suffix, on_fail);
}

// The FirstSets for a matcher is a mapping from subsequences in the
// matcher to the FIRST set for that subsequence.
//
// This mapping is partially precomputed via a backwards scan over the
// token trees of the matcher, which provides a mapping from each
// repetition sequence to its FIRST set.
//
// (Hypothetically sequences should be uniquely identifiable via their
// spans, though perhaps that is false e.g. for macro-generated macros
// that do not try to inject artificial span information. My plan is
// to try to catch such cases ahead of time and not include them in
// the precomputed mapping.)
struct FirstSets {
    // this maps each TokenTree::Sequence `$(tt ...) SEP OP` that is uniquely identified by its
    // span in the original matcher to the First set for the inner sequence `tt ...`.
    //
    // If two sequences have the same span in a matcher, then map that
    // span to None (invalidating the mapping here and forcing the code to
    // use a slow path).
    first: HashMap<Span, Option<TokenSet>>,
}

impl FirstSets {
    fn new(tts: &[TokenTree]) -> FirstSets {
        let mut sets = FirstSets { first: HashMap::new() };
        build_recur(&mut sets, tts);
        return sets;

        // walks backward over `tts`, returning the FIRST for `tts`
        // and updating `sets` at the same time for all sequence
        // substructure we find within `tts`.
        fn build_recur(sets: &mut FirstSets, tts: &[TokenTree]) -> TokenSet {
            let mut first = TokenSet::empty();
            for tt in tts.iter().rev() {
                match *tt {
                    TokenTree::Token(sp, ref tok) => {
                        first.replace_with((sp, tok.clone()));
                    }
                    TokenTree::Delimited(_, ref delimited) => {
                        build_recur(sets, &delimited.tts[..]);
                        first.replace_with((delimited.open_span,
                                            Token::OpenDelim(delimited.delim)));
                    }
                    TokenTree::Sequence(sp, ref seq_rep) => {
                        let subfirst = build_recur(sets, &seq_rep.tts[..]);

                        match sets.first.entry(sp) {
                            Entry::Vacant(vac) => {
                                vac.insert(Some(subfirst.clone()));
                            }
                            Entry::Occupied(mut occ) => {
                                // if there is already an entry, then a span must have collided.
                                // This should not happen with typical macro_rules macros,
                                // but syntax extensions need not maintain distinct spans,
                                // so distinct syntax trees can be assigned the same span.
                                // In such a case, the map cannot be trusted; so mark this
                                // entry as unusable.
                                occ.insert(None);
                            }
                        }

                        // If the sequence contents can be empty, then the first
                        // token could be the separator token itself.

                        if let (Some(ref sep), true) = (seq_rep.separator.clone(),
                                                        subfirst.maybe_empty) {
                            first.add_one_maybe((sp, sep.clone()));
                        }

                        // Reverse scan: Sequence comes before `first`.
                        if subfirst.maybe_empty || seq_rep.op == ast::KleeneOp::ZeroOrMore {
                            // If sequence is potentially empty, then
                            // union them (preserving first emptiness).
                            first.add_all(&TokenSet { maybe_empty: true, ..subfirst });
                        } else {
                            // Otherwise, sequence guaranteed
                            // non-empty; replace first.
                            first = subfirst;
                        }
                    }
                }
            }

            return first;
        }
    }

    // walks forward over `tts` until all potential FIRST tokens are
    // identified.
    fn first(&self, tts: &[TokenTree]) -> TokenSet {
        let mut first = TokenSet::empty();
        for tt in tts.iter() {
            assert!(first.maybe_empty);
            match *tt {
                TokenTree::Token(sp, ref tok) => {
                    first.add_one((sp, tok.clone()));
                    return first;
                }
                TokenTree::Delimited(_, ref delimited) => {
                    first.add_one((delimited.open_span,
                                   Token::OpenDelim(delimited.delim)));
                    return first;
                }
                TokenTree::Sequence(sp, ref seq_rep) => {
                    match self.first.get(&sp) {
                        Some(&Some(ref subfirst)) => {

                            // If the sequence contents can be empty, then the first
                            // token could be the separator token itself.

                            if let (Some(ref sep), true) = (seq_rep.separator.clone(),
                                                            subfirst.maybe_empty) {
                                first.add_one_maybe((sp, sep.clone()));
                            }

                            assert!(first.maybe_empty);
                            first.add_all(subfirst);
                            if subfirst.maybe_empty || seq_rep.op == ast::KleeneOp::ZeroOrMore {
                                // continue scanning for more first
                                // tokens, but also make sure we
                                // restore empty-tracking state
                                first.maybe_empty = true;
                                continue;
                            } else {
                                return first;
                            }
                        }

                        Some(&None) => {
                            panic!("assume all sequences have (unique) spans for now");
                        }

                        None => {
                            panic!("We missed a sequence during FirstSets construction");
                        }
                    }
                }
            }
        }

        // we only exit the loop if `tts` was empty or if every
        // element of `tts` matches the empty sequence.
        assert!(first.maybe_empty);
        return first;
    }
}

// A set of Tokens, which may include MatchNt tokens (for
// macro-by-example syntactic variables). It also carries the
// `maybe_empty` flag; that is true if and only if the matcher can
// match an empty token sequence.
//
// The First set is computed on submatchers like `$($a:expr b),* $(c)* d`,
// which has corresponding FIRST = {$a:expr, c, d}.
// Likewise, `$($a:expr b),* $(c)+ d` has FIRST = {$a:expr, c}.
//
// (Notably, we must allow for *-op to occur zero times.)
#[derive(Clone, Debug)]
struct TokenSet {
    tokens: Vec<(Span, Token)>,
    maybe_empty: bool,
}

impl TokenSet {
    // Returns a set for the empty sequence.
    fn empty() -> Self { TokenSet { tokens: Vec::new(), maybe_empty: true } }

    // Returns the set `{ tok }` for the single-token (and thus
    // non-empty) sequence [tok].
    fn singleton(tok: (Span, Token)) -> Self {
        TokenSet { tokens: vec![tok], maybe_empty: false }
    }

    // Changes self to be the set `{ tok }`.
    // Since `tok` is always present, marks self as non-empty.
    fn replace_with(&mut self, tok: (Span, Token)) {
        self.tokens.clear();
        self.tokens.push(tok);
        self.maybe_empty = false;
    }

    // Changes self to be the empty set `{}`; meant for use when
    // the particular token does not matter, but we want to
    // record that it occurs.
    fn replace_with_irrelevant(&mut self) {
        self.tokens.clear();
        self.maybe_empty = false;
    }

    // Adds `tok` to the set for `self`, marking sequence as non-empy.
    fn add_one(&mut self, tok: (Span, Token)) {
        if !self.tokens.contains(&tok) {
            self.tokens.push(tok);
        }
        self.maybe_empty = false;
    }

    // Adds `tok` to the set for `self`. (Leaves `maybe_empty` flag alone.)
    fn add_one_maybe(&mut self, tok: (Span, Token)) {
        if !self.tokens.contains(&tok) {
            self.tokens.push(tok);
        }
    }

    // Adds all elements of `other` to this.
    //
    // (Since this is a set, we filter out duplicates.)
    //
    // If `other` is potentially empty, then preserves the previous
    // setting of the empty flag of `self`. If `other` is guaranteed
    // non-empty, then `self` is marked non-empty.
    fn add_all(&mut self, other: &Self) {
        for tok in &other.tokens {
            if !self.tokens.contains(tok) {
                self.tokens.push(tok.clone());
            }
        }
        if !other.maybe_empty {
            self.maybe_empty = false;
        }
    }
}

// Checks that `matcher` is internally consistent and that it
// can legally by followed by a token N, for all N in `follow`.
// (If `follow` is empty, then it imposes no constraint on
// the `matcher`.)
//
// Returns the set of NT tokens that could possibly come last in
// `matcher`. (If `matcher` matches the empty sequence, then
// `maybe_empty` will be set to true.)
//
// Requires that `first_sets` is pre-computed for `matcher`;
// see `FirstSets::new`.
fn check_matcher_core(cx: &mut ExtCtxt,
                      first_sets: &FirstSets,
                      matcher: &[TokenTree],
                      follow: &TokenSet,
                      on_fail: &mut OnFail) -> TokenSet {
    use print::pprust::token_to_string;

    let mut last = TokenSet::empty();

    // 2. For each token and suffix  [T, SUFFIX] in M:
    // ensure that T can be followed by SUFFIX, and if SUFFIX may be empty,
    // then ensure T can also be followed by any element of FOLLOW.
    'each_token: for i in 0..matcher.len() {
        let token = &matcher[i];
        let suffix = &matcher[i+1..];

        let build_suffix_first = || {
            let mut s = first_sets.first(suffix);
            if s.maybe_empty { s.add_all(follow); }
            return s;
        };

        // (we build `suffix_first` on demand below; you can tell
        // which cases are supposed to fall through by looking for the
        // initialization of this variable.)
        let suffix_first;

        // First, update `last` so that it corresponds to the set
        // of NT tokens that might end the sequence `... token`.
        match *token {
            TokenTree::Token(sp, ref tok) => {
                let can_be_followed_by_any;
                if let Err(bad_frag) = has_legal_fragment_specifier(tok) {
                    on_fail.react(cx, sp, &format!("invalid fragment specifier `{}`", bad_frag));
                    // (This eliminates false positives and duplicates
                    // from error messages.)
                    can_be_followed_by_any = true;
                } else {
                    can_be_followed_by_any = token_can_be_followed_by_any(tok);
                }

                if can_be_followed_by_any {
                    // don't need to track tokens that work with any,
                    last.replace_with_irrelevant();
                    // ... and don't need to check tokens that can be
                    // followed by anything against SUFFIX.
                    continue 'each_token;
                } else {
                    last.replace_with((sp, tok.clone()));
                    suffix_first = build_suffix_first();
                }
            }
            TokenTree::Delimited(_, ref d) => {
                let my_suffix = TokenSet::singleton((d.close_span, Token::CloseDelim(d.delim)));
                check_matcher_core(cx, first_sets, &d.tts, &my_suffix, on_fail);
                // don't track non NT tokens
                last.replace_with_irrelevant();

                // also, we don't need to check delimited sequences
                // against SUFFIX
                continue 'each_token;
            }
            TokenTree::Sequence(sp, ref seq_rep) => {
                suffix_first = build_suffix_first();
                // The trick here: when we check the interior, we want
                // to include the separator (if any) as a potential
                // (but not guaranteed) element of FOLLOW. So in that
                // case, we make a temp copy of suffix and stuff
                // delimiter in there.
                //
                // FIXME: Should I first scan suffix_first to see if
                // delimiter is already in it before I go through the
                // work of cloning it? But then again, this way I may
                // get a "tighter" span?
                let mut new;
                let my_suffix = if let Some(ref u) = seq_rep.separator {
                    new = suffix_first.clone();
                    new.add_one_maybe((sp, u.clone()));
                    &new
                } else {
                    &suffix_first
                };

                // At this point, `suffix_first` is built, and
                // `my_suffix` is some TokenSet that we can use
                // for checking the interior of `seq_rep`.
                let next = check_matcher_core(cx, first_sets, &seq_rep.tts, my_suffix, on_fail);
                if next.maybe_empty {
                    last.add_all(&next);
                } else {
                    last = next;
                }

                // the recursive call to check_matcher_core already ran the 'each_last
                // check below, so we can just keep going forward here.
                continue 'each_token;
            }
        }

        // (`suffix_first` guaranteed initialized once reaching here.)

        // Now `last` holds the complete set of NT tokens that could
        // end the sequence before SUFFIX. Check that every one works with `suffix`.
        'each_last: for &(_sp, ref t) in &last.tokens {
            if let MatchNt(ref name, ref frag_spec, _, _) = *t {
                for &(sp, ref next_token) in &suffix_first.tokens {
                    match is_in_follow(cx, next_token, &frag_spec.name.as_str()) {
                        Err(msg) => {
                            on_fail.react(cx, sp, &msg);
                            // don't bother reporting every source of
                            // conflict for a particular element of `last`.
                            continue 'each_last;
                        }
                        Ok(true) => {}
                        Ok(false) => {
                            let may_be = if last.tokens.len() == 1 &&
                                suffix_first.tokens.len() == 1
                            {
                                "is"
                            } else {
                                "may be"
                            };

                            on_fail.react(
                                cx, sp,
                                &format!("`${name}:{frag}` {may_be} followed by `{next}`, which \
                                          is not allowed for `{frag}` fragments",
                                         name=name,
                                         frag=frag_spec,
                                         next=token_to_string(next_token),
                                         may_be=may_be));
                        }
                    }
                }
            }
        }
    }
    last
}


fn token_can_be_followed_by_any(tok: &Token) -> bool {
    if let &MatchNt(_, ref frag_spec, _, _) = tok {
        frag_can_be_followed_by_any(&frag_spec.name.as_str())
    } else {
        // (Non NT's can always be followed by anthing in matchers.)
        true
    }
}

/// True if a fragment of type `frag` can be followed by any sort of
/// token.  We use this (among other things) as a useful approximation
/// for when `frag` can be followed by a repetition like `$(...)*` or
/// `$(...)+`. In general, these can be a bit tricky to reason about,
/// so we adopt a conservative position that says that any fragment
/// specifier which consumes at most one token tree can be followed by
/// a fragment specifier (indeed, these fragments can be followed by
/// ANYTHING without fear of future compatibility hazards).
fn frag_can_be_followed_by_any(frag: &str) -> bool {
    match frag {
        "item" |  // always terminated by `}` or `;`
        "block" | // exactly one token tree
        "ident" | // exactly one token tree
        "meta" |  // exactly one token tree
        "tt" =>    // exactly one token tree
            true,

        _ =>
            false,
    }
}

/// True if a fragment of type `frag` can be followed by any sort of
/// token.  We use this (among other things) as a useful approximation
/// for when `frag` can be followed by a repetition like `$(...)*` or
/// `$(...)+`. In general, these can be a bit tricky to reason about,
/// so we adopt a conservative position that says that any fragment
/// specifier which consumes at most one token tree can be followed by
/// a fragment specifier (indeed, these fragments can be followed by
/// ANYTHING without fear of future compatibility hazards).
fn can_be_followed_by_any(frag: &str) -> bool {
    match frag {
        "item" |  // always terminated by `}` or `;`
        "block" | // exactly one token tree
        "ident" | // exactly one token tree
        "meta" |  // exactly one token tree
        "tt" =>    // exactly one token tree
            true,

        _ =>
            false,
    }
}

/// True if `frag` can legally be followed by the token `tok`. For
/// fragments that can consume an unbounded number of tokens, `tok`
/// must be within a well-defined follow set. This is intended to
/// guarantee future compatibility: for example, without this rule, if
/// we expanded `expr` to include a new binary operator, we might
/// break macros that were relying on that binary operator as a
/// separator.
fn is_in_follow(_: &ExtCtxt, tok: &Token, frag: &str) -> Result<bool, String> {
    if let &CloseDelim(_) = tok {
        // closing a token tree can never be matched by any fragment;
        // iow, we always require that `(` and `)` match, etc.
        Ok(true)
    } else {
        match frag {
            "item" => {
                // since items *must* be followed by either a `;` or a `}`, we can
                // accept anything after them
                Ok(true)
            },
            "block" => {
                // anything can follow block, the braces provide an easy boundary to
                // maintain
                Ok(true)
            },
            "stmt" | "expr"  => {
                match *tok {
                    FatArrow | Comma | Semi => Ok(true),
                    _ => Ok(false)
                }
            },
            "pat" => {
                match *tok {
                    FatArrow | Comma | Eq | BinOp(token::Or) => Ok(true),
                    Ident(i, _) if (i.name.as_str() == "if" ||
                                    i.name.as_str() == "in") => Ok(true),
                    _ => Ok(false)
                }
            },
            "path" | "ty" => {
                match *tok {
                    OpenDelim(token::DelimToken::Brace) |
                    Comma | FatArrow | Colon | Eq | Gt | Semi | BinOp(token::Or) => Ok(true),
                    Ident(i, _) if (i.name.as_str() == "as" ||
                                    i.name.as_str() == "where") => Ok(true),
                    _ => Ok(false)
                }
            },
            "ident" => {
                // being a single token, idents are harmless
                Ok(true)
            },
            "meta" | "tt" => {
                // being either a single token or a delimited sequence, tt is
                // harmless
                Ok(true)
            },
            _ => Err(format!("invalid fragment specifier `{}`", frag))
        }
    }
}

fn has_legal_fragment_specifier(tok: &Token) -> Result<(), String> {
    debug!("has_legal_fragment_specifier({:?})", tok);
    if let &MatchNt(_, ref frag_spec, _, _) = tok {
        let s = &frag_spec.name.as_str();
        if !is_legal_fragment_specifier(s) {
            return Err(s.to_string());
        }
    }
    Ok(())
}

fn is_legal_fragment_specifier(frag: &str) -> bool {
    match frag {
        "item" | "block" | "stmt" | "expr" | "pat" |
        "path" | "ty" | "ident" | "meta" | "tt" => true,
        _ => false,
    }
}
