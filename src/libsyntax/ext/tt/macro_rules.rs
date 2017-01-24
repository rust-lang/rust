// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use {ast, attr};
use syntax_pos::{Span, DUMMY_SP};
use ext::base::{DummyResult, ExtCtxt, MacResult, SyntaxExtension};
use ext::base::{NormalTT, TTMacroExpander};
use ext::expand::{Expansion, ExpansionKind};
use ext::tt::macro_parser::{Success, Error, Failure};
use ext::tt::macro_parser::{MatchedSeq, MatchedNonterminal};
use ext::tt::macro_parser::{parse, parse_failure_msg};
use ext::tt::transcribe::transcribe;
use parse::{Directory, ParseSess};
use parse::parser::Parser;
use parse::token::{self, NtTT, Token};
use parse::token::Token::*;
use print;
use symbol::Symbol;
use tokenstream::{self, TokenTree};

use std::collections::{HashMap};
use std::collections::hash_map::{Entry};
use std::rc::Rc;

pub struct ParserAnyMacro<'a> {
    parser: Parser<'a>,

    /// Span of the expansion site of the macro this parser is for
    site_span: Span,
    /// The ident of the macro we're parsing
    macro_ident: ast::Ident
}

impl<'a> ParserAnyMacro<'a> {
    pub fn make(mut self: Box<ParserAnyMacro<'a>>, kind: ExpansionKind) -> Expansion {
        let ParserAnyMacro { site_span, macro_ident, ref mut parser } = *self;
        let expansion = panictry!(parser.parse_expansion(kind, true));

        // We allow semicolons at the end of expressions -- e.g. the semicolon in
        // `macro_rules! m { () => { panic!(); } }` isn't parsed by `.parse_expr()`,
        // but `m!()` is allowed in expression positions (c.f. issue #34706).
        if kind == ExpansionKind::Expr && parser.token == token::Semi {
            parser.bump();
        }

        // Make sure we don't have any tokens left to parse so we don't silently drop anything.
        parser.ensure_complete_parse(macro_ident.name, kind.name(), site_span);
        expansion
    }
}

struct MacroRulesMacroExpander {
    name: ast::Ident,
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
                          arg,
                          &self.lhses,
                          &self.rhses)
    }
}

/// Given `lhses` and `rhses`, this is the new macro we create
fn generic_extension<'cx>(cx: &'cx ExtCtxt,
                          sp: Span,
                          name: ast::Ident,
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
    let mut best_fail_tok = None;

    for (i, lhs) in lhses.iter().enumerate() { // try each arm's matchers
        let lhs_tt = match *lhs {
            TokenTree::Delimited(_, ref delim) => &delim.tts[..],
            _ => cx.span_bug(sp, "malformed macro lhs")
        };

        match TokenTree::parse(cx, lhs_tt, arg) {
            Success(named_matches) => {
                let rhs = match rhses[i] {
                    // ignore delimiters
                    TokenTree::Delimited(_, ref delimed) => delimed.tts.clone(),
                    _ => cx.span_bug(sp, "malformed macro rhs"),
                };
                // rhs has holes ( `$id` and `$(...)` that need filled)
                let tts = transcribe(&cx.parse_sess.span_diagnostic, Some(named_matches), rhs);
                let directory = Directory {
                    path: cx.current_expansion.module.directory.clone(),
                    ownership: cx.current_expansion.directory_ownership,
                };
                let mut p = Parser::new(cx.parse_sess(), tts, Some(directory), false);
                p.root_module_name = cx.current_expansion.module.mod_path.last()
                    .map(|id| (*id.name.as_str()).to_owned());

                p.check_unknown_macro_variable();
                // Let the context choose how to interpret the result.
                // Weird, but useful for X-macros.
                return Box::new(ParserAnyMacro {
                    parser: p,

                    // Pass along the original expansion site and the name of the macro
                    // so we can print a useful error message if the parse of the expanded
                    // macro leaves unparsed tokens.
                    site_span: sp,
                    macro_ident: name
                })
            }
            Failure(sp, tok) => if sp.lo >= best_fail_spot.lo {
                best_fail_spot = sp;
                best_fail_tok = Some(tok);
            },
            Error(err_sp, ref msg) => {
                cx.span_fatal(err_sp.substitute_dummy(sp), &msg[..])
            }
        }
    }

    let best_fail_msg = parse_failure_msg(best_fail_tok.expect("ran no matchers"));
    cx.span_fatal(best_fail_spot.substitute_dummy(sp), &best_fail_msg);
}

// Note that macro-by-example's input is also matched against a token tree:
//                   $( $lhs:tt => $rhs:tt );+
//
// Holy self-referential!

/// Converts a `macro_rules!` invocation into a syntax extension.
pub fn compile(sess: &ParseSess, def: &ast::MacroDef) -> SyntaxExtension {
    let lhs_nm = ast::Ident::with_empty_ctxt(Symbol::gensym("lhs"));
    let rhs_nm = ast::Ident::with_empty_ctxt(Symbol::gensym("rhs"));

    // The pattern that macro_rules matches.
    // The grammar for macro_rules! is:
    // $( $lhs:tt => $rhs:tt );+
    // ...quasiquoting this would be nice.
    // These spans won't matter, anyways
    let match_lhs_tok = MatchNt(lhs_nm, ast::Ident::from_str("tt"));
    let match_rhs_tok = MatchNt(rhs_nm, ast::Ident::from_str("tt"));
    let argument_gram = vec![
        TokenTree::Sequence(DUMMY_SP, Rc::new(tokenstream::SequenceRepetition {
            tts: vec![
                TokenTree::Token(DUMMY_SP, match_lhs_tok),
                TokenTree::Token(DUMMY_SP, token::FatArrow),
                TokenTree::Token(DUMMY_SP, match_rhs_tok),
            ],
            separator: Some(token::Semi),
            op: tokenstream::KleeneOp::OneOrMore,
            num_captures: 2,
        })),
        // to phase into semicolon-termination instead of semicolon-separation
        TokenTree::Sequence(DUMMY_SP, Rc::new(tokenstream::SequenceRepetition {
            tts: vec![TokenTree::Token(DUMMY_SP, token::Semi)],
            separator: None,
            op: tokenstream::KleeneOp::ZeroOrMore,
            num_captures: 0
        })),
    ];

    // Parse the macro_rules! invocation
    let argument_map = match parse(sess, def.body.clone(), &argument_gram, None) {
        Success(m) => m,
        Failure(sp, tok) => {
            let s = parse_failure_msg(tok);
            panic!(sess.span_diagnostic.span_fatal(sp.substitute_dummy(def.span), &s));
        }
        Error(sp, s) => {
            panic!(sess.span_diagnostic.span_fatal(sp.substitute_dummy(def.span), &s));
        }
    };

    let mut valid = true;

    // Extract the arguments:
    let lhses = match **argument_map.get(&lhs_nm).unwrap() {
        MatchedSeq(ref s, _) => {
            s.iter().map(|m| {
                if let MatchedNonterminal(ref nt) = **m {
                    if let NtTT(ref tt) = **nt {
                        valid &= check_lhs_nt_follows(sess, tt);
                        return (*tt).clone();
                    }
                }
                sess.span_diagnostic.span_bug(def.span, "wrong-structured lhs")
            }).collect::<Vec<TokenTree>>()
        }
        _ => sess.span_diagnostic.span_bug(def.span, "wrong-structured lhs")
    };

    let rhses = match **argument_map.get(&rhs_nm).unwrap() {
        MatchedSeq(ref s, _) => {
            s.iter().map(|m| {
                if let MatchedNonterminal(ref nt) = **m {
                    if let NtTT(ref tt) = **nt {
                        return (*tt).clone();
                    }
                }
                sess.span_diagnostic.span_bug(def.span, "wrong-structured lhs")
            }).collect()
        }
        _ => sess.span_diagnostic.span_bug(def.span, "wrong-structured rhs")
    };

    for rhs in &rhses {
        valid &= check_rhs(sess, rhs);
    }

    // don't abort iteration early, so that errors for multiple lhses can be reported
    for lhs in &lhses {
        valid &= check_lhs_no_empty_seq(sess, &[lhs.clone()])
    }

    let exp: Box<_> = Box::new(MacroRulesMacroExpander {
        name: def.ident,
        lhses: lhses,
        rhses: rhses,
        valid: valid,
    });

    NormalTT(exp, Some(def.span), attr::contains_name(&def.attrs, "allow_internal_unstable"))
}

fn check_lhs_nt_follows(sess: &ParseSess, lhs: &TokenTree) -> bool {
    // lhs is going to be like TokenTree::Delimited(...), where the
    // entire lhs is those tts. Or, it can be a "bare sequence", not wrapped in parens.
    match lhs {
        &TokenTree::Delimited(_, ref tts) => check_matcher(sess, &tts.tts),
        _ => {
            let msg = "invalid macro matcher; matchers must be contained in balanced delimiters";
            sess.span_diagnostic.span_err(lhs.get_span(), msg);
            false
        }
    }
    // we don't abort on errors on rejection, the driver will do that for us
    // after parsing/expansion. we can report every error in every macro this way.
}

/// Check that the lhs contains no repetition which could match an empty token
/// tree, because then the matcher would hang indefinitely.
fn check_lhs_no_empty_seq(sess: &ParseSess, tts: &[TokenTree]) -> bool {
    for tt in tts {
        match *tt {
            TokenTree::Token(_, _) => (),
            TokenTree::Delimited(_, ref del) => if !check_lhs_no_empty_seq(sess, &del.tts) {
                return false;
            },
            TokenTree::Sequence(span, ref seq) => {
                if seq.separator.is_none() {
                    if seq.tts.iter().all(|seq_tt| {
                        match *seq_tt {
                            TokenTree::Sequence(_, ref sub_seq) =>
                                sub_seq.op == tokenstream::KleeneOp::ZeroOrMore,
                            _ => false,
                        }
                    }) {
                        sess.span_diagnostic.span_err(span, "repetition matches empty token tree");
                        return false;
                    }
                }
                if !check_lhs_no_empty_seq(sess, &seq.tts) {
                    return false;
                }
            }
        }
    }

    true
}

fn check_rhs(sess: &ParseSess, rhs: &TokenTree) -> bool {
    match *rhs {
        TokenTree::Delimited(..) => return true,
        _ => sess.span_diagnostic.span_err(rhs.get_span(), "macro rhs must be delimited")
    }
    false
}

fn check_matcher(sess: &ParseSess, matcher: &[TokenTree]) -> bool {
    let first_sets = FirstSets::new(matcher);
    let empty_suffix = TokenSet::empty();
    let err = sess.span_diagnostic.err_count();
    check_matcher_core(sess, &first_sets, matcher, &empty_suffix);
    err == sess.span_diagnostic.err_count()
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
                    TokenTree::Delimited(span, ref delimited) => {
                        build_recur(sets, &delimited.tts[..]);
                        first.replace_with((delimited.open_tt(span).span(),
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
                        if subfirst.maybe_empty || seq_rep.op == tokenstream::KleeneOp::ZeroOrMore {
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
                TokenTree::Delimited(span, ref delimited) => {
                    first.add_one((delimited.open_tt(span).span(),
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
                            if subfirst.maybe_empty ||
                               seq_rep.op == tokenstream::KleeneOp::ZeroOrMore {
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
fn check_matcher_core(sess: &ParseSess,
                      first_sets: &FirstSets,
                      matcher: &[TokenTree],
                      follow: &TokenSet) -> TokenSet {
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
                    let msg = format!("invalid fragment specifier `{}`", bad_frag);
                    sess.span_diagnostic.struct_span_err(sp, &msg)
                        .help("valid fragment specifiers are `ident`, `block`, \
                               `stmt`, `expr`, `pat`, `ty`, `path`, `meta`, `tt` \
                               and `item`")
                        .emit();
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
            TokenTree::Delimited(span, ref d) => {
                let my_suffix = TokenSet::singleton((d.close_tt(span).span(),
                                                     Token::CloseDelim(d.delim)));
                check_matcher_core(sess, first_sets, &d.tts, &my_suffix);
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
                let next = check_matcher_core(sess, first_sets, &seq_rep.tts, my_suffix);
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
            if let MatchNt(ref name, ref frag_spec) = *t {
                for &(sp, ref next_token) in &suffix_first.tokens {
                    match is_in_follow(next_token, &frag_spec.name.as_str()) {
                        Err((msg, help)) => {
                            sess.span_diagnostic.struct_span_err(sp, &msg).help(help).emit();
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

                            sess.span_diagnostic.span_err(
                                sp,
                                &format!("`${name}:{frag}` {may_be} followed by `{next}`, which \
                                          is not allowed for `{frag}` fragments",
                                         name=name,
                                         frag=frag_spec,
                                         next=token_to_string(next_token),
                                         may_be=may_be)
                            );
                        }
                    }
                }
            }
        }
    }
    last
}

fn token_can_be_followed_by_any(tok: &Token) -> bool {
    if let &MatchNt(_, ref frag_spec) = tok {
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
        "item"  | // always terminated by `}` or `;`
        "block" | // exactly one token tree
        "ident" | // exactly one token tree
        "meta"  | // exactly one token tree
        "tt" =>   // exactly one token tree
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
// when changing this do not forget to update doc/book/macros.md!
fn is_in_follow(tok: &Token, frag: &str) -> Result<bool, (String, &'static str)> {
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
                    Ident(i) if i.name == "if" || i.name == "in" => Ok(true),
                    _ => Ok(false)
                }
            },
            "path" | "ty" => {
                match *tok {
                    OpenDelim(token::DelimToken::Brace) | OpenDelim(token::DelimToken::Bracket) |
                    Comma | FatArrow | Colon | Eq | Gt | Semi | BinOp(token::Or) => Ok(true),
                    MatchNt(_, ref frag) if frag.name == "block" => Ok(true),
                    Ident(i) if i.name == "as" || i.name == "where" => Ok(true),
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
            _ => Err((format!("invalid fragment specifier `{}`", frag),
                     "valid fragment specifiers are `ident`, `block`, \
                      `stmt`, `expr`, `pat`, `ty`, `path`, `meta`, `tt` \
                      and `item`"))
        }
    }
}

fn has_legal_fragment_specifier(tok: &Token) -> Result<(), String> {
    debug!("has_legal_fragment_specifier({:?})", tok);
    if let &MatchNt(_, ref frag_spec) = tok {
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
