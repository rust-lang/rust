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
use edition::Edition;
use ext::base::{DummyResult, ExtCtxt, MacResult, SyntaxExtension};
use ext::base::{NormalTT, TTMacroExpander};
use ext::expand::{AstFragment, AstFragmentKind};
use ext::tt::macro_parser::{Success, Error, Failure};
use ext::tt::macro_parser::{MatchedSeq, MatchedNonterminal};
use ext::tt::macro_parser::{parse, parse_failure_msg};
use ext::tt::quoted;
use ext::tt::transcribe::transcribe;
use feature_gate::{self, emit_feature_err, Features, GateIssue};
use parse::{Directory, ParseSess};
use parse::parser::Parser;
use parse::token::{self, NtTT};
use parse::token::Token::*;
use symbol::Symbol;
use tokenstream::{TokenStream, TokenTree};

use std::borrow::Cow;
use std::collections::HashMap;
use std::collections::hash_map::Entry;

use rustc_data_structures::sync::Lrc;

pub struct ParserAnyMacro<'a> {
    parser: Parser<'a>,

    /// Span of the expansion site of the macro this parser is for
    site_span: Span,
    /// The ident of the macro we're parsing
    macro_ident: ast::Ident
}

impl<'a> ParserAnyMacro<'a> {
    pub fn make(mut self: Box<ParserAnyMacro<'a>>, kind: AstFragmentKind) -> AstFragment {
        let ParserAnyMacro { site_span, macro_ident, ref mut parser } = *self;
        let fragment = panictry!(parser.parse_ast_fragment(kind, true));

        // We allow semicolons at the end of expressions -- e.g. the semicolon in
        // `macro_rules! m { () => { panic!(); } }` isn't parsed by `.parse_expr()`,
        // but `m!()` is allowed in expression positions (c.f. issue #34706).
        if kind == AstFragmentKind::Expr && parser.token == token::Semi {
            parser.bump();
        }

        // Make sure we don't have any tokens left to parse so we don't silently drop anything.
        let path = ast::Path::from_ident(macro_ident.with_span_pos(site_span));
        parser.ensure_complete_parse(&path, kind.name(), site_span);
        fragment
    }
}

struct MacroRulesMacroExpander {
    name: ast::Ident,
    lhses: Vec<quoted::TokenTree>,
    rhses: Vec<quoted::TokenTree>,
    valid: bool,
}

impl TTMacroExpander for MacroRulesMacroExpander {
    fn expand<'cx>(&self,
                   cx: &'cx mut ExtCtxt,
                   sp: Span,
                   input: TokenStream)
                   -> Box<dyn MacResult+'cx> {
        if !self.valid {
            return DummyResult::any(sp);
        }
        generic_extension(cx,
                          sp,
                          self.name,
                          input,
                          &self.lhses,
                          &self.rhses)
    }
}

fn trace_macros_note(cx: &mut ExtCtxt, sp: Span, message: String) {
    let sp = sp.macro_backtrace().last().map(|trace| trace.call_site).unwrap_or(sp);
    cx.expansions.entry(sp).or_default().push(message);
}

/// Given `lhses` and `rhses`, this is the new macro we create
fn generic_extension<'cx>(cx: &'cx mut ExtCtxt,
                          sp: Span,
                          name: ast::Ident,
                          arg: TokenStream,
                          lhses: &[quoted::TokenTree],
                          rhses: &[quoted::TokenTree])
                          -> Box<dyn MacResult+'cx> {
    if cx.trace_macros() {
        trace_macros_note(cx, sp, format!("expanding `{}! {{ {} }}`", name, arg));
    }

    // Which arm's failure should we report? (the one furthest along)
    let mut best_fail_spot = DUMMY_SP;
    let mut best_fail_tok = None;

    for (i, lhs) in lhses.iter().enumerate() { // try each arm's matchers
        let lhs_tt = match *lhs {
            quoted::TokenTree::Delimited(_, ref delim) => &delim.tts[..],
            _ => cx.span_bug(sp, "malformed macro lhs")
        };

        match TokenTree::parse(cx, lhs_tt, arg.clone()) {
            Success(named_matches) => {
                let rhs = match rhses[i] {
                    // ignore delimiters
                    quoted::TokenTree::Delimited(_, ref delimed) => delimed.tts.clone(),
                    _ => cx.span_bug(sp, "malformed macro rhs"),
                };

                let rhs_spans = rhs.iter().map(|t| t.span()).collect::<Vec<_>>();
                // rhs has holes ( `$id` and `$(...)` that need filled)
                let mut tts = transcribe(cx, Some(named_matches), rhs);

                // Replace all the tokens for the corresponding positions in the macro, to maintain
                // proper positions in error reporting, while maintaining the macro_backtrace.
                if rhs_spans.len() == tts.len() {
                    tts = tts.map_enumerated(|i, tt| {
                        let mut tt = tt.clone();
                        let mut sp = rhs_spans[i];
                        sp = sp.with_ctxt(tt.span().ctxt());
                        tt.set_span(sp);
                        tt
                    });
                }

                if cx.trace_macros() {
                    trace_macros_note(cx, sp, format!("to `{}`", tts));
                }

                let directory = Directory {
                    path: Cow::from(cx.current_expansion.module.directory.as_path()),
                    ownership: cx.current_expansion.directory_ownership,
                };
                let mut p = Parser::new(cx.parse_sess(), tts, Some(directory), true, false);
                p.root_module_name = cx.current_expansion.module.mod_path.last()
                    .map(|id| id.as_str().to_string());

                p.process_potential_macro_variable();
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
            Failure(sp, tok) => if sp.lo() >= best_fail_spot.lo() {
                best_fail_spot = sp;
                best_fail_tok = Some(tok);
            },
            Error(err_sp, ref msg) => {
                cx.span_fatal(err_sp.substitute_dummy(sp), &msg[..])
            }
        }
    }

    let best_fail_msg = parse_failure_msg(best_fail_tok.expect("ran no matchers"));
    let mut err = cx.struct_span_err(best_fail_spot.substitute_dummy(sp), &best_fail_msg);

    // Check whether there's a missing comma in this macro call, like `println!("{}" a);`
    if let Some((arg, comma_span)) = arg.add_comma() {
        for lhs in lhses { // try each arm's matchers
            let lhs_tt = match *lhs {
                quoted::TokenTree::Delimited(_, ref delim) => &delim.tts[..],
                _ => continue,
            };
            match TokenTree::parse(cx, lhs_tt, arg.clone()) {
                Success(_) => {
                    if comma_span == DUMMY_SP {
                        err.note("you might be missing a comma");
                    } else {
                        err.span_suggestion_short(
                            comma_span,
                            "missing comma here",
                            ", ".to_string(),
                        );
                    }
                }
                _ => {}
            }
        }
    }
    err.emit();
    cx.trace_macros_diag();
    DummyResult::any(sp)
}

// Note that macro-by-example's input is also matched against a token tree:
//                   $( $lhs:tt => $rhs:tt );+
//
// Holy self-referential!

/// Converts a `macro_rules!` invocation into a syntax extension.
pub fn compile(sess: &ParseSess, features: &Features, def: &ast::Item, edition: Edition)
               -> SyntaxExtension {
    let lhs_nm = ast::Ident::with_empty_ctxt(Symbol::gensym("lhs"));
    let rhs_nm = ast::Ident::with_empty_ctxt(Symbol::gensym("rhs"));

    // Parse the macro_rules! invocation
    let body = match def.node {
        ast::ItemKind::MacroDef(ref body) => body,
        _ => unreachable!(),
    };

    // The pattern that macro_rules matches.
    // The grammar for macro_rules! is:
    // $( $lhs:tt => $rhs:tt );+
    // ...quasiquoting this would be nice.
    // These spans won't matter, anyways
    let argument_gram = vec![
        quoted::TokenTree::Sequence(DUMMY_SP, Lrc::new(quoted::SequenceRepetition {
            tts: vec![
                quoted::TokenTree::MetaVarDecl(DUMMY_SP, lhs_nm, ast::Ident::from_str("tt")),
                quoted::TokenTree::Token(DUMMY_SP, token::FatArrow),
                quoted::TokenTree::MetaVarDecl(DUMMY_SP, rhs_nm, ast::Ident::from_str("tt")),
            ],
            separator: Some(if body.legacy { token::Semi } else { token::Comma }),
            op: quoted::KleeneOp::OneOrMore,
            num_captures: 2,
        })),
        // to phase into semicolon-termination instead of semicolon-separation
        quoted::TokenTree::Sequence(DUMMY_SP, Lrc::new(quoted::SequenceRepetition {
            tts: vec![quoted::TokenTree::Token(DUMMY_SP, token::Semi)],
            separator: None,
            op: quoted::KleeneOp::ZeroOrMore,
            num_captures: 0
        })),
    ];

    let argument_map = match parse(sess, body.stream(), &argument_gram, None, true) {
        Success(m) => m,
        Failure(sp, tok) => {
            let s = parse_failure_msg(tok);
            sess.span_diagnostic.span_fatal(sp.substitute_dummy(def.span), &s).raise();
        }
        Error(sp, s) => {
            sess.span_diagnostic.span_fatal(sp.substitute_dummy(def.span), &s).raise();
        }
    };

    let mut valid = true;

    // Extract the arguments:
    let lhses = match *argument_map[&lhs_nm] {
        MatchedSeq(ref s, _) => {
            s.iter().map(|m| {
                if let MatchedNonterminal(ref nt) = *m {
                    if let NtTT(ref tt) = **nt {
                        let tt = quoted::parse(
                            tt.clone().into(),
                            true,
                            sess,
                            features,
                            &def.attrs,
                            edition,
                            def.id,
                        )
                        .pop()
                        .unwrap();
                        valid &= check_lhs_nt_follows(sess, features, &def.attrs, &tt);
                        return tt;
                    }
                }
                sess.span_diagnostic.span_bug(def.span, "wrong-structured lhs")
            }).collect::<Vec<quoted::TokenTree>>()
        }
        _ => sess.span_diagnostic.span_bug(def.span, "wrong-structured lhs")
    };

    let rhses = match *argument_map[&rhs_nm] {
        MatchedSeq(ref s, _) => {
            s.iter().map(|m| {
                if let MatchedNonterminal(ref nt) = *m {
                    if let NtTT(ref tt) = **nt {
                        return quoted::parse(
                            tt.clone().into(),
                            false,
                            sess,
                            features,
                            &def.attrs,
                            edition,
                            def.id,
                        ).pop()
                         .unwrap();
                    }
                }
                sess.span_diagnostic.span_bug(def.span, "wrong-structured lhs")
            }).collect::<Vec<quoted::TokenTree>>()
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

    let expander: Box<_> = Box::new(MacroRulesMacroExpander {
        name: def.ident,
        lhses,
        rhses,
        valid,
    });

    if body.legacy {
        let allow_internal_unstable = attr::contains_name(&def.attrs, "allow_internal_unstable");
        let allow_internal_unsafe = attr::contains_name(&def.attrs, "allow_internal_unsafe");
        let mut local_inner_macros = false;
        if let Some(macro_export) = attr::find_by_name(&def.attrs, "macro_export") {
            if let Some(l) = macro_export.meta_item_list() {
                local_inner_macros = attr::list_contains_name(&l, "local_inner_macros");
            }
        }

        let unstable_feature = attr::find_stability(&sess.span_diagnostic,
                                                    &def.attrs, def.span).and_then(|stability| {
            if let attr::StabilityLevel::Unstable { issue, .. } = stability.level {
                Some((stability.feature, issue))
            } else {
                None
            }
        });

        NormalTT {
            expander,
            def_info: Some((def.id, def.span)),
            allow_internal_unstable,
            allow_internal_unsafe,
            local_inner_macros,
            unstable_feature,
            edition,
        }
    } else {
        let is_transparent = attr::contains_name(&def.attrs, "rustc_transparent_macro");

        SyntaxExtension::DeclMacro {
            expander,
            def_info: Some((def.id, def.span)),
            is_transparent,
            edition,
        }
    }
}

fn check_lhs_nt_follows(sess: &ParseSess,
                        features: &Features,
                        attrs: &[ast::Attribute],
                        lhs: &quoted::TokenTree) -> bool {
    // lhs is going to be like TokenTree::Delimited(...), where the
    // entire lhs is those tts. Or, it can be a "bare sequence", not wrapped in parens.
    if let quoted::TokenTree::Delimited(_, ref tts) = *lhs {
        check_matcher(sess, features, attrs, &tts.tts)
    } else {
        let msg = "invalid macro matcher; matchers must be contained in balanced delimiters";
        sess.span_diagnostic.span_err(lhs.span(), msg);
        false
    }
    // we don't abort on errors on rejection, the driver will do that for us
    // after parsing/expansion. we can report every error in every macro this way.
}

/// Check that the lhs contains no repetition which could match an empty token
/// tree, because then the matcher would hang indefinitely.
fn check_lhs_no_empty_seq(sess: &ParseSess, tts: &[quoted::TokenTree]) -> bool {
    use self::quoted::TokenTree;
    for tt in tts {
        match *tt {
            TokenTree::Token(..) | TokenTree::MetaVar(..) | TokenTree::MetaVarDecl(..) => (),
            TokenTree::Delimited(_, ref del) => if !check_lhs_no_empty_seq(sess, &del.tts) {
                return false;
            },
            TokenTree::Sequence(span, ref seq) => {
                if seq.separator.is_none() && seq.tts.iter().all(|seq_tt| {
                    match *seq_tt {
                        TokenTree::MetaVarDecl(_, _, id) => id.name == "vis",
                        TokenTree::Sequence(_, ref sub_seq) =>
                            sub_seq.op == quoted::KleeneOp::ZeroOrMore,
                        _ => false,
                    }
                }) {
                    sess.span_diagnostic.span_err(span, "repetition matches empty token tree");
                    return false;
                }
                if !check_lhs_no_empty_seq(sess, &seq.tts) {
                    return false;
                }
            }
        }
    }

    true
}

fn check_rhs(sess: &ParseSess, rhs: &quoted::TokenTree) -> bool {
    match *rhs {
        quoted::TokenTree::Delimited(..) => return true,
        _ => sess.span_diagnostic.span_err(rhs.span(), "macro rhs must be delimited")
    }
    false
}

fn check_matcher(sess: &ParseSess,
                 features: &Features,
                 attrs: &[ast::Attribute],
                 matcher: &[quoted::TokenTree]) -> bool {
    let first_sets = FirstSets::new(matcher);
    let empty_suffix = TokenSet::empty();
    let err = sess.span_diagnostic.err_count();
    check_matcher_core(sess, features, attrs, &first_sets, matcher, &empty_suffix);
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
    fn new(tts: &[quoted::TokenTree]) -> FirstSets {
        use self::quoted::TokenTree;

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
                    TokenTree::Token(..) | TokenTree::MetaVar(..) | TokenTree::MetaVarDecl(..) => {
                        first.replace_with(tt.clone());
                    }
                    TokenTree::Delimited(span, ref delimited) => {
                        build_recur(sets, &delimited.tts[..]);
                        first.replace_with(delimited.open_tt(span));
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
                            first.add_one_maybe(TokenTree::Token(sp, sep.clone()));
                        }

                        // Reverse scan: Sequence comes before `first`.
                        if subfirst.maybe_empty || seq_rep.op == quoted::KleeneOp::ZeroOrMore {
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

            first
        }
    }

    // walks forward over `tts` until all potential FIRST tokens are
    // identified.
    fn first(&self, tts: &[quoted::TokenTree]) -> TokenSet {
        use self::quoted::TokenTree;

        let mut first = TokenSet::empty();
        for tt in tts.iter() {
            assert!(first.maybe_empty);
            match *tt {
                TokenTree::Token(..) | TokenTree::MetaVar(..) | TokenTree::MetaVarDecl(..) => {
                    first.add_one(tt.clone());
                    return first;
                }
                TokenTree::Delimited(span, ref delimited) => {
                    first.add_one(delimited.open_tt(span));
                    return first;
                }
                TokenTree::Sequence(sp, ref seq_rep) => {
                    match self.first.get(&sp) {
                        Some(&Some(ref subfirst)) => {

                            // If the sequence contents can be empty, then the first
                            // token could be the separator token itself.

                            if let (Some(ref sep), true) = (seq_rep.separator.clone(),
                                                            subfirst.maybe_empty) {
                                first.add_one_maybe(TokenTree::Token(sp, sep.clone()));
                            }

                            assert!(first.maybe_empty);
                            first.add_all(subfirst);
                            if subfirst.maybe_empty ||
                               seq_rep.op == quoted::KleeneOp::ZeroOrMore {
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
        first
    }
}

// A set of `quoted::TokenTree`s, which may include `TokenTree::Match`s
// (for macro-by-example syntactic variables). It also carries the
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
    tokens: Vec<quoted::TokenTree>,
    maybe_empty: bool,
}

impl TokenSet {
    // Returns a set for the empty sequence.
    fn empty() -> Self { TokenSet { tokens: Vec::new(), maybe_empty: true } }

    // Returns the set `{ tok }` for the single-token (and thus
    // non-empty) sequence [tok].
    fn singleton(tok: quoted::TokenTree) -> Self {
        TokenSet { tokens: vec![tok], maybe_empty: false }
    }

    // Changes self to be the set `{ tok }`.
    // Since `tok` is always present, marks self as non-empty.
    fn replace_with(&mut self, tok: quoted::TokenTree) {
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
    fn add_one(&mut self, tok: quoted::TokenTree) {
        if !self.tokens.contains(&tok) {
            self.tokens.push(tok);
        }
        self.maybe_empty = false;
    }

    // Adds `tok` to the set for `self`. (Leaves `maybe_empty` flag alone.)
    fn add_one_maybe(&mut self, tok: quoted::TokenTree) {
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
                      features: &Features,
                      attrs: &[ast::Attribute],
                      first_sets: &FirstSets,
                      matcher: &[quoted::TokenTree],
                      follow: &TokenSet) -> TokenSet {
    use self::quoted::TokenTree;

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
            s
        };

        // (we build `suffix_first` on demand below; you can tell
        // which cases are supposed to fall through by looking for the
        // initialization of this variable.)
        let suffix_first;

        // First, update `last` so that it corresponds to the set
        // of NT tokens that might end the sequence `... token`.
        match *token {
            TokenTree::Token(..) | TokenTree::MetaVar(..) | TokenTree::MetaVarDecl(..) => {
                let can_be_followed_by_any;
                if let Err(bad_frag) = has_legal_fragment_specifier(sess, features, attrs, token) {
                    let msg = format!("invalid fragment specifier `{}`", bad_frag);
                    sess.span_diagnostic.struct_span_err(token.span(), &msg)
                        .help("valid fragment specifiers are `ident`, `block`, `stmt`, `expr`, \
                              `pat`, `ty`, `literal`, `path`, `meta`, `tt`, `item` and `vis`")
                        .emit();
                    // (This eliminates false positives and duplicates
                    // from error messages.)
                    can_be_followed_by_any = true;
                } else {
                    can_be_followed_by_any = token_can_be_followed_by_any(token);
                }

                if can_be_followed_by_any {
                    // don't need to track tokens that work with any,
                    last.replace_with_irrelevant();
                    // ... and don't need to check tokens that can be
                    // followed by anything against SUFFIX.
                    continue 'each_token;
                } else {
                    last.replace_with(token.clone());
                    suffix_first = build_suffix_first();
                }
            }
            TokenTree::Delimited(span, ref d) => {
                let my_suffix = TokenSet::singleton(d.close_tt(span));
                check_matcher_core(sess, features, attrs, first_sets, &d.tts, &my_suffix);
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
                    new.add_one_maybe(TokenTree::Token(sp, u.clone()));
                    &new
                } else {
                    &suffix_first
                };

                // At this point, `suffix_first` is built, and
                // `my_suffix` is some TokenSet that we can use
                // for checking the interior of `seq_rep`.
                let next = check_matcher_core(sess,
                                              features,
                                              attrs,
                                              first_sets,
                                              &seq_rep.tts,
                                              my_suffix);
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
        'each_last: for token in &last.tokens {
            if let TokenTree::MetaVarDecl(_, ref name, ref frag_spec) = *token {
                for next_token in &suffix_first.tokens {
                    match is_in_follow(next_token, &frag_spec.as_str()) {
                        Err((msg, help)) => {
                            sess.span_diagnostic.struct_span_err(next_token.span(), &msg)
                                .help(help).emit();
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
                                next_token.span(),
                                &format!("`${name}:{frag}` {may_be} followed by `{next}`, which \
                                          is not allowed for `{frag}` fragments",
                                         name=name,
                                         frag=frag_spec,
                                         next=quoted_tt_to_string(next_token),
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

fn token_can_be_followed_by_any(tok: &quoted::TokenTree) -> bool {
    if let quoted::TokenTree::MetaVarDecl(_, _, frag_spec) = *tok {
        frag_can_be_followed_by_any(&frag_spec.as_str())
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
        "item"     | // always terminated by `}` or `;`
        "block"    | // exactly one token tree
        "ident"    | // exactly one token tree
        "literal"  | // exactly one token tree
        "meta"     | // exactly one token tree
        "lifetime" | // exactly one token tree
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
fn is_in_follow(tok: &quoted::TokenTree, frag: &str) -> Result<bool, (String, &'static str)> {
    use self::quoted::TokenTree;

    if let TokenTree::Token(_, token::CloseDelim(_)) = *tok {
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
            "stmt" | "expr"  => match *tok {
                TokenTree::Token(_, ref tok) => match *tok {
                    FatArrow | Comma | Semi => Ok(true),
                    _ => Ok(false)
                },
                _ => Ok(false),
            },
            "pat" => match *tok {
                TokenTree::Token(_, ref tok) => match *tok {
                    FatArrow | Comma | Eq | BinOp(token::Or) => Ok(true),
                    Ident(i, false) if i.name == "if" || i.name == "in" => Ok(true),
                    _ => Ok(false)
                },
                _ => Ok(false),
            },
            "path" | "ty" => match *tok {
                TokenTree::Token(_, ref tok) => match *tok {
                    OpenDelim(token::DelimToken::Brace) | OpenDelim(token::DelimToken::Bracket) |
                    Comma | FatArrow | Colon | Eq | Gt | Semi | BinOp(token::Or) => Ok(true),
                    Ident(i, false) if i.name == "as" || i.name == "where" => Ok(true),
                    _ => Ok(false)
                },
                TokenTree::MetaVarDecl(_, _, frag) if frag.name == "block" => Ok(true),
                _ => Ok(false),
            },
            "ident" | "lifetime" => {
                // being a single token, idents and lifetimes are harmless
                Ok(true)
            },
            "literal" => {
                // literals may be of a single token, or two tokens (negative numbers)
                Ok(true)
            },
            "meta" | "tt" => {
                // being either a single token or a delimited sequence, tt is
                // harmless
                Ok(true)
            },
            "vis" => {
                // Explicitly disallow `priv`, on the off chance it comes back.
                match *tok {
                    TokenTree::Token(_, ref tok) => match *tok {
                        Comma => Ok(true),
                        Ident(i, is_raw) if is_raw || i.name != "priv" => Ok(true),
                        ref tok => Ok(tok.can_begin_type())
                    },
                    TokenTree::MetaVarDecl(_, _, frag) if frag.name == "ident"
                                                       || frag.name == "ty"
                                                       || frag.name == "path" => Ok(true),
                    _ => Ok(false)
                }
            },
            "" => Ok(true), // keywords::Invalid
            _ => Err((format!("invalid fragment specifier `{}`", frag),
                     "valid fragment specifiers are `ident`, `block`, \
                      `stmt`, `expr`, `pat`, `ty`, `path`, `meta`, `tt`, \
                      `literal`, `item` and `vis`"))
        }
    }
}

fn has_legal_fragment_specifier(sess: &ParseSess,
                                features: &Features,
                                attrs: &[ast::Attribute],
                                tok: &quoted::TokenTree) -> Result<(), String> {
    debug!("has_legal_fragment_specifier({:?})", tok);
    if let quoted::TokenTree::MetaVarDecl(_, _, ref frag_spec) = *tok {
        let frag_name = frag_spec.as_str();
        let frag_span = tok.span();
        if !is_legal_fragment_specifier(sess, features, attrs, &frag_name, frag_span) {
            return Err(frag_name.to_string());
        }
    }
    Ok(())
}

fn is_legal_fragment_specifier(sess: &ParseSess,
                               features: &Features,
                               attrs: &[ast::Attribute],
                               frag_name: &str,
                               frag_span: Span) -> bool {
    match frag_name {
        "item" | "block" | "stmt" | "expr" | "pat" | "lifetime" |
        "path" | "ty" | "ident" | "meta" | "tt" | "vis" | "" => true,
        "literal" => {
            if !features.macro_literal_matcher &&
               !attr::contains_name(attrs, "allow_internal_unstable") {
                let explain = feature_gate::EXPLAIN_LITERAL_MATCHER;
                emit_feature_err(sess,
                                 "macro_literal_matcher",
                                 frag_span,
                                 GateIssue::Language,
                                 explain);
            }
            true
        },
        _ => false,
    }
}

fn quoted_tt_to_string(tt: &quoted::TokenTree) -> String {
    match *tt {
        quoted::TokenTree::Token(_, ref tok) => ::print::pprust::token_to_string(tok),
        quoted::TokenTree::MetaVar(_, name) => format!("${}", name),
        quoted::TokenTree::MetaVarDecl(_, name, kind) => format!("${}:{}", name, kind),
        _ => panic!("unexpected quoted::TokenTree::{{Sequence or Delimited}} \
                     in follow set checker"),
    }
}
