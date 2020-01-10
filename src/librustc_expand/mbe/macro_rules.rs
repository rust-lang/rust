use crate::base::{DummyResult, ExtCtxt, MacResult, TTMacroExpander};
use crate::base::{SyntaxExtension, SyntaxExtensionKind};
use crate::expand::{ensure_complete_parse, parse_ast_fragment, AstFragment, AstFragmentKind};
use crate::mbe;
use crate::mbe::macro_check;
use crate::mbe::macro_parser::parse;
use crate::mbe::macro_parser::{Error, Failure, Success};
use crate::mbe::macro_parser::{MatchedNonterminal, MatchedSeq, NamedParseResult};
use crate::mbe::transcribe::transcribe;

use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::sync::Lrc;
use rustc_errors::{Applicability, DiagnosticBuilder, FatalError};
use rustc_feature::Features;
use rustc_parse::parser::Parser;
use rustc_parse::Directory;
use rustc_span::edition::Edition;
use rustc_span::hygiene::Transparency;
use rustc_span::symbol::{kw, sym, Symbol};
use rustc_span::Span;
use syntax::ast;
use syntax::attr::{self, TransparencyError};
use syntax::print::pprust;
use syntax::sess::ParseSess;
use syntax::token::{self, NtTT, Token, TokenKind::*};
use syntax::tokenstream::{DelimSpan, TokenStream};

use log::debug;
use std::borrow::Cow;
use std::collections::hash_map::Entry;
use std::{mem, slice};

const VALID_FRAGMENT_NAMES_MSG: &str = "valid fragment specifiers are \
                                        `ident`, `block`, `stmt`, `expr`, `pat`, `ty`, `lifetime`, \
                                        `literal`, `path`, `meta`, `tt`, `item` and `vis`";

crate struct ParserAnyMacro<'a> {
    parser: Parser<'a>,

    /// Span of the expansion site of the macro this parser is for
    site_span: Span,
    /// The ident of the macro we're parsing
    macro_ident: ast::Ident,
    arm_span: Span,
}

crate fn annotate_err_with_kind(
    err: &mut DiagnosticBuilder<'_>,
    kind: AstFragmentKind,
    span: Span,
) {
    match kind {
        AstFragmentKind::Ty => {
            err.span_label(span, "this macro call doesn't expand to a type");
        }
        AstFragmentKind::Pat => {
            err.span_label(span, "this macro call doesn't expand to a pattern");
        }
        _ => {}
    };
}

/// Instead of e.g. `vec![a, b, c]` in a pattern context, suggest `[a, b, c]`.
fn suggest_slice_pat(e: &mut DiagnosticBuilder<'_>, site_span: Span, parser: &Parser<'_>) {
    let mut suggestion = None;
    if let Ok(code) = parser.sess.source_map().span_to_snippet(site_span) {
        if let Some(bang) = code.find('!') {
            suggestion = Some(code[bang + 1..].to_string());
        }
    }
    if let Some(suggestion) = suggestion {
        e.span_suggestion(
            site_span,
            "use a slice pattern here instead",
            suggestion,
            Applicability::MachineApplicable,
        );
    } else {
        e.span_label(site_span, "use a slice pattern here instead");
    }
    e.help(
        "for more information, see https://doc.rust-lang.org/edition-guide/\
        rust-2018/slice-patterns.html",
    );
}

impl<'a> ParserAnyMacro<'a> {
    crate fn make(mut self: Box<ParserAnyMacro<'a>>, kind: AstFragmentKind) -> AstFragment {
        let ParserAnyMacro { site_span, macro_ident, ref mut parser, arm_span } = *self;
        let fragment = panictry!(parse_ast_fragment(parser, kind, true).map_err(|mut e| {
            if parser.token == token::Eof && e.message().ends_with(", found `<eof>`") {
                if !e.span.is_dummy() {
                    // early end of macro arm (#52866)
                    e.replace_span_with(parser.sess.source_map().next_point(parser.token.span));
                }
                let msg = &e.message[0];
                e.message[0] = (
                    format!(
                        "macro expansion ends with an incomplete expression: {}",
                        msg.0.replace(", found `<eof>`", ""),
                    ),
                    msg.1,
                );
            }
            if e.span.is_dummy() {
                // Get around lack of span in error (#30128)
                e.replace_span_with(site_span);
                if parser.sess.source_map().span_to_filename(arm_span).is_real() {
                    e.span_label(arm_span, "in this macro arm");
                }
            } else if !parser.sess.source_map().span_to_filename(parser.token.span).is_real() {
                e.span_label(site_span, "in this macro invocation");
            }
            match kind {
                AstFragmentKind::Pat if macro_ident.name == sym::vec => {
                    suggest_slice_pat(&mut e, site_span, parser);
                }
                _ => annotate_err_with_kind(&mut e, kind, site_span),
            };
            e
        }));

        // We allow semicolons at the end of expressions -- e.g., the semicolon in
        // `macro_rules! m { () => { panic!(); } }` isn't parsed by `.parse_expr()`,
        // but `m!()` is allowed in expression positions (cf. issue #34706).
        if kind == AstFragmentKind::Expr && parser.token == token::Semi {
            parser.bump();
        }

        // Make sure we don't have any tokens left to parse so we don't silently drop anything.
        let path = ast::Path::from_ident(macro_ident.with_span_pos(site_span));
        ensure_complete_parse(parser, &path, kind.name(), site_span);
        fragment
    }
}

struct MacroRulesMacroExpander {
    name: ast::Ident,
    span: Span,
    transparency: Transparency,
    lhses: Vec<mbe::TokenTree>,
    rhses: Vec<mbe::TokenTree>,
    valid: bool,
}

impl TTMacroExpander for MacroRulesMacroExpander {
    fn expand<'cx>(
        &self,
        cx: &'cx mut ExtCtxt<'_>,
        sp: Span,
        input: TokenStream,
    ) -> Box<dyn MacResult + 'cx> {
        if !self.valid {
            return DummyResult::any(sp);
        }
        generic_extension(
            cx,
            sp,
            self.span,
            self.name,
            self.transparency,
            input,
            &self.lhses,
            &self.rhses,
        )
    }
}

fn trace_macros_note(cx: &mut ExtCtxt<'_>, sp: Span, message: String) {
    let sp = sp.macro_backtrace().last().map(|trace| trace.call_site).unwrap_or(sp);
    cx.expansions.entry(sp).or_default().push(message);
}

/// Given `lhses` and `rhses`, this is the new macro we create
fn generic_extension<'cx>(
    cx: &'cx mut ExtCtxt<'_>,
    sp: Span,
    def_span: Span,
    name: ast::Ident,
    transparency: Transparency,
    arg: TokenStream,
    lhses: &[mbe::TokenTree],
    rhses: &[mbe::TokenTree],
) -> Box<dyn MacResult + 'cx> {
    if cx.trace_macros() {
        let msg = format!("expanding `{}! {{ {} }}`", name, pprust::tts_to_string(arg.clone()));
        trace_macros_note(cx, sp, msg);
    }

    // Which arm's failure should we report? (the one furthest along)
    let mut best_failure: Option<(Token, &str)> = None;
    for (i, lhs) in lhses.iter().enumerate() {
        // try each arm's matchers
        let lhs_tt = match *lhs {
            mbe::TokenTree::Delimited(_, ref delim) => &delim.tts[..],
            _ => cx.span_bug(sp, "malformed macro lhs"),
        };

        // Take a snapshot of the state of pre-expansion gating at this point.
        // This is used so that if a matcher is not `Success(..)`ful,
        // then the spans which became gated when parsing the unsuccessful matcher
        // are not recorded. On the first `Success(..)`ful matcher, the spans are merged.
        let mut gated_spans_snaphot = mem::take(&mut *cx.parse_sess.gated_spans.spans.borrow_mut());

        match parse_tt(cx, lhs_tt, arg.clone()) {
            Success(named_matches) => {
                // The matcher was `Success(..)`ful.
                // Merge the gated spans from parsing the matcher with the pre-existing ones.
                cx.parse_sess.gated_spans.merge(gated_spans_snaphot);

                let rhs = match rhses[i] {
                    // ignore delimiters
                    mbe::TokenTree::Delimited(_, ref delimed) => delimed.tts.clone(),
                    _ => cx.span_bug(sp, "malformed macro rhs"),
                };
                let arm_span = rhses[i].span();

                let rhs_spans = rhs.iter().map(|t| t.span()).collect::<Vec<_>>();
                // rhs has holes ( `$id` and `$(...)` that need filled)
                let mut tts = transcribe(cx, &named_matches, rhs, transparency);

                // Replace all the tokens for the corresponding positions in the macro, to maintain
                // proper positions in error reporting, while maintaining the macro_backtrace.
                if rhs_spans.len() == tts.len() {
                    tts = tts.map_enumerated(|i, mut tt| {
                        let mut sp = rhs_spans[i];
                        sp = sp.with_ctxt(tt.span().ctxt());
                        tt.set_span(sp);
                        tt
                    });
                }

                if cx.trace_macros() {
                    let msg = format!("to `{}`", pprust::tts_to_string(tts.clone()));
                    trace_macros_note(cx, sp, msg);
                }

                let directory = Directory {
                    path: Cow::from(cx.current_expansion.module.directory.as_path()),
                    ownership: cx.current_expansion.directory_ownership,
                };
                let mut p = Parser::new(cx.parse_sess(), tts, Some(directory), true, false, None);
                p.root_module_name =
                    cx.current_expansion.module.mod_path.last().map(|id| id.to_string());
                p.last_type_ascription = cx.current_expansion.prior_type_ascription;

                p.process_potential_macro_variable();
                // Let the context choose how to interpret the result.
                // Weird, but useful for X-macros.
                return Box::new(ParserAnyMacro {
                    parser: p,

                    // Pass along the original expansion site and the name of the macro
                    // so we can print a useful error message if the parse of the expanded
                    // macro leaves unparsed tokens.
                    site_span: sp,
                    macro_ident: name,
                    arm_span,
                });
            }
            Failure(token, msg) => match best_failure {
                Some((ref best_token, _)) if best_token.span.lo() >= token.span.lo() => {}
                _ => best_failure = Some((token, msg)),
            },
            Error(err_sp, ref msg) => cx.span_fatal(err_sp.substitute_dummy(sp), &msg[..]),
        }

        // The matcher was not `Success(..)`ful.
        // Restore to the state before snapshotting and maybe try again.
        mem::swap(&mut gated_spans_snaphot, &mut cx.parse_sess.gated_spans.spans.borrow_mut());
    }

    let (token, label) = best_failure.expect("ran no matchers");
    let span = token.span.substitute_dummy(sp);
    let mut err = cx.struct_span_err(span, &parse_failure_msg(&token));
    err.span_label(span, label);
    if !def_span.is_dummy() && cx.source_map().span_to_filename(def_span).is_real() {
        err.span_label(cx.source_map().def_span(def_span), "when calling this macro");
    }

    // Check whether there's a missing comma in this macro call, like `println!("{}" a);`
    if let Some((arg, comma_span)) = arg.add_comma() {
        for lhs in lhses {
            // try each arm's matchers
            let lhs_tt = match *lhs {
                mbe::TokenTree::Delimited(_, ref delim) => &delim.tts[..],
                _ => continue,
            };
            match parse_tt(cx, lhs_tt, arg.clone()) {
                Success(_) => {
                    if comma_span.is_dummy() {
                        err.note("you might be missing a comma");
                    } else {
                        err.span_suggestion_short(
                            comma_span,
                            "missing comma here",
                            ", ".to_string(),
                            Applicability::MachineApplicable,
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

/// Converts a macro item into a syntax extension.
pub fn compile_declarative_macro(
    sess: &ParseSess,
    features: &Features,
    def: &ast::Item,
    edition: Edition,
) -> SyntaxExtension {
    let diag = &sess.span_diagnostic;
    let lhs_nm = ast::Ident::new(sym::lhs, def.span);
    let rhs_nm = ast::Ident::new(sym::rhs, def.span);
    let tt_spec = ast::Ident::new(sym::tt, def.span);

    // Parse the macro_rules! invocation
    let (is_legacy, body) = match &def.kind {
        ast::ItemKind::MacroDef(macro_def) => (macro_def.legacy, macro_def.body.inner_tokens()),
        _ => unreachable!(),
    };

    // The pattern that macro_rules matches.
    // The grammar for macro_rules! is:
    // $( $lhs:tt => $rhs:tt );+
    // ...quasiquoting this would be nice.
    // These spans won't matter, anyways
    let argument_gram = vec![
        mbe::TokenTree::Sequence(
            DelimSpan::dummy(),
            Lrc::new(mbe::SequenceRepetition {
                tts: vec![
                    mbe::TokenTree::MetaVarDecl(def.span, lhs_nm, tt_spec),
                    mbe::TokenTree::token(token::FatArrow, def.span),
                    mbe::TokenTree::MetaVarDecl(def.span, rhs_nm, tt_spec),
                ],
                separator: Some(Token::new(
                    if is_legacy { token::Semi } else { token::Comma },
                    def.span,
                )),
                kleene: mbe::KleeneToken::new(mbe::KleeneOp::OneOrMore, def.span),
                num_captures: 2,
            }),
        ),
        // to phase into semicolon-termination instead of semicolon-separation
        mbe::TokenTree::Sequence(
            DelimSpan::dummy(),
            Lrc::new(mbe::SequenceRepetition {
                tts: vec![mbe::TokenTree::token(
                    if is_legacy { token::Semi } else { token::Comma },
                    def.span,
                )],
                separator: None,
                kleene: mbe::KleeneToken::new(mbe::KleeneOp::ZeroOrMore, def.span),
                num_captures: 0,
            }),
        ),
    ];

    let argument_map = match parse(sess, body, &argument_gram, None, true) {
        Success(m) => m,
        Failure(token, msg) => {
            let s = parse_failure_msg(&token);
            let sp = token.span.substitute_dummy(def.span);
            let mut err = sess.span_diagnostic.struct_span_fatal(sp, &s);
            err.span_label(sp, msg);
            err.emit();
            FatalError.raise();
        }
        Error(sp, s) => {
            sess.span_diagnostic.span_fatal(sp.substitute_dummy(def.span), &s).raise();
        }
    };

    let mut valid = true;

    // Extract the arguments:
    let lhses = match argument_map[&lhs_nm] {
        MatchedSeq(ref s) => s
            .iter()
            .map(|m| {
                if let MatchedNonterminal(ref nt) = *m {
                    if let NtTT(ref tt) = **nt {
                        let tt = mbe::quoted::parse(tt.clone().into(), true, sess).pop().unwrap();
                        valid &= check_lhs_nt_follows(sess, features, &def.attrs, &tt);
                        return tt;
                    }
                }
                sess.span_diagnostic.span_bug(def.span, "wrong-structured lhs")
            })
            .collect::<Vec<mbe::TokenTree>>(),
        _ => sess.span_diagnostic.span_bug(def.span, "wrong-structured lhs"),
    };

    let rhses = match argument_map[&rhs_nm] {
        MatchedSeq(ref s) => s
            .iter()
            .map(|m| {
                if let MatchedNonterminal(ref nt) = *m {
                    if let NtTT(ref tt) = **nt {
                        return mbe::quoted::parse(tt.clone().into(), false, sess).pop().unwrap();
                    }
                }
                sess.span_diagnostic.span_bug(def.span, "wrong-structured lhs")
            })
            .collect::<Vec<mbe::TokenTree>>(),
        _ => sess.span_diagnostic.span_bug(def.span, "wrong-structured rhs"),
    };

    for rhs in &rhses {
        valid &= check_rhs(sess, rhs);
    }

    // don't abort iteration early, so that errors for multiple lhses can be reported
    for lhs in &lhses {
        valid &= check_lhs_no_empty_seq(sess, slice::from_ref(lhs));
    }

    // We use CRATE_NODE_ID instead of `def.id` otherwise we may emit buffered lints for a node id
    // that is not lint-checked and trigger the "failed to process buffered lint here" bug.
    valid &= macro_check::check_meta_variables(sess, ast::CRATE_NODE_ID, def.span, &lhses, &rhses);

    let (transparency, transparency_error) = attr::find_transparency(&def.attrs, is_legacy);
    match transparency_error {
        Some(TransparencyError::UnknownTransparency(value, span)) => {
            diag.span_err(span, &format!("unknown macro transparency: `{}`", value))
        }
        Some(TransparencyError::MultipleTransparencyAttrs(old_span, new_span)) => {
            diag.span_err(vec![old_span, new_span], "multiple macro transparency attributes")
        }
        None => {}
    }

    let expander: Box<_> = Box::new(MacroRulesMacroExpander {
        name: def.ident,
        span: def.span,
        transparency,
        lhses,
        rhses,
        valid,
    });

    SyntaxExtension::new(
        sess,
        SyntaxExtensionKind::LegacyBang(expander),
        def.span,
        Vec::new(),
        edition,
        def.ident.name,
        &def.attrs,
    )
}

fn check_lhs_nt_follows(
    sess: &ParseSess,
    features: &Features,
    attrs: &[ast::Attribute],
    lhs: &mbe::TokenTree,
) -> bool {
    // lhs is going to be like TokenTree::Delimited(...), where the
    // entire lhs is those tts. Or, it can be a "bare sequence", not wrapped in parens.
    if let mbe::TokenTree::Delimited(_, ref tts) = *lhs {
        check_matcher(sess, features, attrs, &tts.tts)
    } else {
        let msg = "invalid macro matcher; matchers must be contained in balanced delimiters";
        sess.span_diagnostic.span_err(lhs.span(), msg);
        false
    }
    // we don't abort on errors on rejection, the driver will do that for us
    // after parsing/expansion. we can report every error in every macro this way.
}

/// Checks that the lhs contains no repetition which could match an empty token
/// tree, because then the matcher would hang indefinitely.
fn check_lhs_no_empty_seq(sess: &ParseSess, tts: &[mbe::TokenTree]) -> bool {
    use mbe::TokenTree;
    for tt in tts {
        match *tt {
            TokenTree::Token(..) | TokenTree::MetaVar(..) | TokenTree::MetaVarDecl(..) => (),
            TokenTree::Delimited(_, ref del) => {
                if !check_lhs_no_empty_seq(sess, &del.tts) {
                    return false;
                }
            }
            TokenTree::Sequence(span, ref seq) => {
                if seq.separator.is_none()
                    && seq.tts.iter().all(|seq_tt| match *seq_tt {
                        TokenTree::MetaVarDecl(_, _, id) => id.name == sym::vis,
                        TokenTree::Sequence(_, ref sub_seq) => {
                            sub_seq.kleene.op == mbe::KleeneOp::ZeroOrMore
                                || sub_seq.kleene.op == mbe::KleeneOp::ZeroOrOne
                        }
                        _ => false,
                    })
                {
                    let sp = span.entire();
                    sess.span_diagnostic.span_err(sp, "repetition matches empty token tree");
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

fn check_rhs(sess: &ParseSess, rhs: &mbe::TokenTree) -> bool {
    match *rhs {
        mbe::TokenTree::Delimited(..) => return true,
        _ => sess.span_diagnostic.span_err(rhs.span(), "macro rhs must be delimited"),
    }
    false
}

fn check_matcher(
    sess: &ParseSess,
    features: &Features,
    attrs: &[ast::Attribute],
    matcher: &[mbe::TokenTree],
) -> bool {
    let first_sets = FirstSets::new(matcher);
    let empty_suffix = TokenSet::empty();
    let err = sess.span_diagnostic.err_count();
    check_matcher_core(sess, features, attrs, &first_sets, matcher, &empty_suffix);
    err == sess.span_diagnostic.err_count()
}

// `The FirstSets` for a matcher is a mapping from subsequences in the
// matcher to the FIRST set for that subsequence.
//
// This mapping is partially precomputed via a backwards scan over the
// token trees of the matcher, which provides a mapping from each
// repetition sequence to its *first* set.
//
// (Hypothetically, sequences should be uniquely identifiable via their
// spans, though perhaps that is false, e.g., for macro-generated macros
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
    first: FxHashMap<Span, Option<TokenSet>>,
}

impl FirstSets {
    fn new(tts: &[mbe::TokenTree]) -> FirstSets {
        use mbe::TokenTree;

        let mut sets = FirstSets { first: FxHashMap::default() };
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

                        match sets.first.entry(sp.entire()) {
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

                        if let (Some(sep), true) = (&seq_rep.separator, subfirst.maybe_empty) {
                            first.add_one_maybe(TokenTree::Token(sep.clone()));
                        }

                        // Reverse scan: Sequence comes before `first`.
                        if subfirst.maybe_empty
                            || seq_rep.kleene.op == mbe::KleeneOp::ZeroOrMore
                            || seq_rep.kleene.op == mbe::KleeneOp::ZeroOrOne
                        {
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
    fn first(&self, tts: &[mbe::TokenTree]) -> TokenSet {
        use mbe::TokenTree;

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
                    let subfirst_owned;
                    let subfirst = match self.first.get(&sp.entire()) {
                        Some(&Some(ref subfirst)) => subfirst,
                        Some(&None) => {
                            subfirst_owned = self.first(&seq_rep.tts[..]);
                            &subfirst_owned
                        }
                        None => {
                            panic!("We missed a sequence during FirstSets construction");
                        }
                    };

                    // If the sequence contents can be empty, then the first
                    // token could be the separator token itself.
                    if let (Some(sep), true) = (&seq_rep.separator, subfirst.maybe_empty) {
                        first.add_one_maybe(TokenTree::Token(sep.clone()));
                    }

                    assert!(first.maybe_empty);
                    first.add_all(subfirst);
                    if subfirst.maybe_empty
                        || seq_rep.kleene.op == mbe::KleeneOp::ZeroOrMore
                        || seq_rep.kleene.op == mbe::KleeneOp::ZeroOrOne
                    {
                        // Continue scanning for more first
                        // tokens, but also make sure we
                        // restore empty-tracking state.
                        first.maybe_empty = true;
                        continue;
                    } else {
                        return first;
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

// A set of `mbe::TokenTree`s, which may include `TokenTree::Match`s
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
    tokens: Vec<mbe::TokenTree>,
    maybe_empty: bool,
}

impl TokenSet {
    // Returns a set for the empty sequence.
    fn empty() -> Self {
        TokenSet { tokens: Vec::new(), maybe_empty: true }
    }

    // Returns the set `{ tok }` for the single-token (and thus
    // non-empty) sequence [tok].
    fn singleton(tok: mbe::TokenTree) -> Self {
        TokenSet { tokens: vec![tok], maybe_empty: false }
    }

    // Changes self to be the set `{ tok }`.
    // Since `tok` is always present, marks self as non-empty.
    fn replace_with(&mut self, tok: mbe::TokenTree) {
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
    fn add_one(&mut self, tok: mbe::TokenTree) {
        if !self.tokens.contains(&tok) {
            self.tokens.push(tok);
        }
        self.maybe_empty = false;
    }

    // Adds `tok` to the set for `self`. (Leaves `maybe_empty` flag alone.)
    fn add_one_maybe(&mut self, tok: mbe::TokenTree) {
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
// can legally be followed by a token `N`, for all `N` in `follow`.
// (If `follow` is empty, then it imposes no constraint on
// the `matcher`.)
//
// Returns the set of NT tokens that could possibly come last in
// `matcher`. (If `matcher` matches the empty sequence, then
// `maybe_empty` will be set to true.)
//
// Requires that `first_sets` is pre-computed for `matcher`;
// see `FirstSets::new`.
fn check_matcher_core(
    sess: &ParseSess,
    features: &Features,
    attrs: &[ast::Attribute],
    first_sets: &FirstSets,
    matcher: &[mbe::TokenTree],
    follow: &TokenSet,
) -> TokenSet {
    use mbe::TokenTree;

    let mut last = TokenSet::empty();

    // 2. For each token and suffix  [T, SUFFIX] in M:
    // ensure that T can be followed by SUFFIX, and if SUFFIX may be empty,
    // then ensure T can also be followed by any element of FOLLOW.
    'each_token: for i in 0..matcher.len() {
        let token = &matcher[i];
        let suffix = &matcher[i + 1..];

        let build_suffix_first = || {
            let mut s = first_sets.first(suffix);
            if s.maybe_empty {
                s.add_all(follow);
            }
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
                    sess.span_diagnostic
                        .struct_span_err(token.span(), &msg)
                        .help(VALID_FRAGMENT_NAMES_MSG)
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
            TokenTree::Sequence(_, ref seq_rep) => {
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
                let my_suffix = if let Some(sep) = &seq_rep.separator {
                    new = suffix_first.clone();
                    new.add_one_maybe(TokenTree::Token(sep.clone()));
                    &new
                } else {
                    &suffix_first
                };

                // At this point, `suffix_first` is built, and
                // `my_suffix` is some TokenSet that we can use
                // for checking the interior of `seq_rep`.
                let next =
                    check_matcher_core(sess, features, attrs, first_sets, &seq_rep.tts, my_suffix);
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
            if let TokenTree::MetaVarDecl(_, name, frag_spec) = *token {
                for next_token in &suffix_first.tokens {
                    match is_in_follow(next_token, frag_spec.name) {
                        IsInFollow::Invalid(msg, help) => {
                            sess.span_diagnostic
                                .struct_span_err(next_token.span(), &msg)
                                .help(help)
                                .emit();
                            // don't bother reporting every source of
                            // conflict for a particular element of `last`.
                            continue 'each_last;
                        }
                        IsInFollow::Yes => {}
                        IsInFollow::No(possible) => {
                            let may_be = if last.tokens.len() == 1 && suffix_first.tokens.len() == 1
                            {
                                "is"
                            } else {
                                "may be"
                            };

                            let sp = next_token.span();
                            let mut err = sess.span_diagnostic.struct_span_err(
                                sp,
                                &format!(
                                    "`${name}:{frag}` {may_be} followed by `{next}`, which \
                                     is not allowed for `{frag}` fragments",
                                    name = name,
                                    frag = frag_spec,
                                    next = quoted_tt_to_string(next_token),
                                    may_be = may_be
                                ),
                            );
                            err.span_label(
                                sp,
                                format!("not allowed after `{}` fragments", frag_spec),
                            );
                            let msg = "allowed there are: ";
                            match possible {
                                &[] => {}
                                &[t] => {
                                    err.note(&format!(
                                        "only {} is allowed after `{}` fragments",
                                        t, frag_spec,
                                    ));
                                }
                                ts => {
                                    err.note(&format!(
                                        "{}{} or {}",
                                        msg,
                                        ts[..ts.len() - 1]
                                            .iter()
                                            .map(|s| *s)
                                            .collect::<Vec<_>>()
                                            .join(", "),
                                        ts[ts.len() - 1],
                                    ));
                                }
                            }
                            err.emit();
                        }
                    }
                }
            }
        }
    }
    last
}

fn token_can_be_followed_by_any(tok: &mbe::TokenTree) -> bool {
    if let mbe::TokenTree::MetaVarDecl(_, _, frag_spec) = *tok {
        frag_can_be_followed_by_any(frag_spec.name)
    } else {
        // (Non NT's can always be followed by anthing in matchers.)
        true
    }
}

/// Returns `true` if a fragment of type `frag` can be followed by any sort of
/// token. We use this (among other things) as a useful approximation
/// for when `frag` can be followed by a repetition like `$(...)*` or
/// `$(...)+`. In general, these can be a bit tricky to reason about,
/// so we adopt a conservative position that says that any fragment
/// specifier which consumes at most one token tree can be followed by
/// a fragment specifier (indeed, these fragments can be followed by
/// ANYTHING without fear of future compatibility hazards).
fn frag_can_be_followed_by_any(frag: Symbol) -> bool {
    match frag {
        sym::item     | // always terminated by `}` or `;`
        sym::block    | // exactly one token tree
        sym::ident    | // exactly one token tree
        sym::literal  | // exactly one token tree
        sym::meta     | // exactly one token tree
        sym::lifetime | // exactly one token tree
        sym::tt =>   // exactly one token tree
            true,

        _ =>
            false,
    }
}

enum IsInFollow {
    Yes,
    No(&'static [&'static str]),
    Invalid(String, &'static str),
}

/// Returns `true` if `frag` can legally be followed by the token `tok`. For
/// fragments that can consume an unbounded number of tokens, `tok`
/// must be within a well-defined follow set. This is intended to
/// guarantee future compatibility: for example, without this rule, if
/// we expanded `expr` to include a new binary operator, we might
/// break macros that were relying on that binary operator as a
/// separator.
// when changing this do not forget to update doc/book/macros.md!
fn is_in_follow(tok: &mbe::TokenTree, frag: Symbol) -> IsInFollow {
    use mbe::TokenTree;

    if let TokenTree::Token(Token { kind: token::CloseDelim(_), .. }) = *tok {
        // closing a token tree can never be matched by any fragment;
        // iow, we always require that `(` and `)` match, etc.
        IsInFollow::Yes
    } else {
        match frag {
            sym::item => {
                // since items *must* be followed by either a `;` or a `}`, we can
                // accept anything after them
                IsInFollow::Yes
            }
            sym::block => {
                // anything can follow block, the braces provide an easy boundary to
                // maintain
                IsInFollow::Yes
            }
            sym::stmt | sym::expr => {
                const TOKENS: &[&str] = &["`=>`", "`,`", "`;`"];
                match tok {
                    TokenTree::Token(token) => match token.kind {
                        FatArrow | Comma | Semi => IsInFollow::Yes,
                        _ => IsInFollow::No(TOKENS),
                    },
                    _ => IsInFollow::No(TOKENS),
                }
            }
            sym::pat => {
                const TOKENS: &[&str] = &["`=>`", "`,`", "`=`", "`|`", "`if`", "`in`"];
                match tok {
                    TokenTree::Token(token) => match token.kind {
                        FatArrow | Comma | Eq | BinOp(token::Or) => IsInFollow::Yes,
                        Ident(name, false) if name == kw::If || name == kw::In => IsInFollow::Yes,
                        _ => IsInFollow::No(TOKENS),
                    },
                    _ => IsInFollow::No(TOKENS),
                }
            }
            sym::path | sym::ty => {
                const TOKENS: &[&str] = &[
                    "`{`", "`[`", "`=>`", "`,`", "`>`", "`=`", "`:`", "`;`", "`|`", "`as`",
                    "`where`",
                ];
                match tok {
                    TokenTree::Token(token) => match token.kind {
                        OpenDelim(token::DelimToken::Brace)
                        | OpenDelim(token::DelimToken::Bracket)
                        | Comma
                        | FatArrow
                        | Colon
                        | Eq
                        | Gt
                        | BinOp(token::Shr)
                        | Semi
                        | BinOp(token::Or) => IsInFollow::Yes,
                        Ident(name, false) if name == kw::As || name == kw::Where => {
                            IsInFollow::Yes
                        }
                        _ => IsInFollow::No(TOKENS),
                    },
                    TokenTree::MetaVarDecl(_, _, frag) if frag.name == sym::block => {
                        IsInFollow::Yes
                    }
                    _ => IsInFollow::No(TOKENS),
                }
            }
            sym::ident | sym::lifetime => {
                // being a single token, idents and lifetimes are harmless
                IsInFollow::Yes
            }
            sym::literal => {
                // literals may be of a single token, or two tokens (negative numbers)
                IsInFollow::Yes
            }
            sym::meta | sym::tt => {
                // being either a single token or a delimited sequence, tt is
                // harmless
                IsInFollow::Yes
            }
            sym::vis => {
                // Explicitly disallow `priv`, on the off chance it comes back.
                const TOKENS: &[&str] = &["`,`", "an ident", "a type"];
                match tok {
                    TokenTree::Token(token) => match token.kind {
                        Comma => IsInFollow::Yes,
                        Ident(name, is_raw) if is_raw || name != kw::Priv => IsInFollow::Yes,
                        _ => {
                            if token.can_begin_type() {
                                IsInFollow::Yes
                            } else {
                                IsInFollow::No(TOKENS)
                            }
                        }
                    },
                    TokenTree::MetaVarDecl(_, _, frag)
                        if frag.name == sym::ident
                            || frag.name == sym::ty
                            || frag.name == sym::path =>
                    {
                        IsInFollow::Yes
                    }
                    _ => IsInFollow::No(TOKENS),
                }
            }
            kw::Invalid => IsInFollow::Yes,
            _ => IsInFollow::Invalid(
                format!("invalid fragment specifier `{}`", frag),
                VALID_FRAGMENT_NAMES_MSG,
            ),
        }
    }
}

fn has_legal_fragment_specifier(
    sess: &ParseSess,
    features: &Features,
    attrs: &[ast::Attribute],
    tok: &mbe::TokenTree,
) -> Result<(), String> {
    debug!("has_legal_fragment_specifier({:?})", tok);
    if let mbe::TokenTree::MetaVarDecl(_, _, ref frag_spec) = *tok {
        let frag_span = tok.span();
        if !is_legal_fragment_specifier(sess, features, attrs, frag_spec.name, frag_span) {
            return Err(frag_spec.to_string());
        }
    }
    Ok(())
}

fn is_legal_fragment_specifier(
    _sess: &ParseSess,
    _features: &Features,
    _attrs: &[ast::Attribute],
    frag_name: Symbol,
    _frag_span: Span,
) -> bool {
    /*
     * If new fragment specifiers are invented in nightly, `_sess`,
     * `_features`, `_attrs`, and `_frag_span` will be useful here
     * for checking against feature gates. See past versions of
     * this function.
     */
    match frag_name {
        sym::item
        | sym::block
        | sym::stmt
        | sym::expr
        | sym::pat
        | sym::lifetime
        | sym::path
        | sym::ty
        | sym::ident
        | sym::meta
        | sym::tt
        | sym::vis
        | sym::literal
        | kw::Invalid => true,
        _ => false,
    }
}

fn quoted_tt_to_string(tt: &mbe::TokenTree) -> String {
    match *tt {
        mbe::TokenTree::Token(ref token) => pprust::token_to_string(&token),
        mbe::TokenTree::MetaVar(_, name) => format!("${}", name),
        mbe::TokenTree::MetaVarDecl(_, name, kind) => format!("${}:{}", name, kind),
        _ => panic!(
            "unexpected mbe::TokenTree::{{Sequence or Delimited}} \
             in follow set checker"
        ),
    }
}

/// Use this token tree as a matcher to parse given tts.
fn parse_tt(cx: &ExtCtxt<'_>, mtch: &[mbe::TokenTree], tts: TokenStream) -> NamedParseResult {
    // `None` is because we're not interpolating
    let directory = Directory {
        path: Cow::from(cx.current_expansion.module.directory.as_path()),
        ownership: cx.current_expansion.directory_ownership,
    };
    parse(cx.parse_sess(), tts, mtch, Some(directory), true)
}

/// Generates an appropriate parsing failure message. For EOF, this is "unexpected end...". For
/// other tokens, this is "unexpected token...".
fn parse_failure_msg(tok: &Token) -> String {
    match tok.kind {
        token::Eof => "unexpected end of macro invocation".to_string(),
        _ => format!("no rules expected the token `{}`", pprust::token_to_string(tok),),
    }
}
