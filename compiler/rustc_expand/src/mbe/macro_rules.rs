use crate::base::{DummyResult, ExtCtxt, MacResult, TTMacroExpander};
use crate::base::{SyntaxExtension, SyntaxExtensionKind};
use crate::expand::{ensure_complete_parse, parse_ast_fragment, AstFragment, AstFragmentKind};
use crate::mbe;
use crate::mbe::macro_check;
use crate::mbe::macro_parser::parse_tt;
use crate::mbe::macro_parser::{Error, ErrorReported, Failure, Success};
use crate::mbe::macro_parser::{MatchedNonterminal, MatchedSeq};
use crate::mbe::transcribe::transcribe;

use rustc_ast as ast;
use rustc_ast::token::{self, NonterminalKind, NtTT, Token, TokenKind::*};
use rustc_ast::tokenstream::{DelimSpan, TokenStream};
use rustc_ast_pretty::pprust;
use rustc_attr::{self as attr, TransparencyError};
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::sync::Lrc;
use rustc_errors::{Applicability, DiagnosticBuilder};
use rustc_feature::Features;
use rustc_parse::parser::Parser;
use rustc_session::parse::ParseSess;
use rustc_session::Session;
use rustc_span::edition::Edition;
use rustc_span::hygiene::Transparency;
use rustc_span::symbol::{kw, sym, Ident, MacroRulesNormalizedIdent};
use rustc_span::Span;

use std::borrow::Cow;
use std::collections::hash_map::Entry;
use std::{mem, slice};
use tracing::debug;

crate struct ParserAnyMacro<'a> {
    parser: Parser<'a>,

    /// Span of the expansion site of the macro this parser is for
    site_span: Span,
    /// The ident of the macro we're parsing
    macro_ident: Ident,
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

fn emit_frag_parse_err(
    mut e: DiagnosticBuilder<'_>,
    parser: &Parser<'_>,
    orig_parser: &mut Parser<'_>,
    site_span: Span,
    macro_ident: Ident,
    arm_span: Span,
    kind: AstFragmentKind,
) {
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
        if !parser.sess.source_map().is_imported(arm_span) {
            e.span_label(arm_span, "in this macro arm");
        }
    } else if parser.sess.source_map().is_imported(parser.token.span) {
        e.span_label(site_span, "in this macro invocation");
    }
    match kind {
        AstFragmentKind::Pat if macro_ident.name == sym::vec => {
            suggest_slice_pat(&mut e, site_span, parser);
        }
        // Try a statement if an expression is wanted but failed and suggest adding `;` to call.
        AstFragmentKind::Expr => match parse_ast_fragment(orig_parser, AstFragmentKind::Stmts) {
            Err(mut err) => err.cancel(),
            Ok(_) => {
                e.note(
                    "the macro call doesn't expand to an expression, but it can expand to a statement",
                );
                e.span_suggestion_verbose(
                    site_span.shrink_to_hi(),
                    "add `;` to interpret the expansion as a statement",
                    ";".to_string(),
                    Applicability::MaybeIncorrect,
                );
            }
        },
        _ => annotate_err_with_kind(&mut e, kind, site_span),
    };
    e.emit();
}

impl<'a> ParserAnyMacro<'a> {
    crate fn make(mut self: Box<ParserAnyMacro<'a>>, kind: AstFragmentKind) -> AstFragment {
        let ParserAnyMacro { site_span, macro_ident, ref mut parser, arm_span } = *self;
        let snapshot = &mut parser.clone();
        let fragment = match parse_ast_fragment(parser, kind) {
            Ok(f) => f,
            Err(err) => {
                emit_frag_parse_err(err, parser, snapshot, site_span, macro_ident, arm_span, kind);
                return kind.dummy(site_span);
            }
        };

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
    name: Ident,
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

fn macro_rules_dummy_expander<'cx>(
    _: &'cx mut ExtCtxt<'_>,
    span: Span,
    _: TokenStream,
) -> Box<dyn MacResult + 'cx> {
    DummyResult::any(span)
}

fn trace_macros_note(cx_expansions: &mut FxHashMap<Span, Vec<String>>, sp: Span, message: String) {
    let sp = sp.macro_backtrace().last().map(|trace| trace.call_site).unwrap_or(sp);
    cx_expansions.entry(sp).or_default().push(message);
}

/// Given `lhses` and `rhses`, this is the new macro we create
fn generic_extension<'cx>(
    cx: &'cx mut ExtCtxt<'_>,
    sp: Span,
    def_span: Span,
    name: Ident,
    transparency: Transparency,
    arg: TokenStream,
    lhses: &[mbe::TokenTree],
    rhses: &[mbe::TokenTree],
) -> Box<dyn MacResult + 'cx> {
    let sess = &cx.sess.parse_sess;

    if cx.trace_macros() {
        let msg = format!("expanding `{}! {{ {} }}`", name, pprust::tts_to_string(&arg));
        trace_macros_note(&mut cx.expansions, sp, msg);
    }

    // Which arm's failure should we report? (the one furthest along)
    let mut best_failure: Option<(Token, &str)> = None;

    // We create a base parser that can be used for the "black box" parts.
    // Every iteration needs a fresh copy of that parser. However, the parser
    // is not mutated on many of the iterations, particularly when dealing with
    // macros like this:
    //
    // macro_rules! foo {
    //     ("a") => (A);
    //     ("b") => (B);
    //     ("c") => (C);
    //     // ... etc. (maybe hundreds more)
    // }
    //
    // as seen in the `html5ever` benchmark. We use a `Cow` so that the base
    // parser is only cloned when necessary (upon mutation). Furthermore, we
    // reinitialize the `Cow` with the base parser at the start of every
    // iteration, so that any mutated parsers are not reused. This is all quite
    // hacky, but speeds up the `html5ever` benchmark significantly. (Issue
    // 68836 suggests a more comprehensive but more complex change to deal with
    // this situation.)
    let parser = parser_from_cx(sess, arg.clone());

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
        let mut gated_spans_snapshot = mem::take(&mut *sess.gated_spans.spans.borrow_mut());

        match parse_tt(&mut Cow::Borrowed(&parser), lhs_tt) {
            Success(named_matches) => {
                // The matcher was `Success(..)`ful.
                // Merge the gated spans from parsing the matcher with the pre-existing ones.
                sess.gated_spans.merge(gated_spans_snapshot);

                let rhs = match rhses[i] {
                    // ignore delimiters
                    mbe::TokenTree::Delimited(_, ref delimed) => delimed.tts.clone(),
                    _ => cx.span_bug(sp, "malformed macro rhs"),
                };
                let arm_span = rhses[i].span();

                let rhs_spans = rhs.iter().map(|t| t.span()).collect::<Vec<_>>();
                // rhs has holes ( `$id` and `$(...)` that need filled)
                let mut tts = match transcribe(cx, &named_matches, rhs, transparency) {
                    Ok(tts) => tts,
                    Err(mut err) => {
                        err.emit();
                        return DummyResult::any(arm_span);
                    }
                };

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
                    let msg = format!("to `{}`", pprust::tts_to_string(&tts));
                    trace_macros_note(&mut cx.expansions, sp, msg);
                }

                let mut p = Parser::new(sess, tts, false, None);
                p.last_type_ascription = cx.current_expansion.prior_type_ascription;

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
            Error(err_sp, ref msg) => {
                let span = err_sp.substitute_dummy(sp);
                cx.struct_span_err(span, &msg).emit();
                return DummyResult::any(span);
            }
            ErrorReported => return DummyResult::any(sp),
        }

        // The matcher was not `Success(..)`ful.
        // Restore to the state before snapshotting and maybe try again.
        mem::swap(&mut gated_spans_snapshot, &mut sess.gated_spans.spans.borrow_mut());
    }
    drop(parser);

    let (token, label) = best_failure.expect("ran no matchers");
    let span = token.span.substitute_dummy(sp);
    let mut err = cx.struct_span_err(span, &parse_failure_msg(&token));
    err.span_label(span, label);
    if !def_span.is_dummy() && !cx.source_map().is_imported(def_span) {
        err.span_label(cx.source_map().guess_head_span(def_span), "when calling this macro");
    }

    // Check whether there's a missing comma in this macro call, like `println!("{}" a);`
    if let Some((arg, comma_span)) = arg.add_comma() {
        for lhs in lhses {
            // try each arm's matchers
            let lhs_tt = match *lhs {
                mbe::TokenTree::Delimited(_, ref delim) => &delim.tts[..],
                _ => continue,
            };
            if let Success(_) =
                parse_tt(&mut Cow::Borrowed(&parser_from_cx(sess, arg.clone())), lhs_tt)
            {
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
    sess: &Session,
    features: &Features,
    def: &ast::Item,
    edition: Edition,
) -> SyntaxExtension {
    debug!("compile_declarative_macro: {:?}", def);
    let mk_syn_ext = |expander| {
        SyntaxExtension::new(
            sess,
            SyntaxExtensionKind::LegacyBang(expander),
            def.span,
            Vec::new(),
            edition,
            def.ident.name,
            &def.attrs,
        )
    };

    let diag = &sess.parse_sess.span_diagnostic;
    let lhs_nm = Ident::new(sym::lhs, def.span);
    let rhs_nm = Ident::new(sym::rhs, def.span);
    let tt_spec = Some(NonterminalKind::TT);

    // Parse the macro_rules! invocation
    let (macro_rules, body) = match &def.kind {
        ast::ItemKind::MacroDef(def) => (def.macro_rules, def.body.inner_tokens()),
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
                    if macro_rules { token::Semi } else { token::Comma },
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
                    if macro_rules { token::Semi } else { token::Comma },
                    def.span,
                )],
                separator: None,
                kleene: mbe::KleeneToken::new(mbe::KleeneOp::ZeroOrMore, def.span),
                num_captures: 0,
            }),
        ),
    ];

    let parser = Parser::new(&sess.parse_sess, body, true, rustc_parse::MACRO_ARGUMENTS);
    let argument_map = match parse_tt(&mut Cow::Borrowed(&parser), &argument_gram) {
        Success(m) => m,
        Failure(token, msg) => {
            let s = parse_failure_msg(&token);
            let sp = token.span.substitute_dummy(def.span);
            sess.parse_sess.span_diagnostic.struct_span_err(sp, &s).span_label(sp, msg).emit();
            return mk_syn_ext(Box::new(macro_rules_dummy_expander));
        }
        Error(sp, msg) => {
            sess.parse_sess
                .span_diagnostic
                .struct_span_err(sp.substitute_dummy(def.span), &msg)
                .emit();
            return mk_syn_ext(Box::new(macro_rules_dummy_expander));
        }
        ErrorReported => {
            return mk_syn_ext(Box::new(macro_rules_dummy_expander));
        }
    };

    let mut valid = true;

    // Extract the arguments:
    let lhses = match argument_map[&MacroRulesNormalizedIdent::new(lhs_nm)] {
        MatchedSeq(ref s) => s
            .iter()
            .map(|m| {
                if let MatchedNonterminal(ref nt) = *m {
                    if let NtTT(ref tt) = **nt {
                        let tt = mbe::quoted::parse(
                            tt.clone().into(),
                            true,
                            &sess.parse_sess,
                            def.id,
                            features,
                        )
                        .pop()
                        .unwrap();
                        valid &= check_lhs_nt_follows(&sess.parse_sess, features, &def.attrs, &tt);
                        return tt;
                    }
                }
                sess.parse_sess.span_diagnostic.span_bug(def.span, "wrong-structured lhs")
            })
            .collect::<Vec<mbe::TokenTree>>(),
        _ => sess.parse_sess.span_diagnostic.span_bug(def.span, "wrong-structured lhs"),
    };

    let rhses = match argument_map[&MacroRulesNormalizedIdent::new(rhs_nm)] {
        MatchedSeq(ref s) => s
            .iter()
            .map(|m| {
                if let MatchedNonterminal(ref nt) = *m {
                    if let NtTT(ref tt) = **nt {
                        return mbe::quoted::parse(
                            tt.clone().into(),
                            false,
                            &sess.parse_sess,
                            def.id,
                            features,
                        )
                        .pop()
                        .unwrap();
                    }
                }
                sess.parse_sess.span_diagnostic.span_bug(def.span, "wrong-structured lhs")
            })
            .collect::<Vec<mbe::TokenTree>>(),
        _ => sess.parse_sess.span_diagnostic.span_bug(def.span, "wrong-structured rhs"),
    };

    for rhs in &rhses {
        valid &= check_rhs(&sess.parse_sess, rhs);
    }

    // don't abort iteration early, so that errors for multiple lhses can be reported
    for lhs in &lhses {
        valid &= check_lhs_no_empty_seq(&sess.parse_sess, slice::from_ref(lhs));
    }

    valid &= macro_check::check_meta_variables(&sess.parse_sess, def.id, def.span, &lhses, &rhses);

    let (transparency, transparency_error) = attr::find_transparency(sess, &def.attrs, macro_rules);
    match transparency_error {
        Some(TransparencyError::UnknownTransparency(value, span)) => {
            diag.span_err(span, &format!("unknown macro transparency: `{}`", value))
        }
        Some(TransparencyError::MultipleTransparencyAttrs(old_span, new_span)) => {
            diag.span_err(vec![old_span, new_span], "multiple macro transparency attributes")
        }
        None => {}
    }

    mk_syn_ext(Box::new(MacroRulesMacroExpander {
        name: def.ident,
        span: def.span,
        transparency,
        lhses,
        rhses,
        valid,
    }))
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
                        TokenTree::MetaVarDecl(_, _, Some(NonterminalKind::Vis)) => true,
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
                if token_can_be_followed_by_any(token) {
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
        for token in &last.tokens {
            if let TokenTree::MetaVarDecl(_, name, Some(kind)) = *token {
                for next_token in &suffix_first.tokens {
                    match is_in_follow(next_token, kind) {
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
                                    frag = kind,
                                    next = quoted_tt_to_string(next_token),
                                    may_be = may_be
                                ),
                            );
                            err.span_label(sp, format!("not allowed after `{}` fragments", kind));
                            let msg = "allowed there are: ";
                            match possible {
                                &[] => {}
                                &[t] => {
                                    err.note(&format!(
                                        "only {} is allowed after `{}` fragments",
                                        t, kind,
                                    ));
                                }
                                ts => {
                                    err.note(&format!(
                                        "{}{} or {}",
                                        msg,
                                        ts[..ts.len() - 1]
                                            .iter()
                                            .copied()
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
    if let mbe::TokenTree::MetaVarDecl(_, _, Some(kind)) = *tok {
        frag_can_be_followed_by_any(kind)
    } else {
        // (Non NT's can always be followed by anything in matchers.)
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
fn frag_can_be_followed_by_any(kind: NonterminalKind) -> bool {
    matches!(
        kind,
        NonterminalKind::Item           // always terminated by `}` or `;`
        | NonterminalKind::Block        // exactly one token tree
        | NonterminalKind::Ident        // exactly one token tree
        | NonterminalKind::Literal      // exactly one token tree
        | NonterminalKind::Meta         // exactly one token tree
        | NonterminalKind::Lifetime     // exactly one token tree
        | NonterminalKind::TT // exactly one token tree
    )
}

enum IsInFollow {
    Yes,
    No(&'static [&'static str]),
}

/// Returns `true` if `frag` can legally be followed by the token `tok`. For
/// fragments that can consume an unbounded number of tokens, `tok`
/// must be within a well-defined follow set. This is intended to
/// guarantee future compatibility: for example, without this rule, if
/// we expanded `expr` to include a new binary operator, we might
/// break macros that were relying on that binary operator as a
/// separator.
// when changing this do not forget to update doc/book/macros.md!
fn is_in_follow(tok: &mbe::TokenTree, kind: NonterminalKind) -> IsInFollow {
    use mbe::TokenTree;

    if let TokenTree::Token(Token { kind: token::CloseDelim(_), .. }) = *tok {
        // closing a token tree can never be matched by any fragment;
        // iow, we always require that `(` and `)` match, etc.
        IsInFollow::Yes
    } else {
        match kind {
            NonterminalKind::Item => {
                // since items *must* be followed by either a `;` or a `}`, we can
                // accept anything after them
                IsInFollow::Yes
            }
            NonterminalKind::Block => {
                // anything can follow block, the braces provide an easy boundary to
                // maintain
                IsInFollow::Yes
            }
            NonterminalKind::Stmt | NonterminalKind::Expr => {
                const TOKENS: &[&str] = &["`=>`", "`,`", "`;`"];
                match tok {
                    TokenTree::Token(token) => match token.kind {
                        FatArrow | Comma | Semi => IsInFollow::Yes,
                        _ => IsInFollow::No(TOKENS),
                    },
                    _ => IsInFollow::No(TOKENS),
                }
            }
            NonterminalKind::Pat2018 { .. } | NonterminalKind::Pat2021 { .. } => {
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
            NonterminalKind::Path | NonterminalKind::Ty => {
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
                    TokenTree::MetaVarDecl(_, _, Some(NonterminalKind::Block)) => IsInFollow::Yes,
                    _ => IsInFollow::No(TOKENS),
                }
            }
            NonterminalKind::Ident | NonterminalKind::Lifetime => {
                // being a single token, idents and lifetimes are harmless
                IsInFollow::Yes
            }
            NonterminalKind::Literal => {
                // literals may be of a single token, or two tokens (negative numbers)
                IsInFollow::Yes
            }
            NonterminalKind::Meta | NonterminalKind::TT => {
                // being either a single token or a delimited sequence, tt is
                // harmless
                IsInFollow::Yes
            }
            NonterminalKind::Vis => {
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
                    TokenTree::MetaVarDecl(
                        _,
                        _,
                        Some(NonterminalKind::Ident | NonterminalKind::Ty | NonterminalKind::Path),
                    ) => IsInFollow::Yes,
                    _ => IsInFollow::No(TOKENS),
                }
            }
        }
    }
}

fn quoted_tt_to_string(tt: &mbe::TokenTree) -> String {
    match *tt {
        mbe::TokenTree::Token(ref token) => pprust::token_to_string(&token),
        mbe::TokenTree::MetaVar(_, name) => format!("${}", name),
        mbe::TokenTree::MetaVarDecl(_, name, Some(kind)) => format!("${}:{}", name, kind),
        mbe::TokenTree::MetaVarDecl(_, name, None) => format!("${}:", name),
        _ => panic!(
            "{}",
            "unexpected mbe::TokenTree::{Sequence or Delimited} \
             in follow set checker"
        ),
    }
}

fn parser_from_cx(sess: &ParseSess, tts: TokenStream) -> Parser<'_> {
    Parser::new(sess, tts, true, rustc_parse::MACRO_ARGUMENTS)
}

/// Generates an appropriate parsing failure message. For EOF, this is "unexpected end...". For
/// other tokens, this is "unexpected token...".
fn parse_failure_msg(tok: &Token) -> String {
    match tok.kind {
        token::Eof => "unexpected end of macro invocation".to_string(),
        _ => format!("no rules expected the token `{}`", pprust::token_to_string(tok),),
    }
}
