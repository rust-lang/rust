use std::borrow::Cow;
use std::collections::hash_map::Entry;
use std::sync::Arc;
use std::{mem, slice};

use ast::token::IdentIsRaw;
use rustc_ast::token::NtPatKind::*;
use rustc_ast::token::TokenKind::*;
use rustc_ast::token::{self, Delimiter, NonterminalKind, Token, TokenKind};
use rustc_ast::tokenstream::{DelimSpan, TokenStream};
use rustc_ast::{self as ast, DUMMY_NODE_ID, NodeId};
use rustc_ast_pretty::pprust;
use rustc_attr_parsing::{AttributeKind, find_attr};
use rustc_data_structures::fx::{FxHashMap, FxIndexMap};
use rustc_errors::{Applicability, Diag, ErrorGuaranteed};
use rustc_feature::Features;
use rustc_hir as hir;
use rustc_lint_defs::BuiltinLintDiag;
use rustc_lint_defs::builtin::{
    RUST_2021_INCOMPATIBLE_OR_PATTERNS, SEMICOLON_IN_EXPRESSIONS_FROM_MACROS,
};
use rustc_parse::parser::{ParseNtResult, Parser, Recovery};
use rustc_session::Session;
use rustc_session::parse::ParseSess;
use rustc_span::edition::Edition;
use rustc_span::hygiene::Transparency;
use rustc_span::{Ident, MacroRulesNormalizedIdent, Span, kw, sym};
use tracing::{debug, instrument, trace, trace_span};

use super::macro_parser::{NamedMatches, NamedParseResult};
use super::{SequenceRepetition, diagnostics};
use crate::base::{
    DummyResult, ExpandResult, ExtCtxt, MacResult, MacroExpanderResult, SyntaxExtension,
    SyntaxExtensionKind, TTMacroExpander,
};
use crate::expand::{AstFragment, AstFragmentKind, ensure_complete_parse, parse_ast_fragment};
use crate::mbe::diagnostics::{annotate_doc_comment, parse_failure_msg};
use crate::mbe::macro_parser::NamedMatch::*;
use crate::mbe::macro_parser::{Error, ErrorReported, Failure, MatcherLoc, Success, TtParser};
use crate::mbe::transcribe::transcribe;
use crate::mbe::{self, KleeneOp, macro_check};

pub(crate) struct ParserAnyMacro<'a> {
    parser: Parser<'a>,

    /// Span of the expansion site of the macro this parser is for
    site_span: Span,
    /// The ident of the macro we're parsing
    macro_ident: Ident,
    lint_node_id: NodeId,
    is_trailing_mac: bool,
    arm_span: Span,
    /// Whether or not this macro is defined in the current crate
    is_local: bool,
}

impl<'a> ParserAnyMacro<'a> {
    pub(crate) fn make(mut self: Box<ParserAnyMacro<'a>>, kind: AstFragmentKind) -> AstFragment {
        let ParserAnyMacro {
            site_span,
            macro_ident,
            ref mut parser,
            lint_node_id,
            arm_span,
            is_trailing_mac,
            is_local,
        } = *self;
        let snapshot = &mut parser.create_snapshot_for_diagnostic();
        let fragment = match parse_ast_fragment(parser, kind) {
            Ok(f) => f,
            Err(err) => {
                let guar = diagnostics::emit_frag_parse_err(
                    err, parser, snapshot, site_span, arm_span, kind,
                );
                return kind.dummy(site_span, guar);
            }
        };

        // We allow semicolons at the end of expressions -- e.g., the semicolon in
        // `macro_rules! m { () => { panic!(); } }` isn't parsed by `.parse_expr()`,
        // but `m!()` is allowed in expression positions (cf. issue #34706).
        if kind == AstFragmentKind::Expr && parser.token == token::Semi {
            if is_local {
                parser.psess.buffer_lint(
                    SEMICOLON_IN_EXPRESSIONS_FROM_MACROS,
                    parser.token.span,
                    lint_node_id,
                    BuiltinLintDiag::TrailingMacro(is_trailing_mac, macro_ident),
                );
            }
            parser.bump();
        }

        // Make sure we don't have any tokens left to parse so we don't silently drop anything.
        let path = ast::Path::from_ident(macro_ident.with_span_pos(site_span));
        ensure_complete_parse(parser, &path, kind.name(), site_span);
        fragment
    }
}

struct MacroRulesMacroExpander {
    node_id: NodeId,
    name: Ident,
    span: Span,
    transparency: Transparency,
    lhses: Vec<Vec<MatcherLoc>>,
    rhses: Vec<mbe::TokenTree>,
}

impl TTMacroExpander for MacroRulesMacroExpander {
    fn expand<'cx>(
        &self,
        cx: &'cx mut ExtCtxt<'_>,
        sp: Span,
        input: TokenStream,
    ) -> MacroExpanderResult<'cx> {
        ExpandResult::Ready(expand_macro(
            cx,
            sp,
            self.span,
            self.node_id,
            self.name,
            self.transparency,
            input,
            &self.lhses,
            &self.rhses,
        ))
    }
}

struct DummyExpander(ErrorGuaranteed);

impl TTMacroExpander for DummyExpander {
    fn expand<'cx>(
        &self,
        _: &'cx mut ExtCtxt<'_>,
        span: Span,
        _: TokenStream,
    ) -> ExpandResult<Box<dyn MacResult + 'cx>, ()> {
        ExpandResult::Ready(DummyResult::any(span, self.0))
    }
}

fn trace_macros_note(cx_expansions: &mut FxIndexMap<Span, Vec<String>>, sp: Span, message: String) {
    let sp = sp.macro_backtrace().last().map_or(sp, |trace| trace.call_site);
    cx_expansions.entry(sp).or_default().push(message);
}

pub(super) trait Tracker<'matcher> {
    /// The contents of `ParseResult::Failure`.
    type Failure;

    /// Arm failed to match. If the token is `token::Eof`, it indicates an unexpected
    /// end of macro invocation. Otherwise, it indicates that no rules expected the given token.
    /// The usize is the approximate position of the token in the input token stream.
    fn build_failure(tok: Token, position: u32, msg: &'static str) -> Self::Failure;

    /// This is called before trying to match next MatcherLoc on the current token.
    fn before_match_loc(&mut self, _parser: &TtParser, _matcher: &'matcher MatcherLoc) {}

    /// This is called after an arm has been parsed, either successfully or unsuccessfully. When
    /// this is called, `before_match_loc` was called at least once (with a `MatcherLoc::Eof`).
    fn after_arm(&mut self, _result: &NamedParseResult<Self::Failure>) {}

    /// For tracing.
    fn description() -> &'static str;

    fn recovery() -> Recovery {
        Recovery::Forbidden
    }

    fn set_expected_token(&mut self, _tok: &'matcher Token) {}
    fn get_expected_token(&self) -> Option<&'matcher Token> {
        None
    }
}

/// A noop tracker that is used in the hot path of the expansion, has zero overhead thanks to
/// monomorphization.
pub(super) struct NoopTracker;

impl<'matcher> Tracker<'matcher> for NoopTracker {
    type Failure = ();

    fn build_failure(_tok: Token, _position: u32, _msg: &'static str) -> Self::Failure {}

    fn description() -> &'static str {
        "none"
    }
}

/// Expands the rules based macro defined by `lhses` and `rhses` for a given
/// input `arg`.
#[instrument(skip(cx, transparency, arg, lhses, rhses))]
fn expand_macro<'cx>(
    cx: &'cx mut ExtCtxt<'_>,
    sp: Span,
    def_span: Span,
    node_id: NodeId,
    name: Ident,
    transparency: Transparency,
    arg: TokenStream,
    lhses: &[Vec<MatcherLoc>],
    rhses: &[mbe::TokenTree],
) -> Box<dyn MacResult + 'cx> {
    let psess = &cx.sess.psess;
    // Macros defined in the current crate have a real node id,
    // whereas macros from an external crate have a dummy id.
    let is_local = node_id != DUMMY_NODE_ID;

    if cx.trace_macros() {
        let msg = format!("expanding `{}! {{ {} }}`", name, pprust::tts_to_string(&arg));
        trace_macros_note(&mut cx.expansions, sp, msg);
    }

    // Track nothing for the best performance.
    let try_success_result = try_match_macro(psess, name, &arg, lhses, &mut NoopTracker);

    match try_success_result {
        Ok((i, named_matches)) => {
            let (rhs, rhs_span): (&mbe::Delimited, DelimSpan) = match &rhses[i] {
                mbe::TokenTree::Delimited(span, _, delimited) => (&delimited, *span),
                _ => cx.dcx().span_bug(sp, "malformed macro rhs"),
            };
            let arm_span = rhses[i].span();

            // rhs has holes ( `$id` and `$(...)` that need filled)
            let id = cx.current_expansion.id;
            let tts = match transcribe(psess, &named_matches, rhs, rhs_span, transparency, id) {
                Ok(tts) => tts,
                Err(err) => {
                    let guar = err.emit();
                    return DummyResult::any(arm_span, guar);
                }
            };

            if cx.trace_macros() {
                let msg = format!("to `{}`", pprust::tts_to_string(&tts));
                trace_macros_note(&mut cx.expansions, sp, msg);
            }

            let p = Parser::new(psess, tts, None);

            if is_local {
                cx.resolver.record_macro_rule_usage(node_id, i);
            }

            // Let the context choose how to interpret the result.
            // Weird, but useful for X-macros.
            Box::new(ParserAnyMacro {
                parser: p,

                // Pass along the original expansion site and the name of the macro
                // so we can print a useful error message if the parse of the expanded
                // macro leaves unparsed tokens.
                site_span: sp,
                macro_ident: name,
                lint_node_id: cx.current_expansion.lint_node_id,
                is_trailing_mac: cx.current_expansion.is_trailing_mac,
                arm_span,
                is_local,
            })
        }
        Err(CanRetry::No(guar)) => {
            debug!("Will not retry matching as an error was emitted already");
            DummyResult::any(sp, guar)
        }
        Err(CanRetry::Yes) => {
            // Retry and emit a better error.
            let (span, guar) =
                diagnostics::failed_to_match_macro(cx.psess(), sp, def_span, name, arg, lhses);
            cx.trace_macros_diag();
            DummyResult::any(span, guar)
        }
    }
}

pub(super) enum CanRetry {
    Yes,
    /// We are not allowed to retry macro expansion as a fatal error has been emitted already.
    No(ErrorGuaranteed),
}

/// Try expanding the macro. Returns the index of the successful arm and its named_matches if it was successful,
/// and nothing if it failed. On failure, it's the callers job to use `track` accordingly to record all errors
/// correctly.
#[instrument(level = "debug", skip(psess, arg, lhses, track), fields(tracking = %T::description()))]
pub(super) fn try_match_macro<'matcher, T: Tracker<'matcher>>(
    psess: &ParseSess,
    name: Ident,
    arg: &TokenStream,
    lhses: &'matcher [Vec<MatcherLoc>],
    track: &mut T,
) -> Result<(usize, NamedMatches), CanRetry> {
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
    let parser = parser_from_cx(psess, arg.clone(), T::recovery());
    // Try each arm's matchers.
    let mut tt_parser = TtParser::new(name);
    for (i, lhs) in lhses.iter().enumerate() {
        let _tracing_span = trace_span!("Matching arm", %i);

        // Take a snapshot of the state of pre-expansion gating at this point.
        // This is used so that if a matcher is not `Success(..)`ful,
        // then the spans which became gated when parsing the unsuccessful matcher
        // are not recorded. On the first `Success(..)`ful matcher, the spans are merged.
        let mut gated_spans_snapshot = mem::take(&mut *psess.gated_spans.spans.borrow_mut());

        let result = tt_parser.parse_tt(&mut Cow::Borrowed(&parser), lhs, track);

        track.after_arm(&result);

        match result {
            Success(named_matches) => {
                debug!("Parsed arm successfully");
                // The matcher was `Success(..)`ful.
                // Merge the gated spans from parsing the matcher with the preexisting ones.
                psess.gated_spans.merge(gated_spans_snapshot);

                return Ok((i, named_matches));
            }
            Failure(_) => {
                trace!("Failed to match arm, trying the next one");
                // Try the next arm.
            }
            Error(_, _) => {
                debug!("Fatal error occurred during matching");
                // We haven't emitted an error yet, so we can retry.
                return Err(CanRetry::Yes);
            }
            ErrorReported(guarantee) => {
                debug!("Fatal error occurred and was reported during matching");
                // An error has been reported already, we cannot retry as that would cause duplicate errors.
                return Err(CanRetry::No(guarantee));
            }
        }

        // The matcher was not `Success(..)`ful.
        // Restore to the state before snapshotting and maybe try again.
        mem::swap(&mut gated_spans_snapshot, &mut psess.gated_spans.spans.borrow_mut());
    }

    Err(CanRetry::Yes)
}

// Note that macro-by-example's input is also matched against a token tree:
//                   $( $lhs:tt => $rhs:tt );+
//
// Holy self-referential!

/// Converts a macro item into a syntax extension.
pub fn compile_declarative_macro(
    sess: &Session,
    features: &Features,
    macro_def: &ast::MacroDef,
    ident: Ident,
    attrs: &[hir::Attribute],
    span: Span,
    node_id: NodeId,
    edition: Edition,
) -> (SyntaxExtension, Vec<(usize, Span)>) {
    let mk_syn_ext = |expander| {
        SyntaxExtension::new(
            sess,
            SyntaxExtensionKind::LegacyBang(expander),
            span,
            Vec::new(),
            edition,
            ident.name,
            attrs,
            node_id != DUMMY_NODE_ID,
        )
    };
    let dummy_syn_ext = |guar| (mk_syn_ext(Arc::new(DummyExpander(guar))), Vec::new());

    let lhs_nm = Ident::new(sym::lhs, span);
    let rhs_nm = Ident::new(sym::rhs, span);
    let tt_spec = Some(NonterminalKind::TT);
    let macro_rules = macro_def.macro_rules;

    // Parse the macro_rules! invocation

    // The pattern that macro_rules matches.
    // The grammar for macro_rules! is:
    // $( $lhs:tt => $rhs:tt );+
    // ...quasiquoting this would be nice.
    // These spans won't matter, anyways
    let argument_gram = vec![
        mbe::TokenTree::Sequence(
            DelimSpan::dummy(),
            mbe::SequenceRepetition {
                tts: vec![
                    mbe::TokenTree::MetaVarDecl(span, lhs_nm, tt_spec),
                    mbe::TokenTree::token(token::FatArrow, span),
                    mbe::TokenTree::MetaVarDecl(span, rhs_nm, tt_spec),
                ],
                separator: Some(Token::new(
                    if macro_rules { token::Semi } else { token::Comma },
                    span,
                )),
                kleene: mbe::KleeneToken::new(mbe::KleeneOp::OneOrMore, span),
                num_captures: 2,
            },
        ),
        // to phase into semicolon-termination instead of semicolon-separation
        mbe::TokenTree::Sequence(
            DelimSpan::dummy(),
            mbe::SequenceRepetition {
                tts: vec![mbe::TokenTree::token(
                    if macro_rules { token::Semi } else { token::Comma },
                    span,
                )],
                separator: None,
                kleene: mbe::KleeneToken::new(mbe::KleeneOp::ZeroOrMore, span),
                num_captures: 0,
            },
        ),
    ];
    // Convert it into `MatcherLoc` form.
    let argument_gram = mbe::macro_parser::compute_locs(&argument_gram);

    let create_parser = || {
        let body = macro_def.body.tokens.clone();
        Parser::new(&sess.psess, body, rustc_parse::MACRO_ARGUMENTS)
    };

    let parser = create_parser();
    let mut tt_parser =
        TtParser::new(Ident::with_dummy_span(if macro_rules { kw::MacroRules } else { kw::Macro }));
    let argument_map =
        match tt_parser.parse_tt(&mut Cow::Owned(parser), &argument_gram, &mut NoopTracker) {
            Success(m) => m,
            Failure(()) => {
                // The fast `NoopTracker` doesn't have any info on failure, so we need to retry it
                // with another one that gives us the information we need.
                // For this we need to reclone the macro body as the previous parser consumed it.
                let retry_parser = create_parser();

                let mut track = diagnostics::FailureForwarder::new();
                let parse_result =
                    tt_parser.parse_tt(&mut Cow::Owned(retry_parser), &argument_gram, &mut track);
                let Failure((token, _, msg)) = parse_result else {
                    unreachable!("matcher returned something other than Failure after retry");
                };

                let s = parse_failure_msg(&token, track.get_expected_token());
                let sp = token.span.substitute_dummy(span);
                let mut err = sess.dcx().struct_span_err(sp, s);
                err.span_label(sp, msg);
                annotate_doc_comment(&mut err, sess.source_map(), sp);
                let guar = err.emit();
                return dummy_syn_ext(guar);
            }
            Error(sp, msg) => {
                let guar = sess.dcx().span_err(sp.substitute_dummy(span), msg);
                return dummy_syn_ext(guar);
            }
            ErrorReported(guar) => {
                return dummy_syn_ext(guar);
            }
        };

    let mut guar = None;
    let mut check_emission = |ret: Result<(), ErrorGuaranteed>| guar = guar.or(ret.err());

    // Extract the arguments:
    let lhses = match &argument_map[&MacroRulesNormalizedIdent::new(lhs_nm)] {
        MatchedSeq(s) => s
            .iter()
            .map(|m| {
                if let MatchedSingle(ParseNtResult::Tt(tt)) = m {
                    let tt = mbe::quoted::parse(
                        &TokenStream::new(vec![tt.clone()]),
                        true,
                        sess,
                        node_id,
                        features,
                        edition,
                    )
                    .pop()
                    .unwrap();
                    // We don't handle errors here, the driver will abort
                    // after parsing/expansion. We can report every error in every macro this way.
                    check_emission(check_lhs_nt_follows(sess, node_id, &tt));
                    return tt;
                }
                sess.dcx().span_bug(span, "wrong-structured lhs")
            })
            .collect::<Vec<mbe::TokenTree>>(),
        _ => sess.dcx().span_bug(span, "wrong-structured lhs"),
    };

    let rhses = match &argument_map[&MacroRulesNormalizedIdent::new(rhs_nm)] {
        MatchedSeq(s) => s
            .iter()
            .map(|m| {
                if let MatchedSingle(ParseNtResult::Tt(tt)) = m {
                    return mbe::quoted::parse(
                        &TokenStream::new(vec![tt.clone()]),
                        false,
                        sess,
                        node_id,
                        features,
                        edition,
                    )
                    .pop()
                    .unwrap();
                }
                sess.dcx().span_bug(span, "wrong-structured rhs")
            })
            .collect::<Vec<mbe::TokenTree>>(),
        _ => sess.dcx().span_bug(span, "wrong-structured rhs"),
    };

    for rhs in &rhses {
        check_emission(check_rhs(sess, rhs));
    }

    // Don't abort iteration early, so that errors for multiple lhses can be reported.
    for lhs in &lhses {
        check_emission(check_lhs_no_empty_seq(sess, slice::from_ref(lhs)));
    }

    check_emission(macro_check::check_meta_variables(&sess.psess, node_id, span, &lhses, &rhses));

    let transparency = find_attr!(attrs, AttributeKind::MacroTransparency(x) => *x)
        .unwrap_or(Transparency::fallback(macro_rules));

    if let Some(guar) = guar {
        // To avoid warning noise, only consider the rules of this
        // macro for the lint, if all rules are valid.
        return dummy_syn_ext(guar);
    }

    // Compute the spans of the macro rules for unused rule linting.
    // Also, we are only interested in non-foreign macros.
    let rule_spans = if node_id != DUMMY_NODE_ID {
        lhses
            .iter()
            .zip(rhses.iter())
            .enumerate()
            // If the rhs contains an invocation like compile_error!,
            // don't consider the rule for the unused rule lint.
            .filter(|(_idx, (_lhs, rhs))| !has_compile_error_macro(rhs))
            // We only take the span of the lhs here,
            // so that the spans of created warnings are smaller.
            .map(|(idx, (lhs, _rhs))| (idx, lhs.span()))
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };

    // Convert the lhses into `MatcherLoc` form, which is better for doing the
    // actual matching.
    let lhses = lhses
        .iter()
        .map(|lhs| {
            // Ignore the delimiters around the matcher.
            match lhs {
                mbe::TokenTree::Delimited(.., delimited) => {
                    mbe::macro_parser::compute_locs(&delimited.tts)
                }
                _ => sess.dcx().span_bug(span, "malformed macro lhs"),
            }
        })
        .collect();

    let expander = Arc::new(MacroRulesMacroExpander {
        name: ident,
        span,
        node_id,
        transparency,
        lhses,
        rhses,
    });
    (mk_syn_ext(expander), rule_spans)
}

fn check_lhs_nt_follows(
    sess: &Session,
    node_id: NodeId,
    lhs: &mbe::TokenTree,
) -> Result<(), ErrorGuaranteed> {
    // lhs is going to be like TokenTree::Delimited(...), where the
    // entire lhs is those tts. Or, it can be a "bare sequence", not wrapped in parens.
    if let mbe::TokenTree::Delimited(.., delimited) = lhs {
        check_matcher(sess, node_id, &delimited.tts)
    } else {
        let msg = "invalid macro matcher; matchers must be contained in balanced delimiters";
        Err(sess.dcx().span_err(lhs.span(), msg))
    }
}

fn is_empty_token_tree(sess: &Session, seq: &mbe::SequenceRepetition) -> bool {
    if seq.separator.is_some() {
        false
    } else {
        let mut is_empty = true;
        let mut iter = seq.tts.iter().peekable();
        while let Some(tt) = iter.next() {
            match tt {
                mbe::TokenTree::MetaVarDecl(_, _, Some(NonterminalKind::Vis)) => {}
                mbe::TokenTree::Token(t @ Token { kind: DocComment(..), .. }) => {
                    let mut now = t;
                    while let Some(&mbe::TokenTree::Token(
                        next @ Token { kind: DocComment(..), .. },
                    )) = iter.peek()
                    {
                        now = next;
                        iter.next();
                    }
                    let span = t.span.to(now.span);
                    sess.dcx().span_note(span, "doc comments are ignored in matcher position");
                }
                mbe::TokenTree::Sequence(_, sub_seq)
                    if (sub_seq.kleene.op == mbe::KleeneOp::ZeroOrMore
                        || sub_seq.kleene.op == mbe::KleeneOp::ZeroOrOne) => {}
                _ => is_empty = false,
            }
        }
        is_empty
    }
}

/// Checks if a `vis` nonterminal fragment is unnecessarily wrapped in an optional repetition.
///
/// When a `vis` fragment (which can already be empty) is wrapped in `$(...)?`,
/// this suggests removing the redundant repetition syntax since it provides no additional benefit.
fn check_redundant_vis_repetition(
    err: &mut Diag<'_>,
    sess: &Session,
    seq: &SequenceRepetition,
    span: &DelimSpan,
) {
    let is_zero_or_one: bool = seq.kleene.op == KleeneOp::ZeroOrOne;
    let is_vis = seq.tts.first().map_or(false, |tt| {
        matches!(tt, mbe::TokenTree::MetaVarDecl(_, _, Some(NonterminalKind::Vis)))
    });

    if is_vis && is_zero_or_one {
        err.note("a `vis` fragment can already be empty");
        err.multipart_suggestion(
            "remove the `$(` and `)?`",
            vec![
                (
                    sess.source_map().span_extend_to_prev_char_before(span.open, '$', true),
                    "".to_string(),
                ),
                (span.close.with_hi(seq.kleene.span.hi()), "".to_string()),
            ],
            Applicability::MaybeIncorrect,
        );
    }
}

/// Checks that the lhs contains no repetition which could match an empty token
/// tree, because then the matcher would hang indefinitely.
fn check_lhs_no_empty_seq(sess: &Session, tts: &[mbe::TokenTree]) -> Result<(), ErrorGuaranteed> {
    use mbe::TokenTree;
    for tt in tts {
        match tt {
            TokenTree::Token(..)
            | TokenTree::MetaVar(..)
            | TokenTree::MetaVarDecl(..)
            | TokenTree::MetaVarExpr(..) => (),
            TokenTree::Delimited(.., del) => check_lhs_no_empty_seq(sess, &del.tts)?,
            TokenTree::Sequence(span, seq) => {
                if is_empty_token_tree(sess, seq) {
                    let sp = span.entire();
                    let mut err =
                        sess.dcx().struct_span_err(sp, "repetition matches empty token tree");
                    check_redundant_vis_repetition(&mut err, sess, seq, span);
                    return Err(err.emit());
                }
                check_lhs_no_empty_seq(sess, &seq.tts)?
            }
        }
    }

    Ok(())
}

fn check_rhs(sess: &Session, rhs: &mbe::TokenTree) -> Result<(), ErrorGuaranteed> {
    match *rhs {
        mbe::TokenTree::Delimited(..) => Ok(()),
        _ => Err(sess.dcx().span_err(rhs.span(), "macro rhs must be delimited")),
    }
}

fn check_matcher(
    sess: &Session,
    node_id: NodeId,
    matcher: &[mbe::TokenTree],
) -> Result<(), ErrorGuaranteed> {
    let first_sets = FirstSets::new(matcher);
    let empty_suffix = TokenSet::empty();
    check_matcher_core(sess, node_id, &first_sets, matcher, &empty_suffix)?;
    Ok(())
}

fn has_compile_error_macro(rhs: &mbe::TokenTree) -> bool {
    match rhs {
        mbe::TokenTree::Delimited(.., d) => {
            let has_compile_error = d.tts.array_windows::<3>().any(|[ident, bang, args]| {
                if let mbe::TokenTree::Token(ident) = ident
                    && let TokenKind::Ident(ident, _) = ident.kind
                    && ident == sym::compile_error
                    && let mbe::TokenTree::Token(bang) = bang
                    && let TokenKind::Bang = bang.kind
                    && let mbe::TokenTree::Delimited(.., del) = args
                    && !del.delim.skip()
                {
                    true
                } else {
                    false
                }
            });
            if has_compile_error { true } else { d.tts.iter().any(has_compile_error_macro) }
        }
        _ => false,
    }
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
struct FirstSets<'tt> {
    // this maps each TokenTree::Sequence `$(tt ...) SEP OP` that is uniquely identified by its
    // span in the original matcher to the First set for the inner sequence `tt ...`.
    //
    // If two sequences have the same span in a matcher, then map that
    // span to None (invalidating the mapping here and forcing the code to
    // use a slow path).
    first: FxHashMap<Span, Option<TokenSet<'tt>>>,
}

impl<'tt> FirstSets<'tt> {
    fn new(tts: &'tt [mbe::TokenTree]) -> FirstSets<'tt> {
        use mbe::TokenTree;

        let mut sets = FirstSets { first: FxHashMap::default() };
        build_recur(&mut sets, tts);
        return sets;

        // walks backward over `tts`, returning the FIRST for `tts`
        // and updating `sets` at the same time for all sequence
        // substructure we find within `tts`.
        fn build_recur<'tt>(sets: &mut FirstSets<'tt>, tts: &'tt [TokenTree]) -> TokenSet<'tt> {
            let mut first = TokenSet::empty();
            for tt in tts.iter().rev() {
                match tt {
                    TokenTree::Token(..)
                    | TokenTree::MetaVar(..)
                    | TokenTree::MetaVarDecl(..)
                    | TokenTree::MetaVarExpr(..) => {
                        first.replace_with(TtHandle::TtRef(tt));
                    }
                    TokenTree::Delimited(span, _, delimited) => {
                        build_recur(sets, &delimited.tts);
                        first.replace_with(TtHandle::from_token_kind(
                            token::OpenDelim(delimited.delim),
                            span.open,
                        ));
                    }
                    TokenTree::Sequence(sp, seq_rep) => {
                        let subfirst = build_recur(sets, &seq_rep.tts);

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
                            first.add_one_maybe(TtHandle::from_token(*sep));
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
    fn first(&self, tts: &'tt [mbe::TokenTree]) -> TokenSet<'tt> {
        use mbe::TokenTree;

        let mut first = TokenSet::empty();
        for tt in tts.iter() {
            assert!(first.maybe_empty);
            match tt {
                TokenTree::Token(..)
                | TokenTree::MetaVar(..)
                | TokenTree::MetaVarDecl(..)
                | TokenTree::MetaVarExpr(..) => {
                    first.add_one(TtHandle::TtRef(tt));
                    return first;
                }
                TokenTree::Delimited(span, _, delimited) => {
                    first.add_one(TtHandle::from_token_kind(
                        token::OpenDelim(delimited.delim),
                        span.open,
                    ));
                    return first;
                }
                TokenTree::Sequence(sp, seq_rep) => {
                    let subfirst_owned;
                    let subfirst = match self.first.get(&sp.entire()) {
                        Some(Some(subfirst)) => subfirst,
                        Some(&None) => {
                            subfirst_owned = self.first(&seq_rep.tts);
                            &subfirst_owned
                        }
                        None => {
                            panic!("We missed a sequence during FirstSets construction");
                        }
                    };

                    // If the sequence contents can be empty, then the first
                    // token could be the separator token itself.
                    if let (Some(sep), true) = (&seq_rep.separator, subfirst.maybe_empty) {
                        first.add_one_maybe(TtHandle::from_token(*sep));
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

// Most `mbe::TokenTree`s are preexisting in the matcher, but some are defined
// implicitly, such as opening/closing delimiters and sequence repetition ops.
// This type encapsulates both kinds. It implements `Clone` while avoiding the
// need for `mbe::TokenTree` to implement `Clone`.
#[derive(Debug)]
enum TtHandle<'tt> {
    /// This is used in most cases.
    TtRef(&'tt mbe::TokenTree),

    /// This is only used for implicit token trees. The `mbe::TokenTree` *must*
    /// be `mbe::TokenTree::Token`. No other variants are allowed. We store an
    /// `mbe::TokenTree` rather than a `Token` so that `get()` can return a
    /// `&mbe::TokenTree`.
    Token(mbe::TokenTree),
}

impl<'tt> TtHandle<'tt> {
    fn from_token(tok: Token) -> Self {
        TtHandle::Token(mbe::TokenTree::Token(tok))
    }

    fn from_token_kind(kind: TokenKind, span: Span) -> Self {
        TtHandle::from_token(Token::new(kind, span))
    }

    // Get a reference to a token tree.
    fn get(&'tt self) -> &'tt mbe::TokenTree {
        match self {
            TtHandle::TtRef(tt) => tt,
            TtHandle::Token(token_tt) => token_tt,
        }
    }
}

impl<'tt> PartialEq for TtHandle<'tt> {
    fn eq(&self, other: &TtHandle<'tt>) -> bool {
        self.get() == other.get()
    }
}

impl<'tt> Clone for TtHandle<'tt> {
    fn clone(&self) -> Self {
        match self {
            TtHandle::TtRef(tt) => TtHandle::TtRef(tt),

            // This variant *must* contain a `mbe::TokenTree::Token`, and not
            // any other variant of `mbe::TokenTree`.
            TtHandle::Token(mbe::TokenTree::Token(tok)) => {
                TtHandle::Token(mbe::TokenTree::Token(*tok))
            }

            _ => unreachable!(),
        }
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
struct TokenSet<'tt> {
    tokens: Vec<TtHandle<'tt>>,
    maybe_empty: bool,
}

impl<'tt> TokenSet<'tt> {
    // Returns a set for the empty sequence.
    fn empty() -> Self {
        TokenSet { tokens: Vec::new(), maybe_empty: true }
    }

    // Returns the set `{ tok }` for the single-token (and thus
    // non-empty) sequence [tok].
    fn singleton(tt: TtHandle<'tt>) -> Self {
        TokenSet { tokens: vec![tt], maybe_empty: false }
    }

    // Changes self to be the set `{ tok }`.
    // Since `tok` is always present, marks self as non-empty.
    fn replace_with(&mut self, tt: TtHandle<'tt>) {
        self.tokens.clear();
        self.tokens.push(tt);
        self.maybe_empty = false;
    }

    // Changes self to be the empty set `{}`; meant for use when
    // the particular token does not matter, but we want to
    // record that it occurs.
    fn replace_with_irrelevant(&mut self) {
        self.tokens.clear();
        self.maybe_empty = false;
    }

    // Adds `tok` to the set for `self`, marking sequence as non-empty.
    fn add_one(&mut self, tt: TtHandle<'tt>) {
        if !self.tokens.contains(&tt) {
            self.tokens.push(tt);
        }
        self.maybe_empty = false;
    }

    // Adds `tok` to the set for `self`. (Leaves `maybe_empty` flag alone.)
    fn add_one_maybe(&mut self, tt: TtHandle<'tt>) {
        if !self.tokens.contains(&tt) {
            self.tokens.push(tt);
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
        for tt in &other.tokens {
            if !self.tokens.contains(tt) {
                self.tokens.push(tt.clone());
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
fn check_matcher_core<'tt>(
    sess: &Session,
    node_id: NodeId,
    first_sets: &FirstSets<'tt>,
    matcher: &'tt [mbe::TokenTree],
    follow: &TokenSet<'tt>,
) -> Result<TokenSet<'tt>, ErrorGuaranteed> {
    use mbe::TokenTree;

    let mut last = TokenSet::empty();

    let mut errored = Ok(());

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
        match token {
            TokenTree::Token(..)
            | TokenTree::MetaVar(..)
            | TokenTree::MetaVarDecl(..)
            | TokenTree::MetaVarExpr(..) => {
                if token_can_be_followed_by_any(token) {
                    // don't need to track tokens that work with any,
                    last.replace_with_irrelevant();
                    // ... and don't need to check tokens that can be
                    // followed by anything against SUFFIX.
                    continue 'each_token;
                } else {
                    last.replace_with(TtHandle::TtRef(token));
                    suffix_first = build_suffix_first();
                }
            }
            TokenTree::Delimited(span, _, d) => {
                let my_suffix = TokenSet::singleton(TtHandle::from_token_kind(
                    token::CloseDelim(d.delim),
                    span.close,
                ));
                check_matcher_core(sess, node_id, first_sets, &d.tts, &my_suffix)?;
                // don't track non NT tokens
                last.replace_with_irrelevant();

                // also, we don't need to check delimited sequences
                // against SUFFIX
                continue 'each_token;
            }
            TokenTree::Sequence(_, seq_rep) => {
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
                    new.add_one_maybe(TtHandle::from_token(*sep));
                    &new
                } else {
                    &suffix_first
                };

                // At this point, `suffix_first` is built, and
                // `my_suffix` is some TokenSet that we can use
                // for checking the interior of `seq_rep`.
                let next = check_matcher_core(sess, node_id, first_sets, &seq_rep.tts, my_suffix)?;
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
        for tt in &last.tokens {
            if let &TokenTree::MetaVarDecl(span, name, Some(kind)) = tt.get() {
                for next_token in &suffix_first.tokens {
                    let next_token = next_token.get();

                    // Check if the old pat is used and the next token is `|`
                    // to warn about incompatibility with Rust 2021.
                    // We only emit this lint if we're parsing the original
                    // definition of this macro_rules, not while (re)parsing
                    // the macro when compiling another crate that is using the
                    // macro. (See #86567.)
                    // Macros defined in the current crate have a real node id,
                    // whereas macros from an external crate have a dummy id.
                    if node_id != DUMMY_NODE_ID
                        && matches!(kind, NonterminalKind::Pat(PatParam { inferred: true }))
                        && matches!(
                            next_token,
                            TokenTree::Token(token) if *token == token::Or
                        )
                    {
                        // It is suggestion to use pat_param, for example: $x:pat -> $x:pat_param.
                        let suggestion = quoted_tt_to_string(&TokenTree::MetaVarDecl(
                            span,
                            name,
                            Some(NonterminalKind::Pat(PatParam { inferred: false })),
                        ));
                        sess.psess.buffer_lint(
                            RUST_2021_INCOMPATIBLE_OR_PATTERNS,
                            span,
                            ast::CRATE_NODE_ID,
                            BuiltinLintDiag::OrPatternsBackCompat(span, suggestion),
                        );
                    }
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
                            let mut err = sess.dcx().struct_span_err(
                                sp,
                                format!(
                                    "`${name}:{frag}` {may_be} followed by `{next}`, which \
                                     is not allowed for `{frag}` fragments",
                                    name = name,
                                    frag = kind,
                                    next = quoted_tt_to_string(next_token),
                                    may_be = may_be
                                ),
                            );
                            err.span_label(sp, format!("not allowed after `{kind}` fragments"));

                            if kind == NonterminalKind::Pat(PatWithOr)
                                && sess.psess.edition.at_least_rust_2021()
                                && next_token.is_token(&token::Or)
                            {
                                let suggestion = quoted_tt_to_string(&TokenTree::MetaVarDecl(
                                    span,
                                    name,
                                    Some(NonterminalKind::Pat(PatParam { inferred: false })),
                                ));
                                err.span_suggestion(
                                    span,
                                    "try a `pat_param` fragment specifier instead",
                                    suggestion,
                                    Applicability::MaybeIncorrect,
                                );
                            }

                            let msg = "allowed there are: ";
                            match possible {
                                &[] => {}
                                &[t] => {
                                    err.note(format!(
                                        "only {t} is allowed after `{kind}` fragments",
                                    ));
                                }
                                ts => {
                                    err.note(format!(
                                        "{}{} or {}",
                                        msg,
                                        ts[..ts.len() - 1].to_vec().join(", "),
                                        ts[ts.len() - 1],
                                    ));
                                }
                            }
                            errored = Err(err.emit());
                        }
                    }
                }
            }
        }
    }
    errored?;
    Ok(last)
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
            NonterminalKind::Stmt | NonterminalKind::Expr(_) => {
                const TOKENS: &[&str] = &["`=>`", "`,`", "`;`"];
                match tok {
                    TokenTree::Token(token) => match token.kind {
                        FatArrow | Comma | Semi => IsInFollow::Yes,
                        _ => IsInFollow::No(TOKENS),
                    },
                    _ => IsInFollow::No(TOKENS),
                }
            }
            NonterminalKind::Pat(PatParam { .. }) => {
                const TOKENS: &[&str] = &["`=>`", "`,`", "`=`", "`|`", "`if`", "`in`"];
                match tok {
                    TokenTree::Token(token) => match token.kind {
                        FatArrow | Comma | Eq | Or => IsInFollow::Yes,
                        Ident(name, IdentIsRaw::No) if name == kw::If || name == kw::In => {
                            IsInFollow::Yes
                        }
                        _ => IsInFollow::No(TOKENS),
                    },
                    _ => IsInFollow::No(TOKENS),
                }
            }
            NonterminalKind::Pat(PatWithOr) => {
                const TOKENS: &[&str] = &["`=>`", "`,`", "`=`", "`if`", "`in`"];
                match tok {
                    TokenTree::Token(token) => match token.kind {
                        FatArrow | Comma | Eq => IsInFollow::Yes,
                        Ident(name, IdentIsRaw::No) if name == kw::If || name == kw::In => {
                            IsInFollow::Yes
                        }
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
                        OpenDelim(Delimiter::Brace)
                        | OpenDelim(Delimiter::Bracket)
                        | Comma
                        | FatArrow
                        | Colon
                        | Eq
                        | Gt
                        | Shr
                        | Semi
                        | Or => IsInFollow::Yes,
                        Ident(name, IdentIsRaw::No) if name == kw::As || name == kw::Where => {
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
                        Ident(_, IdentIsRaw::Yes) => IsInFollow::Yes,
                        Ident(name, _) if name != kw::Priv => IsInFollow::Yes,
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
    match tt {
        mbe::TokenTree::Token(token) => pprust::token_to_string(token).into(),
        mbe::TokenTree::MetaVar(_, name) => format!("${name}"),
        mbe::TokenTree::MetaVarDecl(_, name, Some(kind)) => format!("${name}:{kind}"),
        mbe::TokenTree::MetaVarDecl(_, name, None) => format!("${name}:"),
        _ => panic!(
            "{}",
            "unexpected mbe::TokenTree::{Sequence or Delimited} \
             in follow set checker"
        ),
    }
}

pub(super) fn parser_from_cx(
    psess: &ParseSess,
    mut tts: TokenStream,
    recovery: Recovery,
) -> Parser<'_> {
    tts.desugar_doc_comments();
    Parser::new(psess, tts, rustc_parse::MACRO_ARGUMENTS).recovery(recovery)
}
