use crate::base::{DummyResult, ExtCtxt, MacResult, TTMacroExpander};
use crate::base::{SyntaxExtension, SyntaxExtensionKind};
use crate::expand::{ensure_complete_parse, parse_ast_fragment, AstFragment, AstFragmentKind};
use crate::mbe;
use crate::mbe::diagnostics::{annotate_doc_comment, parse_failure_msg};
use crate::mbe::macro_check;
use crate::mbe::macro_parser::{Error, ErrorReported, Failure, Success, TtParser};
use crate::mbe::macro_parser::{MatchedSeq, MatchedTokenTree, MatcherLoc};
use crate::mbe::transcribe::transcribe;

use rustc_ast as ast;
use rustc_ast::token::{self, Delimiter, NonterminalKind, Token, TokenKind, TokenKind::*};
use rustc_ast::tokenstream::{DelimSpan, TokenStream, TokenTree};
use rustc_ast::{NodeId, DUMMY_NODE_ID};
use rustc_ast_pretty::pprust;
use rustc_attr::{self as attr, TransparencyError};
use rustc_data_structures::fx::{FxHashMap, FxIndexMap};
use rustc_errors::{Applicability, ErrorGuaranteed};
use rustc_feature::Features;
use rustc_lint_defs::builtin::{
    RUST_2021_INCOMPATIBLE_OR_PATTERNS, SEMICOLON_IN_EXPRESSIONS_FROM_MACROS,
};
use rustc_lint_defs::BuiltinLintDiagnostics;
use rustc_parse::parser::{Parser, Recovery};
use rustc_session::parse::ParseSess;
use rustc_session::Session;
use rustc_span::edition::Edition;
use rustc_span::hygiene::Transparency;
use rustc_span::symbol::{kw, sym, Ident, MacroRulesNormalizedIdent};
use rustc_span::Span;

use std::borrow::Cow;
use std::collections::hash_map::Entry;
use std::{mem, slice};

use super::diagnostics;
use super::macro_parser::{NamedMatches, NamedParseResult};

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
                diagnostics::emit_frag_parse_err(err, parser, snapshot, site_span, arm_span, kind);
                return kind.dummy(site_span);
            }
        };

        // We allow semicolons at the end of expressions -- e.g., the semicolon in
        // `macro_rules! m { () => { panic!(); } }` isn't parsed by `.parse_expr()`,
        // but `m!()` is allowed in expression positions (cf. issue #34706).
        if kind == AstFragmentKind::Expr && parser.token == token::Semi {
            if is_local {
                parser.sess.buffer_lint_with_diagnostic(
                    SEMICOLON_IN_EXPRESSIONS_FROM_MACROS,
                    parser.token.span,
                    lint_node_id,
                    "trailing semicolon in macro used in expression position",
                    BuiltinLintDiagnostics::TrailingMacro(is_trailing_mac, macro_ident),
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
        expand_macro(
            cx,
            sp,
            self.span,
            self.node_id,
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
    fn build_failure(tok: Token, position: usize, msg: &'static str) -> Self::Failure;

    /// This is called before trying to match next MatcherLoc on the current token.
    fn before_match_loc(&mut self, _parser: &TtParser, _matcher: &'matcher MatcherLoc) {}

    /// This is called after an arm has been parsed, either successfully or unsuccessfully. When this is called,
    /// `before_match_loc` was called at least once (with a `MatcherLoc::Eof`).
    fn after_arm(&mut self, _result: &NamedParseResult<Self::Failure>) {}

    /// For tracing.
    fn description() -> &'static str;

    fn recovery() -> Recovery {
        Recovery::Forbidden
    }
}

/// A noop tracker that is used in the hot path of the expansion, has zero overhead thanks to monomorphization.
pub(super) struct NoopTracker;

impl<'matcher> Tracker<'matcher> for NoopTracker {
    type Failure = ();

    fn build_failure(_tok: Token, _position: usize, _msg: &'static str) -> Self::Failure {}

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
    let sess = &cx.sess.parse_sess;
    // Macros defined in the current crate have a real node id,
    // whereas macros from an external crate have a dummy id.
    let is_local = node_id != DUMMY_NODE_ID;

    if cx.trace_macros() {
        let msg = format!("expanding `{}! {{ {} }}`", name, pprust::tts_to_string(&arg));
        trace_macros_note(&mut cx.expansions, sp, msg);
    }

    // Track nothing for the best performance.
    let try_success_result = try_match_macro(sess, name, &arg, lhses, &mut NoopTracker);

    match try_success_result {
        Ok((i, named_matches)) => {
            let (rhs, rhs_span): (&mbe::Delimited, DelimSpan) = match &rhses[i] {
                mbe::TokenTree::Delimited(span, delimited) => (&delimited, *span),
                _ => cx.span_bug(sp, "malformed macro rhs"),
            };
            let arm_span = rhses[i].span();

            // rhs has holes ( `$id` and `$(...)` that need filled)
            let mut tts = match transcribe(cx, &named_matches, &rhs, rhs_span, transparency) {
                Ok(tts) => tts,
                Err(mut err) => {
                    err.emit();
                    return DummyResult::any(arm_span);
                }
            };

            // Replace all the tokens for the corresponding positions in the macro, to maintain
            // proper positions in error reporting, while maintaining the macro_backtrace.
            if tts.len() == rhs.tts.len() {
                tts = tts.map_enumerated(|i, tt| {
                    let mut tt = tt.clone();
                    let rhs_tt = &rhs.tts[i];
                    let ctxt = tt.span().ctxt();
                    match (&mut tt, rhs_tt) {
                        // preserve the delim spans if able
                        (
                            TokenTree::Delimited(target_sp, ..),
                            mbe::TokenTree::Delimited(source_sp, ..),
                        ) => {
                            target_sp.open = source_sp.open.with_ctxt(ctxt);
                            target_sp.close = source_sp.close.with_ctxt(ctxt);
                        }
                        _ => {
                            let sp = rhs_tt.span().with_ctxt(ctxt);
                            tt.set_span(sp);
                        }
                    }
                    tt
                });
            }

            if cx.trace_macros() {
                let msg = format!("to `{}`", pprust::tts_to_string(&tts));
                trace_macros_note(&mut cx.expansions, sp, msg);
            }

            let mut p = Parser::new(sess, tts, false, None);
            p.last_type_ascription = cx.current_expansion.prior_type_ascription;

            if is_local {
                cx.resolver.record_macro_rule_usage(node_id, i);
            }

            // Let the context choose how to interpret the result.
            // Weird, but useful for X-macros.
            return Box::new(ParserAnyMacro {
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
            });
        }
        Err(CanRetry::No(_)) => {
            debug!("Will not retry matching as an error was emitted already");
            return DummyResult::any(sp);
        }
        Err(CanRetry::Yes) => {
            // Retry and emit a better error below.
        }
    }

    diagnostics::failed_to_match_macro(cx, sp, def_span, name, arg, lhses)
}

pub(super) enum CanRetry {
    Yes,
    /// We are not allowed to retry macro expansion as a fatal error has been emitted already.
    No(ErrorGuaranteed),
}

/// Try expanding the macro. Returns the index of the successful arm and its named_matches if it was successful,
/// and nothing if it failed. On failure, it's the callers job to use `track` accordingly to record all errors
/// correctly.
#[instrument(level = "debug", skip(sess, arg, lhses, track), fields(tracking = %T::description()))]
pub(super) fn try_match_macro<'matcher, T: Tracker<'matcher>>(
    sess: &ParseSess,
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
    let parser = parser_from_cx(sess, arg.clone(), T::recovery());
    // Try each arm's matchers.
    let mut tt_parser = TtParser::new(name);
    for (i, lhs) in lhses.iter().enumerate() {
        let _tracing_span = trace_span!("Matching arm", %i);

        // Take a snapshot of the state of pre-expansion gating at this point.
        // This is used so that if a matcher is not `Success(..)`ful,
        // then the spans which became gated when parsing the unsuccessful matcher
        // are not recorded. On the first `Success(..)`ful matcher, the spans are merged.
        let mut gated_spans_snapshot = mem::take(&mut *sess.gated_spans.spans.borrow_mut());

        let result = tt_parser.parse_tt(&mut Cow::Borrowed(&parser), lhs, track);

        track.after_arm(&result);

        match result {
            Success(named_matches) => {
                debug!("Parsed arm successfully");
                // The matcher was `Success(..)`ful.
                // Merge the gated spans from parsing the matcher with the pre-existing ones.
                sess.gated_spans.merge(gated_spans_snapshot);

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
        mem::swap(&mut gated_spans_snapshot, &mut sess.gated_spans.spans.borrow_mut());
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
    def: &ast::Item,
    edition: Edition,
) -> (SyntaxExtension, Vec<(usize, Span)>) {
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
    let dummy_syn_ext = || (mk_syn_ext(Box::new(macro_rules_dummy_expander)), Vec::new());

    let diag = &sess.parse_sess.span_diagnostic;
    let lhs_nm = Ident::new(sym::lhs, def.span);
    let rhs_nm = Ident::new(sym::rhs, def.span);
    let tt_spec = Some(NonterminalKind::TT);

    let macro_def = match &def.kind {
        ast::ItemKind::MacroDef(def) => def,
        _ => unreachable!(),
    };
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
            },
        ),
        // to phase into semicolon-termination instead of semicolon-separation
        mbe::TokenTree::Sequence(
            DelimSpan::dummy(),
            mbe::SequenceRepetition {
                tts: vec![mbe::TokenTree::token(
                    if macro_rules { token::Semi } else { token::Comma },
                    def.span,
                )],
                separator: None,
                kleene: mbe::KleeneToken::new(mbe::KleeneOp::ZeroOrMore, def.span),
                num_captures: 0,
            },
        ),
    ];
    // Convert it into `MatcherLoc` form.
    let argument_gram = mbe::macro_parser::compute_locs(&argument_gram);

    let create_parser = || {
        let body = macro_def.body.tokens.clone();
        Parser::new(&sess.parse_sess, body, true, rustc_parse::MACRO_ARGUMENTS)
    };

    let parser = create_parser();
    let mut tt_parser =
        TtParser::new(Ident::with_dummy_span(if macro_rules { kw::MacroRules } else { kw::Macro }));
    let argument_map =
        match tt_parser.parse_tt(&mut Cow::Owned(parser), &argument_gram, &mut NoopTracker) {
            Success(m) => m,
            Failure(()) => {
                // The fast `NoopTracker` doesn't have any info on failure, so we need to retry it with another one
                // that gives us the information we need.
                // For this we need to reclone the macro body as the previous parser consumed it.
                let retry_parser = create_parser();

                let parse_result = tt_parser.parse_tt(
                    &mut Cow::Owned(retry_parser),
                    &argument_gram,
                    &mut diagnostics::FailureForwarder,
                );
                let Failure((token, _, msg)) = parse_result else {
                    unreachable!("matcher returned something other than Failure after retry");
                };

                let s = parse_failure_msg(&token);
                let sp = token.span.substitute_dummy(def.span);
                let mut err = sess.parse_sess.span_diagnostic.struct_span_err(sp, &s);
                err.span_label(sp, msg);
                annotate_doc_comment(&mut err, sess.source_map(), sp);
                err.emit();
                return dummy_syn_ext();
            }
            Error(sp, msg) => {
                sess.parse_sess
                    .span_diagnostic
                    .struct_span_err(sp.substitute_dummy(def.span), &msg)
                    .emit();
                return dummy_syn_ext();
            }
            ErrorReported(_) => {
                return dummy_syn_ext();
            }
        };

    let mut valid = true;

    // Extract the arguments:
    let lhses = match &argument_map[&MacroRulesNormalizedIdent::new(lhs_nm)] {
        MatchedSeq(s) => s
            .iter()
            .map(|m| {
                if let MatchedTokenTree(tt) = m {
                    let tt = mbe::quoted::parse(
                        TokenStream::new(vec![tt.clone()]),
                        true,
                        &sess.parse_sess,
                        def.id,
                        features,
                        edition,
                    )
                    .pop()
                    .unwrap();
                    valid &= check_lhs_nt_follows(&sess.parse_sess, &def, &tt);
                    return tt;
                }
                sess.parse_sess.span_diagnostic.span_bug(def.span, "wrong-structured lhs")
            })
            .collect::<Vec<mbe::TokenTree>>(),
        _ => sess.parse_sess.span_diagnostic.span_bug(def.span, "wrong-structured lhs"),
    };

    let rhses = match &argument_map[&MacroRulesNormalizedIdent::new(rhs_nm)] {
        MatchedSeq(s) => s
            .iter()
            .map(|m| {
                if let MatchedTokenTree(tt) = m {
                    return mbe::quoted::parse(
                        TokenStream::new(vec![tt.clone()]),
                        false,
                        &sess.parse_sess,
                        def.id,
                        features,
                        edition,
                    )
                    .pop()
                    .unwrap();
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

    let (transparency, transparency_error) = attr::find_transparency(&def.attrs, macro_rules);
    match transparency_error {
        Some(TransparencyError::UnknownTransparency(value, span)) => {
            diag.span_err(span, &format!("unknown macro transparency: `{}`", value));
        }
        Some(TransparencyError::MultipleTransparencyAttrs(old_span, new_span)) => {
            diag.span_err(vec![old_span, new_span], "multiple macro transparency attributes");
        }
        None => {}
    }

    // Compute the spans of the macro rules for unused rule linting.
    // To avoid warning noise, only consider the rules of this
    // macro for the lint, if all rules are valid.
    // Also, we are only interested in non-foreign macros.
    let rule_spans = if valid && def.id != DUMMY_NODE_ID {
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
    // actual matching. Unless the matcher is invalid.
    let lhses = if valid {
        lhses
            .iter()
            .map(|lhs| {
                // Ignore the delimiters around the matcher.
                match lhs {
                    mbe::TokenTree::Delimited(_, delimited) => {
                        mbe::macro_parser::compute_locs(&delimited.tts)
                    }
                    _ => sess.parse_sess.span_diagnostic.span_bug(def.span, "malformed macro lhs"),
                }
            })
            .collect()
    } else {
        vec![]
    };

    let expander = Box::new(MacroRulesMacroExpander {
        name: def.ident,
        span: def.span,
        node_id: def.id,
        transparency,
        lhses,
        rhses,
        valid,
    });
    (mk_syn_ext(expander), rule_spans)
}

fn check_lhs_nt_follows(sess: &ParseSess, def: &ast::Item, lhs: &mbe::TokenTree) -> bool {
    // lhs is going to be like TokenTree::Delimited(...), where the
    // entire lhs is those tts. Or, it can be a "bare sequence", not wrapped in parens.
    if let mbe::TokenTree::Delimited(_, delimited) = lhs {
        check_matcher(sess, def, &delimited.tts)
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
        match tt {
            TokenTree::Token(..)
            | TokenTree::MetaVar(..)
            | TokenTree::MetaVarDecl(..)
            | TokenTree::MetaVarExpr(..) => (),
            TokenTree::Delimited(_, del) => {
                if !check_lhs_no_empty_seq(sess, &del.tts) {
                    return false;
                }
            }
            TokenTree::Sequence(span, seq) => {
                if seq.separator.is_none()
                    && seq.tts.iter().all(|seq_tt| match seq_tt {
                        TokenTree::MetaVarDecl(_, _, Some(NonterminalKind::Vis)) => true,
                        TokenTree::Sequence(_, sub_seq) => {
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
        _ => {
            sess.span_diagnostic.span_err(rhs.span(), "macro rhs must be delimited");
        }
    }
    false
}

fn check_matcher(sess: &ParseSess, def: &ast::Item, matcher: &[mbe::TokenTree]) -> bool {
    let first_sets = FirstSets::new(matcher);
    let empty_suffix = TokenSet::empty();
    let err = sess.span_diagnostic.err_count();
    check_matcher_core(sess, def, &first_sets, matcher, &empty_suffix);
    err == sess.span_diagnostic.err_count()
}

fn has_compile_error_macro(rhs: &mbe::TokenTree) -> bool {
    match rhs {
        mbe::TokenTree::Delimited(_sp, d) => {
            let has_compile_error = d.tts.array_windows::<3>().any(|[ident, bang, args]| {
                if let mbe::TokenTree::Token(ident) = ident &&
                        let TokenKind::Ident(ident, _) = ident.kind &&
                        ident == sym::compile_error &&
                        let mbe::TokenTree::Token(bang) = bang &&
                        let TokenKind::Not = bang.kind &&
                        let mbe::TokenTree::Delimited(_, del) = args &&
                        del.delim != Delimiter::Invisible
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
                    TokenTree::Delimited(span, delimited) => {
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
                            first.add_one_maybe(TtHandle::from_token(sep.clone()));
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
                TokenTree::Delimited(span, delimited) => {
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
                        first.add_one_maybe(TtHandle::from_token(sep.clone()));
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

// Most `mbe::TokenTree`s are pre-existing in the matcher, but some are defined
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
            TtHandle::Token(token_tt) => &token_tt,
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
                TtHandle::Token(mbe::TokenTree::Token(tok.clone()))
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
    sess: &ParseSess,
    def: &ast::Item,
    first_sets: &FirstSets<'tt>,
    matcher: &'tt [mbe::TokenTree],
    follow: &TokenSet<'tt>,
) -> TokenSet<'tt> {
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
            TokenTree::Delimited(span, d) => {
                let my_suffix = TokenSet::singleton(TtHandle::from_token_kind(
                    token::CloseDelim(d.delim),
                    span.close,
                ));
                check_matcher_core(sess, def, first_sets, &d.tts, &my_suffix);
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
                    new.add_one_maybe(TtHandle::from_token(sep.clone()));
                    &new
                } else {
                    &suffix_first
                };

                // At this point, `suffix_first` is built, and
                // `my_suffix` is some TokenSet that we can use
                // for checking the interior of `seq_rep`.
                let next = check_matcher_core(sess, def, first_sets, &seq_rep.tts, my_suffix);
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
                    if def.id != DUMMY_NODE_ID
                        && matches!(kind, NonterminalKind::PatParam { inferred: true })
                        && matches!(next_token, TokenTree::Token(token) if token.kind == BinOp(token::BinOpToken::Or))
                    {
                        // It is suggestion to use pat_param, for example: $x:pat -> $x:pat_param.
                        let suggestion = quoted_tt_to_string(&TokenTree::MetaVarDecl(
                            span,
                            name,
                            Some(NonterminalKind::PatParam { inferred: false }),
                        ));
                        sess.buffer_lint_with_diagnostic(
                            &RUST_2021_INCOMPATIBLE_OR_PATTERNS,
                            span,
                            ast::CRATE_NODE_ID,
                            "the meaning of the `pat` fragment specifier is changing in Rust 2021, which may affect this macro",
                            BuiltinLintDiagnostics::OrPatternsBackCompat(span, suggestion),
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

                            if kind == NonterminalKind::PatWithOr
                                && sess.edition.rust_2021()
                                && next_token.is_token(&BinOp(token::BinOpToken::Or))
                            {
                                let suggestion = quoted_tt_to_string(&TokenTree::MetaVarDecl(
                                    span,
                                    name,
                                    Some(NonterminalKind::PatParam { inferred: false }),
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
                                    err.note(&format!(
                                        "only {} is allowed after `{}` fragments",
                                        t, kind,
                                    ));
                                }
                                ts => {
                                    err.note(&format!(
                                        "{}{} or {}",
                                        msg,
                                        ts[..ts.len() - 1].to_vec().join(", "),
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
            NonterminalKind::PatParam { .. } => {
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
            NonterminalKind::PatWithOr { .. } => {
                const TOKENS: &[&str] = &["`=>`", "`,`", "`=`", "`if`", "`in`"];
                match tok {
                    TokenTree::Token(token) => match token.kind {
                        FatArrow | Comma | Eq => IsInFollow::Yes,
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
                        OpenDelim(Delimiter::Brace)
                        | OpenDelim(Delimiter::Bracket)
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
    match tt {
        mbe::TokenTree::Token(token) => pprust::token_to_string(&token).into(),
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

pub(super) fn parser_from_cx(sess: &ParseSess, tts: TokenStream, recovery: Recovery) -> Parser<'_> {
    Parser::new(sess, tts, true, rustc_parse::MACRO_ARGUMENTS).recovery(recovery)
}
