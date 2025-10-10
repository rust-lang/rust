use std::borrow::Cow;
use std::collections::hash_map::Entry;
use std::sync::Arc;
use std::{mem, slice};

use ast::token::IdentIsRaw;
use rustc_ast::token::NtPatKind::*;
use rustc_ast::token::TokenKind::*;
use rustc_ast::token::{self, Delimiter, NonterminalKind, Token, TokenKind};
use rustc_ast::tokenstream::{self, DelimSpan, TokenStream};
use rustc_ast::{self as ast, DUMMY_NODE_ID, NodeId, Safety};
use rustc_ast_pretty::pprust;
use rustc_data_structures::fx::{FxHashMap, FxIndexMap};
use rustc_errors::{Applicability, Diag, ErrorGuaranteed, MultiSpan};
use rustc_feature::Features;
use rustc_hir as hir;
use rustc_hir::attrs::AttributeKind;
use rustc_hir::def::MacroKinds;
use rustc_hir::find_attr;
use rustc_lint_defs::builtin::{
    RUST_2021_INCOMPATIBLE_OR_PATTERNS, SEMICOLON_IN_EXPRESSIONS_FROM_MACROS,
};
use rustc_parse::exp;
use rustc_parse::parser::{Parser, Recovery};
use rustc_session::Session;
use rustc_session::parse::{ParseSess, feature_err};
use rustc_span::edition::Edition;
use rustc_span::hygiene::Transparency;
use rustc_span::{Ident, Span, Symbol, kw, sym};
use tracing::{debug, instrument, trace, trace_span};

use super::diagnostics::{FailedMacro, failed_to_match_macro};
use super::macro_parser::{NamedMatches, NamedParseResult};
use super::{SequenceRepetition, diagnostics};
use crate::base::{
    AttrProcMacro, BangProcMacro, DummyResult, ExpandResult, ExtCtxt, MacResult,
    MacroExpanderResult, SyntaxExtension, SyntaxExtensionKind, TTMacroExpander,
};
use crate::errors;
use crate::expand::{AstFragment, AstFragmentKind, ensure_complete_parse, parse_ast_fragment};
use crate::mbe::macro_check::check_meta_variables;
use crate::mbe::macro_parser::{Error, ErrorReported, Failure, MatcherLoc, Success, TtParser};
use crate::mbe::quoted::{RulePart, parse_one_tt};
use crate::mbe::transcribe::transcribe;
use crate::mbe::{self, KleeneOp};

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
                    errors::TrailingMacro { is_trailing: is_trailing_mac, name: macro_ident },
                );
            }
            parser.bump();
        }

        // Make sure we don't have any tokens left to parse so we don't silently drop anything.
        let path = ast::Path::from_ident(macro_ident.with_span_pos(site_span));
        ensure_complete_parse(parser, &path, kind.name(), site_span);
        fragment
    }

    #[instrument(skip(cx, tts))]
    pub(crate) fn from_tts<'cx>(
        cx: &'cx mut ExtCtxt<'a>,
        tts: TokenStream,
        site_span: Span,
        arm_span: Span,
        is_local: bool,
        macro_ident: Ident,
    ) -> Self {
        Self {
            parser: Parser::new(&cx.sess.psess, tts, None),

            // Pass along the original expansion site and the name of the macro
            // so we can print a useful error message if the parse of the expanded
            // macro leaves unparsed tokens.
            site_span,
            macro_ident,
            lint_node_id: cx.current_expansion.lint_node_id,
            is_trailing_mac: cx.current_expansion.is_trailing_mac,
            arm_span,
            is_local,
        }
    }
}

pub(super) enum MacroRule {
    /// A function-style rule, for use with `m!()`
    Func { lhs: Vec<MatcherLoc>, lhs_span: Span, rhs: mbe::TokenTree },
    /// An attr rule, for use with `#[m]`
    Attr {
        unsafe_rule: bool,
        args: Vec<MatcherLoc>,
        args_span: Span,
        body: Vec<MatcherLoc>,
        body_span: Span,
        rhs: mbe::TokenTree,
    },
    /// A derive rule, for use with `#[m]`
    Derive { body: Vec<MatcherLoc>, body_span: Span, rhs: mbe::TokenTree },
}

pub struct MacroRulesMacroExpander {
    node_id: NodeId,
    name: Ident,
    span: Span,
    transparency: Transparency,
    kinds: MacroKinds,
    rules: Vec<MacroRule>,
}

impl MacroRulesMacroExpander {
    pub fn get_unused_rule(&self, rule_i: usize) -> Option<(&Ident, MultiSpan)> {
        // If the rhs contains an invocation like `compile_error!`, don't report it as unused.
        let (span, rhs) = match self.rules[rule_i] {
            MacroRule::Func { lhs_span, ref rhs, .. } => (MultiSpan::from_span(lhs_span), rhs),
            MacroRule::Attr { args_span, body_span, ref rhs, .. } => {
                (MultiSpan::from_spans(vec![args_span, body_span]), rhs)
            }
            MacroRule::Derive { body_span, ref rhs, .. } => (MultiSpan::from_span(body_span), rhs),
        };
        if has_compile_error_macro(rhs) { None } else { Some((&self.name, span)) }
    }

    pub fn kinds(&self) -> MacroKinds {
        self.kinds
    }

    pub fn expand_derive(
        &self,
        cx: &mut ExtCtxt<'_>,
        sp: Span,
        body: &TokenStream,
    ) -> Result<TokenStream, ErrorGuaranteed> {
        // This is similar to `expand_macro`, but they have very different signatures, and will
        // diverge further once derives support arguments.
        let Self { name, ref rules, node_id, .. } = *self;
        let psess = &cx.sess.psess;

        if cx.trace_macros() {
            let msg = format!("expanding `#[derive({name})] {}`", pprust::tts_to_string(body));
            trace_macros_note(&mut cx.expansions, sp, msg);
        }

        match try_match_macro_derive(psess, name, body, rules, &mut NoopTracker) {
            Ok((rule_index, rule, named_matches)) => {
                let MacroRule::Derive { rhs, .. } = rule else {
                    panic!("try_match_macro_derive returned non-derive rule");
                };
                let mbe::TokenTree::Delimited(rhs_span, _, rhs) = rhs else {
                    cx.dcx().span_bug(sp, "malformed macro derive rhs");
                };

                let id = cx.current_expansion.id;
                let tts = transcribe(psess, &named_matches, rhs, *rhs_span, self.transparency, id)
                    .map_err(|e| e.emit())?;

                if cx.trace_macros() {
                    let msg = format!("to `{}`", pprust::tts_to_string(&tts));
                    trace_macros_note(&mut cx.expansions, sp, msg);
                }

                if is_defined_in_current_crate(node_id) {
                    cx.resolver.record_macro_rule_usage(node_id, rule_index);
                }

                Ok(tts)
            }
            Err(CanRetry::No(guar)) => Err(guar),
            Err(CanRetry::Yes) => {
                let (_, guar) = failed_to_match_macro(
                    cx.psess(),
                    sp,
                    self.span,
                    name,
                    FailedMacro::Derive,
                    body,
                    rules,
                );
                cx.macro_error_and_trace_macros_diag();
                Err(guar)
            }
        }
    }
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
            &self.rules,
        ))
    }
}

impl AttrProcMacro for MacroRulesMacroExpander {
    fn expand(
        &self,
        _cx: &mut ExtCtxt<'_>,
        _sp: Span,
        _args: TokenStream,
        _body: TokenStream,
    ) -> Result<TokenStream, ErrorGuaranteed> {
        unreachable!("`expand` called on `MacroRulesMacroExpander`, expected `expand_with_safety`")
    }

    fn expand_with_safety(
        &self,
        cx: &mut ExtCtxt<'_>,
        safety: Safety,
        sp: Span,
        args: TokenStream,
        body: TokenStream,
    ) -> Result<TokenStream, ErrorGuaranteed> {
        expand_macro_attr(
            cx,
            sp,
            self.span,
            self.node_id,
            self.name,
            self.transparency,
            safety,
            args,
            body,
            &self.rules,
        )
    }
}

struct DummyBang(ErrorGuaranteed);

impl BangProcMacro for DummyBang {
    fn expand<'cx>(
        &self,
        _: &'cx mut ExtCtxt<'_>,
        _: Span,
        _: TokenStream,
    ) -> Result<TokenStream, ErrorGuaranteed> {
        Err(self.0)
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
    fn after_arm(&mut self, _in_body: bool, _result: &NamedParseResult<Self::Failure>) {}

    /// For tracing.
    fn description() -> &'static str;

    fn recovery() -> Recovery {
        Recovery::Forbidden
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

/// Expands the rules based macro defined by `rules` for a given input `arg`.
#[instrument(skip(cx, transparency, arg, rules))]
fn expand_macro<'cx>(
    cx: &'cx mut ExtCtxt<'_>,
    sp: Span,
    def_span: Span,
    node_id: NodeId,
    name: Ident,
    transparency: Transparency,
    arg: TokenStream,
    rules: &[MacroRule],
) -> Box<dyn MacResult + 'cx> {
    let psess = &cx.sess.psess;

    if cx.trace_macros() {
        let msg = format!("expanding `{}! {{ {} }}`", name, pprust::tts_to_string(&arg));
        trace_macros_note(&mut cx.expansions, sp, msg);
    }

    // Track nothing for the best performance.
    let try_success_result = try_match_macro(psess, name, &arg, rules, &mut NoopTracker);

    match try_success_result {
        Ok((rule_index, rule, named_matches)) => {
            let MacroRule::Func { rhs, .. } = rule else {
                panic!("try_match_macro returned non-func rule");
            };
            let mbe::TokenTree::Delimited(rhs_span, _, rhs) = rhs else {
                cx.dcx().span_bug(sp, "malformed macro rhs");
            };
            let arm_span = rhs_span.entire();

            // rhs has holes ( `$id` and `$(...)` that need filled)
            let id = cx.current_expansion.id;
            let tts = match transcribe(psess, &named_matches, rhs, *rhs_span, transparency, id) {
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

            let is_local = is_defined_in_current_crate(node_id);
            if is_local {
                cx.resolver.record_macro_rule_usage(node_id, rule_index);
            }

            // Let the context choose how to interpret the result. Weird, but useful for X-macros.
            Box::new(ParserAnyMacro::from_tts(cx, tts, sp, arm_span, is_local, name))
        }
        Err(CanRetry::No(guar)) => {
            debug!("Will not retry matching as an error was emitted already");
            DummyResult::any(sp, guar)
        }
        Err(CanRetry::Yes) => {
            // Retry and emit a better error.
            let (span, guar) = failed_to_match_macro(
                cx.psess(),
                sp,
                def_span,
                name,
                FailedMacro::Func,
                &arg,
                rules,
            );
            cx.macro_error_and_trace_macros_diag();
            DummyResult::any(span, guar)
        }
    }
}

/// Expands the rules based macro defined by `rules` for a given attribute `args` and `body`.
#[instrument(skip(cx, transparency, args, body, rules))]
fn expand_macro_attr(
    cx: &mut ExtCtxt<'_>,
    sp: Span,
    def_span: Span,
    node_id: NodeId,
    name: Ident,
    transparency: Transparency,
    safety: Safety,
    args: TokenStream,
    body: TokenStream,
    rules: &[MacroRule],
) -> Result<TokenStream, ErrorGuaranteed> {
    let psess = &cx.sess.psess;
    // Macros defined in the current crate have a real node id,
    // whereas macros from an external crate have a dummy id.
    let is_local = node_id != DUMMY_NODE_ID;

    if cx.trace_macros() {
        let msg = format!(
            "expanding `#[{name}({})] {}`",
            pprust::tts_to_string(&args),
            pprust::tts_to_string(&body),
        );
        trace_macros_note(&mut cx.expansions, sp, msg);
    }

    // Track nothing for the best performance.
    match try_match_macro_attr(psess, name, &args, &body, rules, &mut NoopTracker) {
        Ok((i, rule, named_matches)) => {
            let MacroRule::Attr { rhs, unsafe_rule, .. } = rule else {
                panic!("try_macro_match_attr returned non-attr rule");
            };
            let mbe::TokenTree::Delimited(rhs_span, _, rhs) = rhs else {
                cx.dcx().span_bug(sp, "malformed macro rhs");
            };

            match (safety, unsafe_rule) {
                (Safety::Default, false) | (Safety::Unsafe(_), true) => {}
                (Safety::Default, true) => {
                    cx.dcx().span_err(sp, "unsafe attribute invocation requires `unsafe`");
                }
                (Safety::Unsafe(span), false) => {
                    cx.dcx().span_err(span, "unnecessary `unsafe` on safe attribute invocation");
                }
                (Safety::Safe(span), _) => {
                    cx.dcx().span_bug(span, "unexpected `safe` keyword");
                }
            }

            let id = cx.current_expansion.id;
            let tts = transcribe(psess, &named_matches, rhs, *rhs_span, transparency, id)
                .map_err(|e| e.emit())?;

            if cx.trace_macros() {
                let msg = format!("to `{}`", pprust::tts_to_string(&tts));
                trace_macros_note(&mut cx.expansions, sp, msg);
            }

            if is_local {
                cx.resolver.record_macro_rule_usage(node_id, i);
            }

            Ok(tts)
        }
        Err(CanRetry::No(guar)) => Err(guar),
        Err(CanRetry::Yes) => {
            // Retry and emit a better error.
            let (_, guar) = failed_to_match_macro(
                cx.psess(),
                sp,
                def_span,
                name,
                FailedMacro::Attr(&args),
                &body,
                rules,
            );
            cx.trace_macros_diag();
            Err(guar)
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
#[instrument(level = "debug", skip(psess, arg, rules, track), fields(tracking = %T::description()))]
pub(super) fn try_match_macro<'matcher, T: Tracker<'matcher>>(
    psess: &ParseSess,
    name: Ident,
    arg: &TokenStream,
    rules: &'matcher [MacroRule],
    track: &mut T,
) -> Result<(usize, &'matcher MacroRule, NamedMatches), CanRetry> {
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
    for (i, rule) in rules.iter().enumerate() {
        let MacroRule::Func { lhs, .. } = rule else { continue };
        let _tracing_span = trace_span!("Matching arm", %i);

        // Take a snapshot of the state of pre-expansion gating at this point.
        // This is used so that if a matcher is not `Success(..)`ful,
        // then the spans which became gated when parsing the unsuccessful matcher
        // are not recorded. On the first `Success(..)`ful matcher, the spans are merged.
        let mut gated_spans_snapshot = mem::take(&mut *psess.gated_spans.spans.borrow_mut());

        let result = tt_parser.parse_tt(&mut Cow::Borrowed(&parser), lhs, track);

        track.after_arm(true, &result);

        match result {
            Success(named_matches) => {
                debug!("Parsed arm successfully");
                // The matcher was `Success(..)`ful.
                // Merge the gated spans from parsing the matcher with the preexisting ones.
                psess.gated_spans.merge(gated_spans_snapshot);

                return Ok((i, rule, named_matches));
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

/// Try expanding the macro attribute. Returns the index of the successful arm and its
/// named_matches if it was successful, and nothing if it failed. On failure, it's the caller's job
/// to use `track` accordingly to record all errors correctly.
#[instrument(level = "debug", skip(psess, attr_args, attr_body, rules, track), fields(tracking = %T::description()))]
pub(super) fn try_match_macro_attr<'matcher, T: Tracker<'matcher>>(
    psess: &ParseSess,
    name: Ident,
    attr_args: &TokenStream,
    attr_body: &TokenStream,
    rules: &'matcher [MacroRule],
    track: &mut T,
) -> Result<(usize, &'matcher MacroRule, NamedMatches), CanRetry> {
    // This uses the same strategy as `try_match_macro`
    let args_parser = parser_from_cx(psess, attr_args.clone(), T::recovery());
    let body_parser = parser_from_cx(psess, attr_body.clone(), T::recovery());
    let mut tt_parser = TtParser::new(name);
    for (i, rule) in rules.iter().enumerate() {
        let MacroRule::Attr { args, body, .. } = rule else { continue };

        let mut gated_spans_snapshot = mem::take(&mut *psess.gated_spans.spans.borrow_mut());

        let result = tt_parser.parse_tt(&mut Cow::Borrowed(&args_parser), args, track);
        track.after_arm(false, &result);

        let mut named_matches = match result {
            Success(named_matches) => named_matches,
            Failure(_) => {
                mem::swap(&mut gated_spans_snapshot, &mut psess.gated_spans.spans.borrow_mut());
                continue;
            }
            Error(_, _) => return Err(CanRetry::Yes),
            ErrorReported(guar) => return Err(CanRetry::No(guar)),
        };

        let result = tt_parser.parse_tt(&mut Cow::Borrowed(&body_parser), body, track);
        track.after_arm(true, &result);

        match result {
            Success(body_named_matches) => {
                psess.gated_spans.merge(gated_spans_snapshot);
                #[allow(rustc::potential_query_instability)]
                named_matches.extend(body_named_matches);
                return Ok((i, rule, named_matches));
            }
            Failure(_) => {
                mem::swap(&mut gated_spans_snapshot, &mut psess.gated_spans.spans.borrow_mut())
            }
            Error(_, _) => return Err(CanRetry::Yes),
            ErrorReported(guar) => return Err(CanRetry::No(guar)),
        }
    }

    Err(CanRetry::Yes)
}

/// Try expanding the macro derive. Returns the index of the successful arm and its
/// named_matches if it was successful, and nothing if it failed. On failure, it's the caller's job
/// to use `track` accordingly to record all errors correctly.
#[instrument(level = "debug", skip(psess, body, rules, track), fields(tracking = %T::description()))]
pub(super) fn try_match_macro_derive<'matcher, T: Tracker<'matcher>>(
    psess: &ParseSess,
    name: Ident,
    body: &TokenStream,
    rules: &'matcher [MacroRule],
    track: &mut T,
) -> Result<(usize, &'matcher MacroRule, NamedMatches), CanRetry> {
    // This uses the same strategy as `try_match_macro`
    let body_parser = parser_from_cx(psess, body.clone(), T::recovery());
    let mut tt_parser = TtParser::new(name);
    for (i, rule) in rules.iter().enumerate() {
        let MacroRule::Derive { body, .. } = rule else { continue };

        let mut gated_spans_snapshot = mem::take(&mut *psess.gated_spans.spans.borrow_mut());

        let result = tt_parser.parse_tt(&mut Cow::Borrowed(&body_parser), body, track);
        track.after_arm(true, &result);

        match result {
            Success(named_matches) => {
                psess.gated_spans.merge(gated_spans_snapshot);
                return Ok((i, rule, named_matches));
            }
            Failure(_) => {
                mem::swap(&mut gated_spans_snapshot, &mut psess.gated_spans.spans.borrow_mut())
            }
            Error(_, _) => return Err(CanRetry::Yes),
            ErrorReported(guar) => return Err(CanRetry::No(guar)),
        }
    }

    Err(CanRetry::Yes)
}

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
) -> (SyntaxExtension, usize) {
    let mk_syn_ext = |kind| {
        let is_local = is_defined_in_current_crate(node_id);
        SyntaxExtension::new(sess, kind, span, Vec::new(), edition, ident.name, attrs, is_local)
    };
    let dummy_syn_ext =
        |guar| (mk_syn_ext(SyntaxExtensionKind::Bang(Arc::new(DummyBang(guar)))), 0);

    let macro_rules = macro_def.macro_rules;
    let exp_sep = if macro_rules { exp!(Semi) } else { exp!(Comma) };

    let body = macro_def.body.tokens.clone();
    let mut p = Parser::new(&sess.psess, body, rustc_parse::MACRO_ARGUMENTS);

    // Don't abort iteration early, so that multiple errors can be reported. We only abort early on
    // parse failures we can't recover from.
    let mut guar = None;
    let mut check_emission = |ret: Result<(), ErrorGuaranteed>| guar = guar.or(ret.err());

    let mut kinds = MacroKinds::empty();
    let mut rules = Vec::new();

    while p.token != token::Eof {
        let unsafe_rule = p.eat_keyword_noexpect(kw::Unsafe);
        let unsafe_keyword_span = p.prev_token.span;
        if unsafe_rule && let Some(guar) = check_no_eof(sess, &p, "expected `attr`") {
            return dummy_syn_ext(guar);
        }
        let (args, is_derive) = if p.eat_keyword_noexpect(sym::attr) {
            kinds |= MacroKinds::ATTR;
            if !features.macro_attr() {
                feature_err(sess, sym::macro_attr, span, "`macro_rules!` attributes are unstable")
                    .emit();
            }
            if let Some(guar) = check_no_eof(sess, &p, "expected macro attr args") {
                return dummy_syn_ext(guar);
            }
            let args = p.parse_token_tree();
            check_args_parens(sess, sym::attr, &args);
            let args = parse_one_tt(args, RulePart::Pattern, sess, node_id, features, edition);
            check_emission(check_lhs(sess, node_id, &args));
            if let Some(guar) = check_no_eof(sess, &p, "expected macro attr body") {
                return dummy_syn_ext(guar);
            }
            (Some(args), false)
        } else if p.eat_keyword_noexpect(sym::derive) {
            kinds |= MacroKinds::DERIVE;
            let derive_keyword_span = p.prev_token.span;
            if !features.macro_derive() {
                feature_err(sess, sym::macro_derive, span, "`macro_rules!` derives are unstable")
                    .emit();
            }
            if unsafe_rule {
                sess.dcx()
                    .span_err(unsafe_keyword_span, "`unsafe` is only supported on `attr` rules");
            }
            if let Some(guar) = check_no_eof(sess, &p, "expected `()` after `derive`") {
                return dummy_syn_ext(guar);
            }
            let args = p.parse_token_tree();
            check_args_parens(sess, sym::derive, &args);
            let args_empty_result = check_args_empty(sess, &args);
            let args_not_empty = args_empty_result.is_err();
            check_emission(args_empty_result);
            if let Some(guar) = check_no_eof(sess, &p, "expected macro derive body") {
                return dummy_syn_ext(guar);
            }
            // If the user has `=>` right after the `()`, they might have forgotten the empty
            // parentheses.
            if p.token == token::FatArrow {
                let mut err = sess
                    .dcx()
                    .struct_span_err(p.token.span, "expected macro derive body, got `=>`");
                if args_not_empty {
                    err.span_label(derive_keyword_span, "need `()` after this `derive`");
                }
                return dummy_syn_ext(err.emit());
            }
            (None, true)
        } else {
            kinds |= MacroKinds::BANG;
            if unsafe_rule {
                sess.dcx()
                    .span_err(unsafe_keyword_span, "`unsafe` is only supported on `attr` rules");
            }
            (None, false)
        };
        let lhs_tt = p.parse_token_tree();
        let lhs_tt = parse_one_tt(lhs_tt, RulePart::Pattern, sess, node_id, features, edition);
        check_emission(check_lhs(sess, node_id, &lhs_tt));
        if let Err(e) = p.expect(exp!(FatArrow)) {
            return dummy_syn_ext(e.emit());
        }
        if let Some(guar) = check_no_eof(sess, &p, "expected right-hand side of macro rule") {
            return dummy_syn_ext(guar);
        }
        let rhs = p.parse_token_tree();
        let rhs = parse_one_tt(rhs, RulePart::Body, sess, node_id, features, edition);
        check_emission(check_rhs(sess, &rhs));
        check_emission(check_meta_variables(&sess.psess, node_id, args.as_ref(), &lhs_tt, &rhs));
        let lhs_span = lhs_tt.span();
        // Convert the lhs into `MatcherLoc` form, which is better for doing the
        // actual matching.
        let lhs = if let mbe::TokenTree::Delimited(.., delimited) = lhs_tt {
            mbe::macro_parser::compute_locs(&delimited.tts)
        } else {
            return dummy_syn_ext(guar.unwrap());
        };
        if let Some(args) = args {
            let args_span = args.span();
            let mbe::TokenTree::Delimited(.., delimited) = args else {
                return dummy_syn_ext(guar.unwrap());
            };
            let args = mbe::macro_parser::compute_locs(&delimited.tts);
            let body_span = lhs_span;
            rules.push(MacroRule::Attr { unsafe_rule, args, args_span, body: lhs, body_span, rhs });
        } else if is_derive {
            rules.push(MacroRule::Derive { body: lhs, body_span: lhs_span, rhs });
        } else {
            rules.push(MacroRule::Func { lhs, lhs_span, rhs });
        }
        if p.token == token::Eof {
            break;
        }
        if let Err(e) = p.expect(exp_sep) {
            return dummy_syn_ext(e.emit());
        }
    }

    if rules.is_empty() {
        let guar = sess.dcx().span_err(span, "macros must contain at least one rule");
        return dummy_syn_ext(guar);
    }
    assert!(!kinds.is_empty());

    let transparency = find_attr!(attrs, AttributeKind::MacroTransparency(x) => *x)
        .unwrap_or(Transparency::fallback(macro_rules));

    if let Some(guar) = guar {
        // To avoid warning noise, only consider the rules of this
        // macro for the lint, if all rules are valid.
        return dummy_syn_ext(guar);
    }

    // Return the number of rules for unused rule linting, if this is a local macro.
    let nrules = if is_defined_in_current_crate(node_id) { rules.len() } else { 0 };

    let exp = MacroRulesMacroExpander { name: ident, kinds, span, node_id, transparency, rules };
    (mk_syn_ext(SyntaxExtensionKind::MacroRules(Arc::new(exp))), nrules)
}

fn check_no_eof(sess: &Session, p: &Parser<'_>, msg: &'static str) -> Option<ErrorGuaranteed> {
    if p.token == token::Eof {
        let err_sp = p.token.span.shrink_to_hi();
        let guar = sess
            .dcx()
            .struct_span_err(err_sp, "macro definition ended unexpectedly")
            .with_span_label(err_sp, msg)
            .emit();
        return Some(guar);
    }
    None
}

fn check_args_parens(sess: &Session, rule_kw: Symbol, args: &tokenstream::TokenTree) {
    // This does not handle the non-delimited case; that gets handled separately by `check_lhs`.
    if let tokenstream::TokenTree::Delimited(dspan, _, delim, _) = args
        && *delim != Delimiter::Parenthesis
    {
        sess.dcx().emit_err(errors::MacroArgsBadDelim {
            span: dspan.entire(),
            sugg: errors::MacroArgsBadDelimSugg { open: dspan.open, close: dspan.close },
            rule_kw,
        });
    }
}

fn check_args_empty(sess: &Session, args: &tokenstream::TokenTree) -> Result<(), ErrorGuaranteed> {
    match args {
        tokenstream::TokenTree::Delimited(.., delimited) if delimited.is_empty() => Ok(()),
        _ => {
            let msg = "`derive` rules do not accept arguments; `derive` must be followed by `()`";
            Err(sess.dcx().span_err(args.span(), msg))
        }
    }
}

fn check_lhs(sess: &Session, node_id: NodeId, lhs: &mbe::TokenTree) -> Result<(), ErrorGuaranteed> {
    let e1 = check_lhs_nt_follows(sess, node_id, lhs);
    let e2 = check_lhs_no_empty_seq(sess, slice::from_ref(lhs));
    e1.and(e2)
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
                mbe::TokenTree::MetaVarDecl { kind: NonterminalKind::Vis, .. } => {}
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
    if seq.kleene.op == KleeneOp::ZeroOrOne
        && matches!(
            seq.tts.first(),
            Some(mbe::TokenTree::MetaVarDecl { kind: NonterminalKind::Vis, .. })
        )
    {
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
            | TokenTree::MetaVarDecl { .. }
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
                    | TokenTree::MetaVarDecl { .. }
                    | TokenTree::MetaVarExpr(..) => {
                        first.replace_with(TtHandle::TtRef(tt));
                    }
                    TokenTree::Delimited(span, _, delimited) => {
                        build_recur(sets, &delimited.tts);
                        first.replace_with(TtHandle::from_token_kind(
                            delimited.delim.as_open_token_kind(),
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
                | TokenTree::MetaVarDecl { .. }
                | TokenTree::MetaVarExpr(..) => {
                    first.add_one(TtHandle::TtRef(tt));
                    return first;
                }
                TokenTree::Delimited(span, _, delimited) => {
                    first.add_one(TtHandle::from_token_kind(
                        delimited.delim.as_open_token_kind(),
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
            | TokenTree::MetaVarDecl { .. }
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
                    d.delim.as_close_token_kind(),
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
            if let &TokenTree::MetaVarDecl { span, name, kind } = tt.get() {
                for next_token in &suffix_first.tokens {
                    let next_token = next_token.get();

                    // Check if the old pat is used and the next token is `|`
                    // to warn about incompatibility with Rust 2021.
                    // We only emit this lint if we're parsing the original
                    // definition of this macro_rules, not while (re)parsing
                    // the macro when compiling another crate that is using the
                    // macro. (See #86567.)
                    if is_defined_in_current_crate(node_id)
                        && matches!(kind, NonterminalKind::Pat(PatParam { inferred: true }))
                        && matches!(
                            next_token,
                            TokenTree::Token(token) if *token == token::Or
                        )
                    {
                        // It is suggestion to use pat_param, for example: $x:pat -> $x:pat_param.
                        let suggestion = quoted_tt_to_string(&TokenTree::MetaVarDecl {
                            span,
                            name,
                            kind: NonterminalKind::Pat(PatParam { inferred: false }),
                        });
                        sess.psess.buffer_lint(
                            RUST_2021_INCOMPATIBLE_OR_PATTERNS,
                            span,
                            ast::CRATE_NODE_ID,
                            errors::OrPatternsBackCompat { span, suggestion },
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
                                let suggestion = quoted_tt_to_string(&TokenTree::MetaVarDecl {
                                    span,
                                    name,
                                    kind: NonterminalKind::Pat(PatParam { inferred: false }),
                                });
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
    if let mbe::TokenTree::MetaVarDecl { kind, .. } = *tok {
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

    if let TokenTree::Token(Token { kind, .. }) = tok
        && kind.close_delim().is_some()
    {
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
                        OpenBrace | OpenBracket | Comma | FatArrow | Colon | Eq | Gt | Shr
                        | Semi | Or => IsInFollow::Yes,
                        Ident(name, IdentIsRaw::No) if name == kw::As || name == kw::Where => {
                            IsInFollow::Yes
                        }
                        _ => IsInFollow::No(TOKENS),
                    },
                    TokenTree::MetaVarDecl { kind: NonterminalKind::Block, .. } => IsInFollow::Yes,
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
                    TokenTree::MetaVarDecl {
                        kind: NonterminalKind::Ident | NonterminalKind::Ty | NonterminalKind::Path,
                        ..
                    } => IsInFollow::Yes,
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
        mbe::TokenTree::MetaVarDecl { name, kind, .. } => format!("${name}:{kind}"),
        _ => panic!(
            "{}",
            "unexpected mbe::TokenTree::{Sequence or Delimited} \
             in follow set checker"
        ),
    }
}

fn is_defined_in_current_crate(node_id: NodeId) -> bool {
    // Macros defined in the current crate have a real node id,
    // whereas macros from an external crate have a dummy id.
    node_id != DUMMY_NODE_ID
}

pub(super) fn parser_from_cx(
    psess: &ParseSess,
    mut tts: TokenStream,
    recovery: Recovery,
) -> Parser<'_> {
    tts.desugar_doc_comments();
    Parser::new(psess, tts, rustc_parse::MACRO_ARGUMENTS).recovery(recovery)
}
