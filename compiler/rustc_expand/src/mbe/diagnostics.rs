use crate::base::{DummyResult, ExtCtxt, MacResult};
use crate::expand::{parse_ast_fragment, AstFragmentKind};
use crate::mbe::{
    macro_parser::{MatcherLoc, NamedParseResult, ParseResult::*, TtParser},
    macro_rules::{try_match_macro, Tracker},
};
use rustc_ast::token::{self, Token, TokenKind};
use rustc_ast::tokenstream::TokenStream;
use rustc_ast_pretty::pprust;
use rustc_errors::{Applicability, Diagnostic, DiagnosticBuilder, DiagnosticMessage};
use rustc_parse::parser::{Parser, Recovery};
use rustc_span::source_map::SourceMap;
use rustc_span::symbol::Ident;
use rustc_span::Span;
use std::borrow::Cow;

use super::macro_rules::{parser_from_cx, NoopTracker};

pub(super) fn failed_to_match_macro<'cx>(
    cx: &'cx mut ExtCtxt<'_>,
    sp: Span,
    def_span: Span,
    name: Ident,
    arg: TokenStream,
    lhses: &[Vec<MatcherLoc>],
) -> Box<dyn MacResult + 'cx> {
    let sess = &cx.sess.parse_sess;

    // An error occurred, try the expansion again, tracking the expansion closely for better diagnostics.
    let mut tracker = CollectTrackerAndEmitter::new(cx, sp);

    let try_success_result = try_match_macro(sess, name, &arg, lhses, &mut tracker);

    if try_success_result.is_ok() {
        // Nonterminal parser recovery might turn failed matches into successful ones,
        // but for that it must have emitted an error already
        tracker.cx.sess.delay_span_bug(sp, "Macro matching returned a success on the second try");
    }

    if let Some(result) = tracker.result {
        // An irrecoverable error occurred and has been emitted.
        return result;
    }

    let Some(BestFailure { token, msg: label, remaining_matcher, .. }) = tracker.best_failure else {
        return DummyResult::any(sp);
    };

    let span = token.span.substitute_dummy(sp);

    let mut err = cx.struct_span_err(span, &parse_failure_msg(&token));
    err.span_label(span, label);
    if !def_span.is_dummy() && !cx.source_map().is_imported(def_span) {
        err.span_label(cx.source_map().guess_head_span(def_span), "when calling this macro");
    }

    annotate_doc_comment(&mut err, sess.source_map(), span);

    if let Some(span) = remaining_matcher.span() {
        err.span_note(span, format!("while trying to match {remaining_matcher}"));
    } else {
        err.note(format!("while trying to match {remaining_matcher}"));
    }

    if let MatcherLoc::Token { token: expected_token } = &remaining_matcher
        && (matches!(expected_token.kind, TokenKind::Interpolated(_))
            || matches!(token.kind, TokenKind::Interpolated(_)))
    {
        err.note("captured metavariables except for `$tt`, `$ident` and `$lifetime` cannot be compared to other tokens");
    }

    // Check whether there's a missing comma in this macro call, like `println!("{}" a);`
    if let Some((arg, comma_span)) = arg.add_comma() {
        for lhs in lhses {
            let parser = parser_from_cx(sess, arg.clone(), Recovery::Allowed);
            let mut tt_parser = TtParser::new(name);

            if let Success(_) =
                tt_parser.parse_tt(&mut Cow::Borrowed(&parser), lhs, &mut NoopTracker)
            {
                if comma_span.is_dummy() {
                    err.note("you might be missing a comma");
                } else {
                    err.span_suggestion_short(
                        comma_span,
                        "missing comma here",
                        ", ",
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

/// The tracker used for the slow error path that collects useful info for diagnostics.
struct CollectTrackerAndEmitter<'a, 'cx, 'matcher> {
    cx: &'a mut ExtCtxt<'cx>,
    remaining_matcher: Option<&'matcher MatcherLoc>,
    /// Which arm's failure should we report? (the one furthest along)
    best_failure: Option<BestFailure>,
    root_span: Span,
    result: Option<Box<dyn MacResult + 'cx>>,
}

struct BestFailure {
    token: Token,
    position_in_tokenstream: usize,
    msg: &'static str,
    remaining_matcher: MatcherLoc,
}

impl BestFailure {
    fn is_better_position(&self, position: usize) -> bool {
        position > self.position_in_tokenstream
    }
}

impl<'a, 'cx, 'matcher> Tracker<'matcher> for CollectTrackerAndEmitter<'a, 'cx, 'matcher> {
    type Failure = (Token, usize, &'static str);

    fn build_failure(tok: Token, position: usize, msg: &'static str) -> Self::Failure {
        (tok, position, msg)
    }

    fn before_match_loc(&mut self, parser: &TtParser, matcher: &'matcher MatcherLoc) {
        if self.remaining_matcher.is_none()
            || (parser.has_no_remaining_items_for_step() && *matcher != MatcherLoc::Eof)
        {
            self.remaining_matcher = Some(matcher);
        }
    }

    fn after_arm(&mut self, result: &NamedParseResult<Self::Failure>) {
        match result {
            Success(_) => {
                // Nonterminal parser recovery might turn failed matches into successful ones,
                // but for that it must have emitted an error already
                self.cx.sess.delay_span_bug(
                    self.root_span,
                    "should not collect detailed info for successful macro match",
                );
            }
            Failure((token, approx_position, msg)) => {
                debug!(?token, ?msg, "a new failure of an arm");

                if self
                    .best_failure
                    .as_ref()
                    .map_or(true, |failure| failure.is_better_position(*approx_position))
                {
                    self.best_failure = Some(BestFailure {
                        token: token.clone(),
                        position_in_tokenstream: *approx_position,
                        msg,
                        remaining_matcher: self
                            .remaining_matcher
                            .expect("must have collected matcher already")
                            .clone(),
                    })
                }
            }
            Error(err_sp, msg) => {
                let span = err_sp.substitute_dummy(self.root_span);
                self.cx.struct_span_err(span, msg).emit();
                self.result = Some(DummyResult::any(span));
            }
            ErrorReported(_) => self.result = Some(DummyResult::any(self.root_span)),
        }
    }

    fn description() -> &'static str {
        "detailed"
    }

    fn recovery() -> Recovery {
        Recovery::Allowed
    }
}

impl<'a, 'cx> CollectTrackerAndEmitter<'a, 'cx, '_> {
    fn new(cx: &'a mut ExtCtxt<'cx>, root_span: Span) -> Self {
        Self { cx, remaining_matcher: None, best_failure: None, root_span, result: None }
    }
}

/// Currently used by macro_rules! compilation to extract a little information from the `Failure` case.
pub struct FailureForwarder;

impl<'matcher> Tracker<'matcher> for FailureForwarder {
    type Failure = (Token, usize, &'static str);

    fn build_failure(tok: Token, position: usize, msg: &'static str) -> Self::Failure {
        (tok, position, msg)
    }

    fn description() -> &'static str {
        "failure-forwarder"
    }
}

pub(super) fn emit_frag_parse_err(
    mut e: DiagnosticBuilder<'_, rustc_errors::ErrorGuaranteed>,
    parser: &Parser<'_>,
    orig_parser: &mut Parser<'_>,
    site_span: Span,
    arm_span: Span,
    kind: AstFragmentKind,
) {
    // FIXME(davidtwco): avoid depending on the error message text
    if parser.token == token::Eof
        && let DiagnosticMessage::Str(message) = &e.message[0].0
        && message.ends_with(", found `<eof>`")
    {
        let msg = &e.message[0];
        e.message[0] = (
            DiagnosticMessage::Str(format!(
                "macro expansion ends with an incomplete expression: {}",
                message.replace(", found `<eof>`", ""),
            )),
            msg.1,
        );
        if !e.span.is_dummy() {
            // early end of macro arm (#52866)
            e.replace_span_with(parser.token.span.shrink_to_hi(), true);
        }
    }
    if e.span.is_dummy() {
        // Get around lack of span in error (#30128)
        e.replace_span_with(site_span, true);
        if !parser.sess.source_map().is_imported(arm_span) {
            e.span_label(arm_span, "in this macro arm");
        }
    } else if parser.sess.source_map().is_imported(parser.token.span) {
        e.span_label(site_span, "in this macro invocation");
    }
    match kind {
        // Try a statement if an expression is wanted but failed and suggest adding `;` to call.
        AstFragmentKind::Expr => match parse_ast_fragment(orig_parser, AstFragmentKind::Stmts) {
            Err(err) => err.cancel(),
            Ok(_) => {
                e.note(
                    "the macro call doesn't expand to an expression, but it can expand to a statement",
                );
                e.span_suggestion_verbose(
                    site_span.shrink_to_hi(),
                    "add `;` to interpret the expansion as a statement",
                    ";",
                    Applicability::MaybeIncorrect,
                );
            }
        },
        _ => annotate_err_with_kind(&mut e, kind, site_span),
    };
    e.emit();
}

pub(crate) fn annotate_err_with_kind(err: &mut Diagnostic, kind: AstFragmentKind, span: Span) {
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

#[derive(Subdiagnostic)]
enum ExplainDocComment {
    #[label(expand_explain_doc_comment_inner)]
    Inner {
        #[primary_span]
        span: Span,
    },
    #[label(expand_explain_doc_comment_outer)]
    Outer {
        #[primary_span]
        span: Span,
    },
}

pub(super) fn annotate_doc_comment(err: &mut Diagnostic, sm: &SourceMap, span: Span) {
    if let Ok(src) = sm.span_to_snippet(span) {
        if src.starts_with("///") || src.starts_with("/**") {
            err.subdiagnostic(ExplainDocComment::Outer { span });
        } else if src.starts_with("//!") || src.starts_with("/*!") {
            err.subdiagnostic(ExplainDocComment::Inner { span });
        }
    }
}

/// Generates an appropriate parsing failure message. For EOF, this is "unexpected end...". For
/// other tokens, this is "unexpected token...".
pub(super) fn parse_failure_msg(tok: &Token) -> String {
    match tok.kind {
        token::Eof => "unexpected end of macro invocation".to_string(),
        _ => format!("no rules expected the token `{}`", pprust::token_to_string(tok),),
    }
}
