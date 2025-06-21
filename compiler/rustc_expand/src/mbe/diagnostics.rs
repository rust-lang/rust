use std::borrow::Cow;

use rustc_ast::token::{self, Token};
use rustc_ast::tokenstream::TokenStream;
use rustc_errors::{Applicability, Diag, DiagCtxtHandle, DiagMessage};
use rustc_macros::Subdiagnostic;
use rustc_parse::parser::{Parser, Recovery, token_descr};
use rustc_session::parse::ParseSess;
use rustc_span::source_map::SourceMap;
use rustc_span::{ErrorGuaranteed, Ident, Span};
use tracing::debug;

use super::macro_rules::{NoopTracker, parser_from_cx};
use crate::expand::{AstFragmentKind, parse_ast_fragment};
use crate::mbe::macro_parser::ParseResult::*;
use crate::mbe::macro_parser::{MatcherLoc, NamedParseResult, TtParser};
use crate::mbe::macro_rules::{Tracker, try_match_macro};

pub(super) fn failed_to_match_macro(
    psess: &ParseSess,
    sp: Span,
    def_span: Span,
    name: Ident,
    arg: TokenStream,
    lhses: &[Vec<MatcherLoc>],
) -> (Span, ErrorGuaranteed) {
    debug!("failed to match macro");
    // An error occurred, try the expansion again, tracking the expansion closely for better
    // diagnostics.
    let mut tracker = CollectTrackerAndEmitter::new(psess.dcx(), sp);

    let try_success_result = try_match_macro(psess, name, &arg, lhses, &mut tracker);

    if try_success_result.is_ok() {
        // Nonterminal parser recovery might turn failed matches into successful ones,
        // but for that it must have emitted an error already
        assert!(
            tracker.dcx.has_errors().is_some(),
            "Macro matching returned a success on the second try"
        );
    }

    if let Some(result) = tracker.result {
        // An irrecoverable error occurred and has been emitted.
        return result;
    }

    let Some(BestFailure { token, msg: label, remaining_matcher, .. }) = tracker.best_failure
    else {
        return (sp, psess.dcx().span_delayed_bug(sp, "failed to match a macro"));
    };

    let span = token.span.substitute_dummy(sp);

    let mut err = psess.dcx().struct_span_err(span, parse_failure_msg(&token, None));
    err.span_label(span, label);
    if !def_span.is_dummy() && !psess.source_map().is_imported(def_span) {
        err.span_label(psess.source_map().guess_head_span(def_span), "when calling this macro");
    }

    annotate_doc_comment(&mut err, psess.source_map(), span);

    if let Some(span) = remaining_matcher.span() {
        err.span_note(span, format!("while trying to match {remaining_matcher}"));
    } else {
        err.note(format!("while trying to match {remaining_matcher}"));
    }

    if let MatcherLoc::Token { token: expected_token } = &remaining_matcher
        && (matches!(expected_token.kind, token::OpenInvisible(_))
            || matches!(token.kind, token::OpenInvisible(_)))
    {
        err.note("captured metavariables except for `:tt`, `:ident` and `:lifetime` cannot be compared to other tokens");
        err.note("see <https://doc.rust-lang.org/nightly/reference/macros-by-example.html#forwarding-a-matched-fragment> for more information");

        if !def_span.is_dummy() && !psess.source_map().is_imported(def_span) {
            err.help("try using `:tt` instead in the macro definition");
        }
    }

    // Check whether there's a missing comma in this macro call, like `println!("{}" a);`
    if let Some((arg, comma_span)) = arg.add_comma() {
        for lhs in lhses {
            let parser = parser_from_cx(psess, arg.clone(), Recovery::Allowed);
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
    let guar = err.emit();
    (sp, guar)
}

/// The tracker used for the slow error path that collects useful info for diagnostics.
struct CollectTrackerAndEmitter<'dcx, 'matcher> {
    dcx: DiagCtxtHandle<'dcx>,
    remaining_matcher: Option<&'matcher MatcherLoc>,
    /// Which arm's failure should we report? (the one furthest along)
    best_failure: Option<BestFailure>,
    root_span: Span,
    result: Option<(Span, ErrorGuaranteed)>,
}

struct BestFailure {
    token: Token,
    position_in_tokenstream: u32,
    msg: &'static str,
    remaining_matcher: MatcherLoc,
}

impl BestFailure {
    fn is_better_position(&self, position: u32) -> bool {
        position > self.position_in_tokenstream
    }
}

impl<'dcx, 'matcher> Tracker<'matcher> for CollectTrackerAndEmitter<'dcx, 'matcher> {
    type Failure = (Token, u32, &'static str);

    fn build_failure(tok: Token, position: u32, msg: &'static str) -> Self::Failure {
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
                self.dcx.span_delayed_bug(
                    self.root_span,
                    "should not collect detailed info for successful macro match",
                );
            }
            Failure((token, approx_position, msg)) => {
                debug!(?token, ?msg, "a new failure of an arm");

                if self
                    .best_failure
                    .as_ref()
                    .is_none_or(|failure| failure.is_better_position(*approx_position))
                {
                    self.best_failure = Some(BestFailure {
                        token: *token,
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
                let guar = self.dcx.span_err(span, msg.clone());
                self.result = Some((span, guar));
            }
            ErrorReported(guar) => self.result = Some((self.root_span, *guar)),
        }
    }

    fn description() -> &'static str {
        "detailed"
    }

    fn recovery() -> Recovery {
        Recovery::Allowed
    }
}

impl<'dcx> CollectTrackerAndEmitter<'dcx, '_> {
    fn new(dcx: DiagCtxtHandle<'dcx>, root_span: Span) -> Self {
        Self { dcx, remaining_matcher: None, best_failure: None, root_span, result: None }
    }
}

/// Currently used by macro_rules! compilation to extract a little information from the `Failure`
/// case.
pub(crate) struct FailureForwarder<'matcher> {
    expected_token: Option<&'matcher Token>,
}

impl<'matcher> FailureForwarder<'matcher> {
    pub(crate) fn new() -> Self {
        Self { expected_token: None }
    }
}

impl<'matcher> Tracker<'matcher> for FailureForwarder<'matcher> {
    type Failure = (Token, u32, &'static str);

    fn build_failure(tok: Token, position: u32, msg: &'static str) -> Self::Failure {
        (tok, position, msg)
    }

    fn description() -> &'static str {
        "failure-forwarder"
    }

    fn set_expected_token(&mut self, tok: &'matcher Token) {
        self.expected_token = Some(tok);
    }

    fn get_expected_token(&self) -> Option<&'matcher Token> {
        self.expected_token
    }
}

pub(super) fn emit_frag_parse_err(
    mut e: Diag<'_>,
    parser: &Parser<'_>,
    orig_parser: &mut Parser<'_>,
    site_span: Span,
    arm_span: Span,
    kind: AstFragmentKind,
) -> ErrorGuaranteed {
    // FIXME(davidtwco): avoid depending on the error message text
    if parser.token == token::Eof
        && let DiagMessage::Str(message) = &e.messages[0].0
        && message.ends_with(", found `<eof>`")
    {
        let msg = &e.messages[0];
        e.messages[0] = (
            DiagMessage::from(format!(
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
        if !parser.psess.source_map().is_imported(arm_span) {
            e.span_label(arm_span, "in this macro arm");
        }
    } else if parser.psess.source_map().is_imported(parser.token.span) {
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

                if parser.token == token::Semi {
                    if let Ok(snippet) = parser.psess.source_map().span_to_snippet(site_span) {
                        e.span_suggestion_verbose(
                            site_span,
                            "surround the macro invocation with `{}` to interpret the expansion as a statement",
                            format!("{{ {snippet}; }}"),
                            Applicability::MaybeIncorrect,
                        );
                    }
                } else {
                    e.span_suggestion_verbose(
                        site_span.shrink_to_hi(),
                        "add `;` to interpret the expansion as a statement",
                        ";",
                        Applicability::MaybeIncorrect,
                    );
                }
            }
        },
        _ => annotate_err_with_kind(&mut e, kind, site_span),
    };
    e.emit()
}

pub(crate) fn annotate_err_with_kind(err: &mut Diag<'_>, kind: AstFragmentKind, span: Span) {
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

pub(super) fn annotate_doc_comment(err: &mut Diag<'_>, sm: &SourceMap, span: Span) {
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
pub(super) fn parse_failure_msg(tok: &Token, expected_token: Option<&Token>) -> Cow<'static, str> {
    if let Some(expected_token) = expected_token {
        Cow::from(format!("expected {}, found {}", token_descr(expected_token), token_descr(tok)))
    } else {
        match tok.kind {
            token::Eof => Cow::from("unexpected end of macro invocation"),
            _ => Cow::from(format!("no rules expected {}", token_descr(tok))),
        }
    }
}
