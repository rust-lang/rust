use std::borrow::Cow;

use rustc_ast::token::{self, Token};
use rustc_ast::tokenstream::TokenStream;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_errors::{Applicability, Diag, DiagCtxtHandle, DiagMessage, pluralize};
use rustc_hir::attrs::diagnostic::{CustomDiagnostic, Directive, FormatArgs};
use rustc_macros::Subdiagnostic;
use rustc_middle::bug;
use rustc_parse::parser::{Parser, Recovery, token_descr};
use rustc_session::parse::ParseSess;
use rustc_span::source_map::SourceMap;
use rustc_span::{DUMMY_SP, ErrorGuaranteed, Ident, Span};
use tracing::debug;

use super::macro_rules::{MacroRule, NoopTracker, parser_from_cx};
use crate::expand::{AstFragmentKind, parse_ast_fragment};
use crate::mbe::macro_parser::ParseResult::*;
use crate::mbe::macro_parser::{MatcherLoc, NamedParseResult, TtParser};
use crate::mbe::macro_rules::{
    Tracker, WhichMatcher, try_match_macro, try_match_macro_attr, try_match_macro_derive,
};

pub(super) enum FailedMacro<'a> {
    Func,
    Attr(&'a TokenStream),
    Derive,
}

pub(super) fn failed_to_match_macro(
    psess: &ParseSess,
    sp: Span,
    def_span: Span,
    name: Ident,
    args: FailedMacro<'_>,
    body: &TokenStream,
    rules: &[MacroRule],
    on_unmatched_args: Option<&Directive>,
) -> (Span, ErrorGuaranteed) {
    debug!("failed to match macro");
    let def_head_span = if !def_span.is_dummy() && !psess.source_map().is_imported(def_span) {
        psess.source_map().guess_head_span(def_span)
    } else {
        DUMMY_SP
    };

    // An error occurred, try the expansion again, tracking the expansion closely for better
    // diagnostics.
    let mut tracker = CollectTrackerAndEmitter::new(name, psess.dcx(), sp);

    let try_success_result = match args {
        FailedMacro::Func => try_match_macro(psess, name, body, rules, &mut tracker),
        FailedMacro::Attr(attr_args) => {
            try_match_macro_attr(psess, name, attr_args, body, rules, &mut tracker)
        }
        FailedMacro::Derive => try_match_macro_derive(psess, name, body, rules, &mut tracker),
    };

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
    let CustomDiagnostic {
        message: custom_message, label: custom_label, notes: custom_notes, ..
    } = {
        on_unmatched_args
            .map(|directive| directive.eval(None, &FormatArgs { this: name.to_string(), .. }))
            .unwrap_or_default()
    };

    let mut err = match custom_message {
        Some(message) => psess.dcx().struct_span_err(span, message),
        None => psess.dcx().struct_span_err(span, parse_failure_msg(&token, None)),
    };
    err.span_label(span, custom_label.unwrap_or_else(|| label.to_string()));
    if !def_head_span.is_dummy() {
        err.span_label(def_head_span, "when calling this macro");
    }

    annotate_doc_comment(&mut err, psess.source_map(), span);

    if let Some(span) = remaining_matcher.span() {
        err.span_note(span, format!("while trying to match {remaining_matcher}"));
    } else {
        err.note(format!("while trying to match {remaining_matcher}"));
    }
    for note in custom_notes {
        err.note(note);
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
    if let FailedMacro::Func = args
        && let Some((body, comma_span)) = body.add_comma()
    {
        for rule in rules {
            let MacroRule::Func { lhs, .. } = rule else { continue };
            let parser = parser_from_cx(psess, body.clone(), Recovery::Allowed);
            let mut tt_parser = TtParser::new();

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
    macro_name: Ident,
    dcx: DiagCtxtHandle<'dcx>,

    /// The matcher currently being parsed.
    //
    // FIXME: Factor out a per-arm `Tracker` so that the `Option` is unnecessary.
    current: Option<(WhichMatcher, &'matcher [MatcherLoc])>,

    /// Matches of [`MatcherLoc`]s that successfully consumed input from the parser.
    ///
    /// This accumulates all calls to [`Tracker::matched_one()`]. It is used to identify all
    /// competing matches for ambiguity errors.
    matches: FxHashSet<SuccessfulMatch>,

    /// Tokens seen during parsing.
    tokens: FxHashMap<u32, Token>,

    remaining_matcher: Option<&'matcher MatcherLoc>,
    /// Which arm's failure should we report? (the one furthest along)
    best_failure: Option<BestFailure>,
    root_span: Span,
    result: Option<(Span, ErrorGuaranteed)>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct SuccessfulMatch {
    /// The position in the parser.
    ///
    /// As per [`Parser::approx_token_stream_pos()`].
    input_pos: u32,

    /// The index of the [`MatcherLoc`].
    loc_index: u32,
}

struct BestFailure {
    token: Token,

    /// The matcher in which the failure occurred.
    matcher: WhichMatcher,

    /// The approximate (parser) position of the failure.
    ///
    /// This is relative to [`Self::matcher`].
    position: u32,

    msg: &'static str,
    remaining_matcher: MatcherLoc,
}

impl BestFailure {
    fn is_better_position(&self, matcher: WhichMatcher, position: u32) -> bool {
        (matcher, position) > (self.matcher, self.position)
    }
}

impl<'dcx, 'matcher> Tracker<'matcher> for CollectTrackerAndEmitter<'dcx, 'matcher> {
    fn prepare(&mut self, which_matcher: WhichMatcher, matcher: &'matcher [MatcherLoc]) {
        if self.current.is_some() {
            bug!("`Self::after_arm()` was not called to clean up context");
        }

        self.current = Some((which_matcher, matcher));
    }

    fn trying_match(&mut self, input_pos: u32, token: &Token, loc_index: usize) {
        let Some((_, matcher)) = self.current else {
            bug!("`Self::prepare()` was not called to initialize context");
        };
        let matcher = &matcher[loc_index];

        let old_token = self.tokens.insert(input_pos, *token);
        debug_assert!(old_token.is_none_or(|t| t == *token));

        if self.remaining_matcher.is_none() || *matcher != MatcherLoc::Eof {
            self.remaining_matcher = Some(matcher);
        }
    }

    fn matched_one(&mut self, input_pos: u32, loc_index: usize) {
        let loc_index: u32 = loc_index.try_into().unwrap();
        let m = SuccessfulMatch { input_pos, loc_index };
        self.matches.insert(m);
    }

    fn after_arm(&mut self, result: &NamedParseResult) {
        match *result {
            Success(_) => {
                // Nonterminal parser recovery might turn failed matches into successful ones,
                // but for that it must have emitted an error already
                self.dcx.span_delayed_bug(
                    self.root_span,
                    "should not collect detailed info for successful macro match",
                );
            }
            Failure => {
                if self.best_failure.is_none() {
                    bug!("A matching failure occurred but `Self::failure()` was not called");
                }
            }
            Ambiguity => {
                if self.result.is_none() {
                    bug!("An ambiguity error occurred but `Self::ambiguity()` was not called");
                }
            }
            ErrorReported(guar) => self.result = Some((self.root_span, guar)),
        }

        self.current = None;
        self.matches.clear();
        self.tokens.clear();
    }

    fn failure(&mut self, parser: &Parser<'_>) {
        let Some((which_matcher, _)) = self.current else {
            bug!("`Self::prepare()` was not called to initialize context");
        };

        let mut token = parser.token;
        let approx_position = parser.approx_token_stream_pos();
        let msg = if token.kind == token::Eof {
            // FIXME: Can this be factored out of the EOF case?
            if !token.span.is_dummy() {
                token.span = token.span.shrink_to_hi();
            }
            "missing tokens in macro arguments"
        } else {
            "no rules expected this token in macro call"
        };

        debug!(?token, ?msg, "a new failure of an arm");

        if self
            .best_failure
            .as_ref()
            .is_none_or(|failure| failure.is_better_position(which_matcher, approx_position))
        {
            self.best_failure = Some(BestFailure {
                token,
                matcher: which_matcher,
                position: approx_position,
                msg,
                remaining_matcher: self
                    .remaining_matcher
                    .expect("must have collected matcher already")
                    .clone(),
            })
        }
    }

    fn ambiguity(&mut self) {
        let Some((_, matcher)) = self.current else {
            bug!("`Self::prepare()` was not called to initialize context");
        };

        #[expect(
            rustc::potential_query_instability,
            reason = "sorting the results deterministically afterwards"
        )]
        let mut matches = self.matches.iter().collect::<Vec<_>>();
        // Sort by input position, then `MatcherLoc` index.
        matches.sort_unstable();

        // Identify the earliest position where ambiguity occurred.
        let input_pos = matches
            .array_windows::<2>()
            .find(|ms @ [a, b]| {
                let mut locs = ms.iter().map(|x| &matcher[x.loc_index as usize]);
                a.input_pos == b.input_pos
                    && locs
                        .any(|loc| matches!(loc, MatcherLoc::MetaVarDecl { .. } | MatcherLoc::Eof))
            })
            .map(|[a, _]| a.input_pos)
            .unwrap_or_else(|| bug!("no ambiguity detected"));

        let (bb_locs, next_locs) = matches
            .iter()
            .filter(|m| m.input_pos == input_pos)
            .partition::<Vec<&SuccessfulMatch>, _>(|m| {
                let loc = &matcher[m.loc_index as usize];
                matches!(loc, MatcherLoc::MetaVarDecl { .. })
            });

        debug_assert!(bb_locs.iter().is_sorted());
        debug_assert!(next_locs.iter().is_sorted());

        let token = *self.tokens.get(&input_pos).unwrap();

        let span = token.span.substitute_dummy(self.root_span);

        if token == token::Eof {
            let msg = "ambiguity: multiple successful parses".to_string();
            let guar = self.dcx.span_err(span, msg);
            self.result = Some((span, guar));
            return;
        }

        let nts = bb_locs
            .into_iter()
            .map(|m| {
                let loc = &matcher[m.loc_index as usize];
                let MatcherLoc::MetaVarDecl { bind, kind, .. } = loc else { unreachable!() };
                format!("{kind} ('{bind}')")
            })
            .collect::<Vec<String>>()
            .join(" or ");

        let msg = format!(
            "local ambiguity when calling macro `{}`: multiple parsing options: {}",
            self.macro_name,
            match next_locs.len() {
                0 => format!("built-in NTs {nts}."),
                n => format!("built-in NTs {nts} or {n} other option{s}.", s = pluralize!(n)),
            }
        );

        let guar = self.dcx.span_err(span, msg);
        self.result = Some((span, guar));
    }

    fn description() -> &'static str {
        "detailed"
    }

    fn recovery() -> Recovery {
        Recovery::Allowed
    }
}

impl<'dcx> CollectTrackerAndEmitter<'dcx, '_> {
    fn new(macro_name: Ident, dcx: DiagCtxtHandle<'dcx>, root_span: Span) -> Self {
        Self {
            macro_name,
            dcx,
            current: None,
            matches: FxHashSet::default(),
            tokens: FxHashMap::default(),
            remaining_matcher: None,
            best_failure: None,
            root_span,
            result: None,
        }
    }
}

pub(super) fn emit_frag_parse_err(
    mut e: Diag<'_>,
    parser: &mut Parser<'_>,
    orig_parser: &mut Parser<'_>,
    site_span: Span,
    arm_span: Span,
    kind: AstFragmentKind,
    bindings: &[MacroRule],
    matched_rule_bindings: &[MatcherLoc],
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

    if parser.token.kind == token::Dollar {
        let dollar_span = parser.token.span;
        parser.bump();
        if let token::Ident(name, _) = parser.token.kind {
            let metavar_span = dollar_span.to(parser.token.span);
            let mut bindings_names = vec![];
            for rule in bindings {
                let MacroRule::Func { lhs, .. } = rule else { continue };
                for param in lhs {
                    let MatcherLoc::MetaVarDecl { bind, .. } = param else { continue };
                    bindings_names.push(bind.name);
                }
            }

            let mut matched_rule_bindings_names = vec![];
            for param in matched_rule_bindings {
                let MatcherLoc::MetaVarDecl { bind, .. } = param else { continue };
                matched_rule_bindings_names.push(bind.name);
            }

            // Report the unbound metavariable as the primary error up front, so every
            // case is consistent regardless of which suggestion (if any) we attach below.
            e.primary_message(format!("cannot find macro parameter `${name}` in this scope"));
            e.span(metavar_span);
            e.span_label(metavar_span, "not found in this scope");
            if parser.psess.source_map().is_imported(metavar_span) {
                e.span_label(site_span, "in this macro invocation");
            }

            if let Some(matched_name) = rustc_span::edit_distance::find_best_match_for_name(
                &matched_rule_bindings_names[..],
                name,
                None,
            ) {
                e.span_suggestion_verbose(
                    parser.token.span,
                    "there is a macro metavariable with a similar name",
                    matched_name,
                    Applicability::MaybeIncorrect,
                );
            } else if bindings_names.contains(&name) {
                e.span_label(
                    parser.token.span,
                    "there is an macro metavariable with this name in another macro matcher",
                );
            } else if let Some(matched_name) =
                rustc_span::edit_distance::find_best_match_for_name(&bindings_names[..], name, None)
            {
                e.span_suggestion_verbose(
                    parser.token.span,
                    "there is a macro metavariable with a similar name in another macro matcher",
                    matched_name,
                    Applicability::MaybeIncorrect,
                );
            } else if !matched_rule_bindings_names.is_empty() {
                let msg = matched_rule_bindings_names
                    .iter()
                    .map(|sym| format!("${}", sym))
                    .collect::<Vec<_>>()
                    .join(", ");
                e.note(format!("available metavariable names are: {msg}"));
            }
        }
    }
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
    #[label(
        "inner doc comments expand to `#![doc = \"...\"]`, which is what this macro attempted to match"
    )]
    Inner {
        #[primary_span]
        span: Span,
    },
    #[label(
        "outer doc comments expand to `#[doc = \"...\"]`, which is what this macro attempted to match"
    )]
    Outer {
        #[primary_span]
        span: Span,
    },
}

fn annotate_doc_comment(err: &mut Diag<'_>, sm: &SourceMap, span: Span) {
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
fn parse_failure_msg(tok: &Token, expected_token: Option<&Token>) -> Cow<'static, str> {
    if let Some(expected_token) = expected_token {
        Cow::from(format!("expected {}, found {}", token_descr(expected_token), token_descr(tok)))
    } else {
        match tok.kind {
            token::Eof => Cow::from("unexpected end of macro invocation"),
            _ => Cow::from(format!("no rules expected {}", token_descr(tok))),
        }
    }
}
