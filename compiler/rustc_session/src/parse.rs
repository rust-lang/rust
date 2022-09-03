//! Contains `ParseSess` which holds state living beyond what one `Parser` might.
//! It also serves as an input to the parser itself.

use crate::config::CheckCfg;
use crate::errors::{FeatureDiagnosticForIssue, FeatureDiagnosticHelp, FeatureGateError};
use crate::lint::{
    builtin::UNSTABLE_SYNTAX_PRE_EXPANSION, BufferedEarlyLint, BuiltinLintDiagnostics, Lint, LintId,
};
use crate::SessionDiagnostic;
use rustc_ast::node_id::NodeId;
use rustc_data_structures::fx::{FxHashMap, FxHashSet, FxIndexSet};
use rustc_data_structures::sync::{Lock, Lrc};
use rustc_errors::{emitter::SilentEmitter, ColorConfig, Handler};
use rustc_errors::{
    fallback_fluent_bundle, Applicability, Diagnostic, DiagnosticBuilder, DiagnosticId,
    DiagnosticMessage, EmissionGuarantee, ErrorGuaranteed, MultiSpan, StashKey,
};
use rustc_feature::{find_feature_issue, GateIssue, UnstableFeatures};
use rustc_span::edition::Edition;
use rustc_span::hygiene::ExpnId;
use rustc_span::source_map::{FilePathMapping, SourceMap};
use rustc_span::{Span, Symbol};

use std::str;

/// The set of keys (and, optionally, values) that define the compilation
/// environment of the crate, used to drive conditional compilation.
pub type CrateConfig = FxIndexSet<(Symbol, Option<Symbol>)>;
pub type CrateCheckConfig = CheckCfg<Symbol>;

/// Collected spans during parsing for places where a certain feature was
/// used and should be feature gated accordingly in `check_crate`.
#[derive(Default)]
pub struct GatedSpans {
    pub spans: Lock<FxHashMap<Symbol, Vec<Span>>>,
}

impl GatedSpans {
    /// Feature gate the given `span` under the given `feature`
    /// which is same `Symbol` used in `active.rs`.
    pub fn gate(&self, feature: Symbol, span: Span) {
        self.spans.borrow_mut().entry(feature).or_default().push(span);
    }

    /// Ungate the last span under the given `feature`.
    /// Panics if the given `span` wasn't the last one.
    ///
    /// Using this is discouraged unless you have a really good reason to.
    pub fn ungate_last(&self, feature: Symbol, span: Span) {
        let removed_span = self.spans.borrow_mut().entry(feature).or_default().pop().unwrap();
        debug_assert_eq!(span, removed_span);
    }

    /// Is the provided `feature` gate ungated currently?
    ///
    /// Using this is discouraged unless you have a really good reason to.
    pub fn is_ungated(&self, feature: Symbol) -> bool {
        self.spans.borrow().get(&feature).map_or(true, |spans| spans.is_empty())
    }

    /// Prepend the given set of `spans` onto the set in `self`.
    pub fn merge(&self, mut spans: FxHashMap<Symbol, Vec<Span>>) {
        let mut inner = self.spans.borrow_mut();
        for (gate, mut gate_spans) in inner.drain() {
            spans.entry(gate).or_default().append(&mut gate_spans);
        }
        *inner = spans;
    }
}

#[derive(Default)]
pub struct SymbolGallery {
    /// All symbols occurred and their first occurrence span.
    pub symbols: Lock<FxHashMap<Symbol, Span>>,
}

impl SymbolGallery {
    /// Insert a symbol and its span into symbol gallery.
    /// If the symbol has occurred before, ignore the new occurrence.
    pub fn insert(&self, symbol: Symbol, span: Span) {
        self.symbols.lock().entry(symbol).or_insert(span);
    }
}

/// Construct a diagnostic for a language feature error due to the given `span`.
/// The `feature`'s `Symbol` is the one you used in `active.rs` and `rustc_span::symbols`.
pub fn feature_err<'a>(
    sess: &'a ParseSess,
    feature: Symbol,
    span: impl Into<MultiSpan>,
    explain: &str,
) -> DiagnosticBuilder<'a, ErrorGuaranteed> {
    feature_err_issue(sess, feature, span, GateIssue::Language, explain)
}

/// Construct a diagnostic for a feature gate error.
///
/// This variant allows you to control whether it is a library or language feature.
/// Almost always, you want to use this for a language feature. If so, prefer `feature_err`.
pub fn feature_err_issue<'a>(
    sess: &'a ParseSess,
    feature: Symbol,
    span: impl Into<MultiSpan>,
    issue: GateIssue,
    explain: &str,
) -> DiagnosticBuilder<'a, ErrorGuaranteed> {
    let span = span.into();

    // Cancel an earlier warning for this same error, if it exists.
    if let Some(span) = span.primary_span() {
        sess.span_diagnostic
            .steal_diagnostic(span, StashKey::EarlySyntaxWarning)
            .map(|err| err.cancel());
    }

    let mut err = sess.create_err(FeatureGateError { span, explain });
    add_feature_diagnostics_for_issue(&mut err, sess, feature, issue);
    err
}

/// Construct a future incompatibility diagnostic for a feature gate.
///
/// This diagnostic is only a warning and *does not cause compilation to fail*.
pub fn feature_warn<'a>(sess: &'a ParseSess, feature: Symbol, span: Span, explain: &str) {
    feature_warn_issue(sess, feature, span, GateIssue::Language, explain);
}

/// Construct a future incompatibility diagnostic for a feature gate.
///
/// This diagnostic is only a warning and *does not cause compilation to fail*.
///
/// This variant allows you to control whether it is a library or language feature.
/// Almost always, you want to use this for a language feature. If so, prefer `feature_warn`.
#[allow(rustc::diagnostic_outside_of_impl)]
#[allow(rustc::untranslatable_diagnostic)]
pub fn feature_warn_issue<'a>(
    sess: &'a ParseSess,
    feature: Symbol,
    span: Span,
    issue: GateIssue,
    explain: &str,
) {
    let mut err = sess.span_diagnostic.struct_span_warn(span, explain);
    add_feature_diagnostics_for_issue(&mut err, sess, feature, issue);

    // Decorate this as a future-incompatibility lint as in rustc_middle::lint::struct_lint_level
    let lint = UNSTABLE_SYNTAX_PRE_EXPANSION;
    let future_incompatible = lint.future_incompatible.as_ref().unwrap();
    err.code(DiagnosticId::Lint {
        name: lint.name_lower(),
        has_future_breakage: false,
        is_force_warn: false,
    });
    err.warn(lint.desc);
    err.note(format!("for more information, see {}", future_incompatible.reference));

    // A later feature_err call can steal and cancel this warning.
    err.stash(span, StashKey::EarlySyntaxWarning);
}

/// Adds the diagnostics for a feature to an existing error.
pub fn add_feature_diagnostics<'a>(err: &mut Diagnostic, sess: &'a ParseSess, feature: Symbol) {
    add_feature_diagnostics_for_issue(err, sess, feature, GateIssue::Language);
}

/// Adds the diagnostics for a feature to an existing error.
///
/// This variant allows you to control whether it is a library or language feature.
/// Almost always, you want to use this for a language feature. If so, prefer
/// `add_feature_diagnostics`.
pub fn add_feature_diagnostics_for_issue<'a>(
    err: &mut Diagnostic,
    sess: &'a ParseSess,
    feature: Symbol,
    issue: GateIssue,
) {
    if let Some(n) = find_feature_issue(feature, issue) {
        err.subdiagnostic(FeatureDiagnosticForIssue { n });
    }

    // #23973: do not suggest `#![feature(...)]` if we are in beta/stable
    if sess.unstable_features.is_nightly_build() {
        err.subdiagnostic(FeatureDiagnosticHelp { feature });
    }
}

/// Info about a parsing session.
pub struct ParseSess {
    pub span_diagnostic: Handler,
    pub unstable_features: UnstableFeatures,
    pub config: CrateConfig,
    pub check_config: CrateCheckConfig,
    pub edition: Edition,
    /// Places where raw identifiers were used. This is used to avoid complaining about idents
    /// clashing with keywords in new editions.
    pub raw_identifier_spans: Lock<Vec<Span>>,
    /// Places where identifiers that contain invalid Unicode codepoints but that look like they
    /// should be. Useful to avoid bad tokenization when encountering emoji. We group them to
    /// provide a single error per unique incorrect identifier.
    pub bad_unicode_identifiers: Lock<FxHashMap<Symbol, Vec<Span>>>,
    source_map: Lrc<SourceMap>,
    pub buffered_lints: Lock<Vec<BufferedEarlyLint>>,
    /// Contains the spans of block expressions that could have been incomplete based on the
    /// operation token that followed it, but that the parser cannot identify without further
    /// analysis.
    pub ambiguous_block_expr_parse: Lock<FxHashMap<Span, Span>>,
    pub gated_spans: GatedSpans,
    pub symbol_gallery: SymbolGallery,
    /// The parser has reached `Eof` due to an unclosed brace. Used to silence unnecessary errors.
    pub reached_eof: Lock<bool>,
    /// Environment variables accessed during the build and their values when they exist.
    pub env_depinfo: Lock<FxHashSet<(Symbol, Option<Symbol>)>>,
    /// File paths accessed during the build.
    pub file_depinfo: Lock<FxHashSet<Symbol>>,
    /// All the type ascriptions expressions that have had a suggestion for likely path typo.
    pub type_ascription_path_suggestions: Lock<FxHashSet<Span>>,
    /// Whether cfg(version) should treat the current release as incomplete
    pub assume_incomplete_release: bool,
    /// Spans passed to `proc_macro::quote_span`. Each span has a numerical
    /// identifier represented by its position in the vector.
    pub proc_macro_quoted_spans: Lock<Vec<Span>>,
}

impl ParseSess {
    /// Used for testing.
    pub fn new(file_path_mapping: FilePathMapping) -> Self {
        let fallback_bundle = fallback_fluent_bundle(rustc_errors::DEFAULT_LOCALE_RESOURCES, false);
        let sm = Lrc::new(SourceMap::new(file_path_mapping));
        let handler = Handler::with_tty_emitter(
            ColorConfig::Auto,
            true,
            None,
            Some(sm.clone()),
            None,
            fallback_bundle,
        );
        ParseSess::with_span_handler(handler, sm)
    }

    pub fn with_span_handler(handler: Handler, source_map: Lrc<SourceMap>) -> Self {
        Self {
            span_diagnostic: handler,
            unstable_features: UnstableFeatures::from_environment(None),
            config: FxIndexSet::default(),
            check_config: CrateCheckConfig::default(),
            edition: ExpnId::root().expn_data().edition,
            raw_identifier_spans: Lock::new(Vec::new()),
            bad_unicode_identifiers: Lock::new(Default::default()),
            source_map,
            buffered_lints: Lock::new(vec![]),
            ambiguous_block_expr_parse: Lock::new(FxHashMap::default()),
            gated_spans: GatedSpans::default(),
            symbol_gallery: SymbolGallery::default(),
            reached_eof: Lock::new(false),
            env_depinfo: Default::default(),
            file_depinfo: Default::default(),
            type_ascription_path_suggestions: Default::default(),
            assume_incomplete_release: false,
            proc_macro_quoted_spans: Default::default(),
        }
    }

    pub fn with_silent_emitter(fatal_note: Option<String>) -> Self {
        let fallback_bundle = fallback_fluent_bundle(rustc_errors::DEFAULT_LOCALE_RESOURCES, false);
        let sm = Lrc::new(SourceMap::new(FilePathMapping::empty()));
        let fatal_handler =
            Handler::with_tty_emitter(ColorConfig::Auto, false, None, None, None, fallback_bundle);
        let handler = Handler::with_emitter(
            false,
            None,
            Box::new(SilentEmitter { fatal_handler, fatal_note }),
        );
        ParseSess::with_span_handler(handler, sm)
    }

    #[inline]
    pub fn source_map(&self) -> &SourceMap {
        &self.source_map
    }

    pub fn clone_source_map(&self) -> Lrc<SourceMap> {
        self.source_map.clone()
    }

    pub fn buffer_lint(
        &self,
        lint: &'static Lint,
        span: impl Into<MultiSpan>,
        node_id: NodeId,
        msg: &str,
    ) {
        self.buffered_lints.with_lock(|buffered_lints| {
            buffered_lints.push(BufferedEarlyLint {
                span: span.into(),
                node_id,
                msg: msg.into(),
                lint_id: LintId::of(lint),
                diagnostic: BuiltinLintDiagnostics::Normal,
            });
        });
    }

    pub fn buffer_lint_with_diagnostic(
        &self,
        lint: &'static Lint,
        span: impl Into<MultiSpan>,
        node_id: NodeId,
        msg: &str,
        diagnostic: BuiltinLintDiagnostics,
    ) {
        self.buffered_lints.with_lock(|buffered_lints| {
            buffered_lints.push(BufferedEarlyLint {
                span: span.into(),
                node_id,
                msg: msg.into(),
                lint_id: LintId::of(lint),
                diagnostic,
            });
        });
    }

    /// Extend an error with a suggestion to wrap an expression with parentheses to allow the
    /// parser to continue parsing the following operation as part of the same expression.
    pub fn expr_parentheses_needed(&self, err: &mut Diagnostic, span: Span) {
        err.multipart_suggestion(
            "parentheses are required to parse this as an expression",
            vec![(span.shrink_to_lo(), "(".to_string()), (span.shrink_to_hi(), ")".to_string())],
            Applicability::MachineApplicable,
        );
    }

    pub fn save_proc_macro_span(&self, span: Span) -> usize {
        let mut spans = self.proc_macro_quoted_spans.lock();
        spans.push(span);
        return spans.len() - 1;
    }

    pub fn proc_macro_quoted_spans(&self) -> Vec<Span> {
        self.proc_macro_quoted_spans.lock().clone()
    }

    pub fn create_err<'a>(
        &'a self,
        err: impl SessionDiagnostic<'a>,
    ) -> DiagnosticBuilder<'a, ErrorGuaranteed> {
        err.into_diagnostic(self)
    }

    pub fn emit_err<'a>(&'a self, err: impl SessionDiagnostic<'a>) -> ErrorGuaranteed {
        self.create_err(err).emit()
    }

    pub fn create_warning<'a>(
        &'a self,
        warning: impl SessionDiagnostic<'a, ()>,
    ) -> DiagnosticBuilder<'a, ()> {
        warning.into_diagnostic(self)
    }

    pub fn emit_warning<'a>(&'a self, warning: impl SessionDiagnostic<'a, ()>) {
        self.create_warning(warning).emit()
    }

    pub fn create_fatal<'a>(
        &'a self,
        fatal: impl SessionDiagnostic<'a, !>,
    ) -> DiagnosticBuilder<'a, !> {
        fatal.into_diagnostic(self)
    }

    pub fn emit_fatal<'a>(&'a self, fatal: impl SessionDiagnostic<'a, !>) -> ! {
        self.create_fatal(fatal).emit()
    }

    #[rustc_lint_diagnostics]
    #[allow(rustc::diagnostic_outside_of_impl)]
    #[allow(rustc::untranslatable_diagnostic)]
    pub fn struct_err(
        &self,
        msg: impl Into<DiagnosticMessage>,
    ) -> DiagnosticBuilder<'_, ErrorGuaranteed> {
        self.span_diagnostic.struct_err(msg)
    }

    #[rustc_lint_diagnostics]
    #[allow(rustc::diagnostic_outside_of_impl)]
    #[allow(rustc::untranslatable_diagnostic)]
    pub fn struct_warn(&self, msg: impl Into<DiagnosticMessage>) -> DiagnosticBuilder<'_, ()> {
        self.span_diagnostic.struct_warn(msg)
    }

    #[rustc_lint_diagnostics]
    #[allow(rustc::diagnostic_outside_of_impl)]
    #[allow(rustc::untranslatable_diagnostic)]
    pub fn struct_fatal(&self, msg: impl Into<DiagnosticMessage>) -> DiagnosticBuilder<'_, !> {
        self.span_diagnostic.struct_fatal(msg)
    }

    #[rustc_lint_diagnostics]
    #[allow(rustc::diagnostic_outside_of_impl)]
    #[allow(rustc::untranslatable_diagnostic)]
    pub fn struct_diagnostic<G: EmissionGuarantee>(
        &self,
        msg: impl Into<DiagnosticMessage>,
    ) -> DiagnosticBuilder<'_, G> {
        self.span_diagnostic.struct_diagnostic(msg)
    }
}
