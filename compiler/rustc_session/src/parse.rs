//! Contains `ParseSess` which holds state living beyond what one `Parser` might.
//! It also serves as an input to the parser itself.

use std::str;
use std::sync::Arc;

use rustc_ast::attr::AttrIdGenerator;
use rustc_ast::node_id::NodeId;
use rustc_data_structures::fx::{FxHashMap, FxIndexMap, FxIndexSet};
use rustc_data_structures::sync::{AppendOnlyVec, Lock};
use rustc_errors::emitter::{HumanEmitter, SilentEmitter, stderr_destination};
use rustc_errors::{
    ColorConfig, Diag, DiagCtxt, DiagCtxtHandle, DiagMessage, EmissionGuarantee, MultiSpan,
    StashKey, fallback_fluent_bundle,
};
use rustc_feature::{GateIssue, UnstableFeatures, find_feature_issue};
use rustc_span::edition::Edition;
use rustc_span::hygiene::ExpnId;
use rustc_span::source_map::{FilePathMapping, SourceMap};
use rustc_span::{Span, Symbol, sym};

use crate::Session;
use crate::config::{Cfg, CheckCfg};
use crate::errors::{
    CliFeatureDiagnosticHelp, FeatureDiagnosticForIssue, FeatureDiagnosticHelp,
    FeatureDiagnosticSuggestion, FeatureGateError, SuggestUpgradeCompiler,
};
use crate::lint::builtin::UNSTABLE_SYNTAX_PRE_EXPANSION;
use crate::lint::{BufferedEarlyLint, BuiltinLintDiag, Lint, LintId};

/// Collected spans during parsing for places where a certain feature was
/// used and should be feature gated accordingly in `check_crate`.
#[derive(Default)]
pub struct GatedSpans {
    pub spans: Lock<FxHashMap<Symbol, Vec<Span>>>,
}

impl GatedSpans {
    /// Feature gate the given `span` under the given `feature`
    /// which is same `Symbol` used in `unstable.rs`.
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

    /// Prepend the given set of `spans` onto the set in `self`.
    pub fn merge(&self, mut spans: FxHashMap<Symbol, Vec<Span>>) {
        let mut inner = self.spans.borrow_mut();
        // The entries will be moved to another map so the drain order does not
        // matter.
        #[allow(rustc::potential_query_instability)]
        for (gate, mut gate_spans) in inner.drain() {
            spans.entry(gate).or_default().append(&mut gate_spans);
        }
        *inner = spans;
    }
}

#[derive(Default)]
pub struct SymbolGallery {
    /// All symbols occurred and their first occurrence span.
    pub symbols: Lock<FxIndexMap<Symbol, Span>>,
}

impl SymbolGallery {
    /// Insert a symbol and its span into symbol gallery.
    /// If the symbol has occurred before, ignore the new occurrence.
    pub fn insert(&self, symbol: Symbol, span: Span) {
        self.symbols.lock().entry(symbol).or_insert(span);
    }
}

// todo: this function now accepts `Session` instead of `ParseSess` and should be relocated
/// Construct a diagnostic for a language feature error due to the given `span`.
/// The `feature`'s `Symbol` is the one you used in `unstable.rs` and `rustc_span::symbol`.
#[track_caller]
pub fn feature_err(
    sess: &Session,
    feature: Symbol,
    span: impl Into<MultiSpan>,
    explain: impl Into<DiagMessage>,
) -> Diag<'_> {
    feature_err_issue(sess, feature, span, GateIssue::Language, explain)
}

/// Construct a diagnostic for a feature gate error.
///
/// This variant allows you to control whether it is a library or language feature.
/// Almost always, you want to use this for a language feature. If so, prefer `feature_err`.
#[track_caller]
pub fn feature_err_issue(
    sess: &Session,
    feature: Symbol,
    span: impl Into<MultiSpan>,
    issue: GateIssue,
    explain: impl Into<DiagMessage>,
) -> Diag<'_> {
    let span = span.into();

    // Cancel an earlier warning for this same error, if it exists.
    if let Some(span) = span.primary_span() {
        if let Some(err) = sess.dcx().steal_non_err(span, StashKey::EarlySyntaxWarning) {
            err.cancel()
        }
    }

    let mut err = sess.dcx().create_err(FeatureGateError { span, explain: explain.into() });
    add_feature_diagnostics_for_issue(&mut err, sess, feature, issue, false, None);
    err
}

/// Construct a future incompatibility diagnostic for a feature gate.
///
/// This diagnostic is only a warning and *does not cause compilation to fail*.
#[track_caller]
pub fn feature_warn(sess: &Session, feature: Symbol, span: Span, explain: &'static str) {
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
#[track_caller]
pub fn feature_warn_issue(
    sess: &Session,
    feature: Symbol,
    span: Span,
    issue: GateIssue,
    explain: &'static str,
) {
    let mut err = sess.dcx().struct_span_warn(span, explain);
    add_feature_diagnostics_for_issue(&mut err, sess, feature, issue, false, None);

    // Decorate this as a future-incompatibility lint as in rustc_middle::lint::lint_level
    let lint = UNSTABLE_SYNTAX_PRE_EXPANSION;
    let future_incompatible = lint.future_incompatible.as_ref().unwrap();
    err.is_lint(lint.name_lower(), /* has_future_breakage */ false);
    err.warn(lint.desc);
    err.note(format!("for more information, see {}", future_incompatible.reference));

    // A later feature_err call can steal and cancel this warning.
    err.stash(span, StashKey::EarlySyntaxWarning);
}

/// Adds the diagnostics for a feature to an existing error.
/// Must be a language feature!
pub fn add_feature_diagnostics<G: EmissionGuarantee>(
    err: &mut Diag<'_, G>,
    sess: &Session,
    feature: Symbol,
) {
    add_feature_diagnostics_for_issue(err, sess, feature, GateIssue::Language, false, None);
}

/// Adds the diagnostics for a feature to an existing error.
///
/// This variant allows you to control whether it is a library or language feature.
/// Almost always, you want to use this for a language feature. If so, prefer
/// `add_feature_diagnostics`.
#[allow(rustc::diagnostic_outside_of_impl)] // FIXME
pub fn add_feature_diagnostics_for_issue<G: EmissionGuarantee>(
    err: &mut Diag<'_, G>,
    sess: &Session,
    feature: Symbol,
    issue: GateIssue,
    feature_from_cli: bool,
    inject_span: Option<Span>,
) {
    if let Some(n) = find_feature_issue(feature, issue) {
        err.subdiagnostic(FeatureDiagnosticForIssue { n });
    }

    // #23973: do not suggest `#![feature(...)]` if we are in beta/stable
    if sess.psess.unstable_features.is_nightly_build() {
        if feature_from_cli {
            err.subdiagnostic(CliFeatureDiagnosticHelp { feature });
        } else if let Some(span) = inject_span {
            err.subdiagnostic(FeatureDiagnosticSuggestion { feature, span });
        } else {
            err.subdiagnostic(FeatureDiagnosticHelp { feature });
        }
        if feature == sym::rustc_attrs {
            // We're unlikely to stabilize something out of `rustc_attrs`
            // without at least renaming it, so pointing out how old
            // the compiler is will do little good.
        } else if sess.opts.unstable_opts.ui_testing {
            err.subdiagnostic(SuggestUpgradeCompiler::ui_testing());
        } else if let Some(suggestion) = SuggestUpgradeCompiler::new() {
            err.subdiagnostic(suggestion);
        }
    }
}

/// Info about a parsing session.
pub struct ParseSess {
    dcx: DiagCtxt,
    pub unstable_features: UnstableFeatures,
    pub config: Cfg,
    pub check_config: CheckCfg,
    pub edition: Edition,
    /// Places where raw identifiers were used. This is used to avoid complaining about idents
    /// clashing with keywords in new editions.
    pub raw_identifier_spans: AppendOnlyVec<Span>,
    /// Places where identifiers that contain invalid Unicode codepoints but that look like they
    /// should be. Useful to avoid bad tokenization when encountering emoji. We group them to
    /// provide a single error per unique incorrect identifier.
    pub bad_unicode_identifiers: Lock<FxIndexMap<Symbol, Vec<Span>>>,
    source_map: Arc<SourceMap>,
    pub buffered_lints: Lock<Vec<BufferedEarlyLint>>,
    /// Contains the spans of block expressions that could have been incomplete based on the
    /// operation token that followed it, but that the parser cannot identify without further
    /// analysis.
    pub ambiguous_block_expr_parse: Lock<FxIndexMap<Span, Span>>,
    pub gated_spans: GatedSpans,
    pub symbol_gallery: SymbolGallery,
    /// Environment variables accessed during the build and their values when they exist.
    pub env_depinfo: Lock<FxIndexSet<(Symbol, Option<Symbol>)>>,
    /// File paths accessed during the build.
    pub file_depinfo: Lock<FxIndexSet<Symbol>>,
    /// Whether cfg(version) should treat the current release as incomplete
    pub assume_incomplete_release: bool,
    /// Spans passed to `proc_macro::quote_span`. Each span has a numerical
    /// identifier represented by its position in the vector.
    proc_macro_quoted_spans: AppendOnlyVec<Span>,
    /// Used to generate new `AttrId`s. Every `AttrId` is unique.
    pub attr_id_generator: AttrIdGenerator,
}

impl ParseSess {
    /// Used for testing.
    pub fn new(locale_resources: Vec<&'static str>) -> Self {
        let fallback_bundle = fallback_fluent_bundle(locale_resources, false);
        let sm = Arc::new(SourceMap::new(FilePathMapping::empty()));
        let emitter = Box::new(
            HumanEmitter::new(stderr_destination(ColorConfig::Auto), fallback_bundle)
                .sm(Some(Arc::clone(&sm))),
        );
        let dcx = DiagCtxt::new(emitter);
        ParseSess::with_dcx(dcx, sm)
    }

    pub fn with_dcx(dcx: DiagCtxt, source_map: Arc<SourceMap>) -> Self {
        Self {
            dcx,
            unstable_features: UnstableFeatures::from_environment(None),
            config: Cfg::default(),
            check_config: CheckCfg::default(),
            edition: ExpnId::root().expn_data().edition,
            raw_identifier_spans: Default::default(),
            bad_unicode_identifiers: Lock::new(Default::default()),
            source_map,
            buffered_lints: Lock::new(vec![]),
            ambiguous_block_expr_parse: Lock::new(Default::default()),
            gated_spans: GatedSpans::default(),
            symbol_gallery: SymbolGallery::default(),
            env_depinfo: Default::default(),
            file_depinfo: Default::default(),
            assume_incomplete_release: false,
            proc_macro_quoted_spans: Default::default(),
            attr_id_generator: AttrIdGenerator::new(),
        }
    }

    pub fn with_silent_emitter(
        locale_resources: Vec<&'static str>,
        fatal_note: String,
        emit_fatal_diagnostic: bool,
    ) -> Self {
        let fallback_bundle = fallback_fluent_bundle(locale_resources, false);
        let sm = Arc::new(SourceMap::new(FilePathMapping::empty()));
        let fatal_emitter =
            Box::new(HumanEmitter::new(stderr_destination(ColorConfig::Auto), fallback_bundle));
        let dcx = DiagCtxt::new(Box::new(SilentEmitter {
            fatal_emitter,
            fatal_note: Some(fatal_note),
            emit_fatal_diagnostic,
        }))
        .disable_warnings();
        ParseSess::with_dcx(dcx, sm)
    }

    #[inline]
    pub fn source_map(&self) -> &SourceMap {
        &self.source_map
    }

    pub fn clone_source_map(&self) -> Arc<SourceMap> {
        Arc::clone(&self.source_map)
    }

    pub fn buffer_lint(
        &self,
        lint: &'static Lint,
        span: impl Into<MultiSpan>,
        node_id: NodeId,
        diagnostic: BuiltinLintDiag,
    ) {
        self.opt_span_buffer_lint(lint, Some(span.into()), node_id, diagnostic)
    }

    pub fn opt_span_buffer_lint(
        &self,
        lint: &'static Lint,
        span: Option<MultiSpan>,
        node_id: NodeId,
        diagnostic: BuiltinLintDiag,
    ) {
        self.buffered_lints.with_lock(|buffered_lints| {
            buffered_lints.push(BufferedEarlyLint {
                span,
                node_id,
                lint_id: LintId::of(lint),
                diagnostic,
            });
        });
    }

    pub fn save_proc_macro_span(&self, span: Span) -> usize {
        self.proc_macro_quoted_spans.push(span)
    }

    pub fn proc_macro_quoted_spans(&self) -> impl Iterator<Item = (usize, Span)> {
        // This is equivalent to `.iter().copied().enumerate()`, but that isn't possible for
        // AppendOnlyVec, so we resort to this scheme.
        self.proc_macro_quoted_spans.iter_enumerated()
    }

    pub fn dcx(&self) -> DiagCtxtHandle<'_> {
        self.dcx.handle()
    }
}
