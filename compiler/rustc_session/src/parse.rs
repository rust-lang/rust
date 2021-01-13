//! Contains `ParseSess` which holds state living beyond what one `Parser` might.
//! It also serves as an input to the parser itself.

use crate::lint::{BufferedEarlyLint, BuiltinLintDiagnostics, Lint, LintId};
use rustc_ast::node_id::NodeId;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_data_structures::sync::{Lock, Lrc};
use rustc_errors::{emitter::SilentEmitter, ColorConfig, Handler};
use rustc_errors::{error_code, Applicability, DiagnosticBuilder};
use rustc_feature::{find_feature_issue, GateIssue, UnstableFeatures};
use rustc_span::edition::Edition;
use rustc_span::hygiene::ExpnId;
use rustc_span::source_map::{FilePathMapping, SourceMap};
use rustc_span::{MultiSpan, Span, Symbol};

use std::path::PathBuf;
use std::str;

/// The set of keys (and, optionally, values) that define the compilation
/// environment of the crate, used to drive conditional compilation.
pub type CrateConfig = FxHashSet<(Symbol, Option<Symbol>)>;

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
    /// If the symbol has occurred before, ignore the new occurance.
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
) -> DiagnosticBuilder<'a> {
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
) -> DiagnosticBuilder<'a> {
    let mut err = sess.span_diagnostic.struct_span_err_with_code(span, explain, error_code!(E0658));

    if let Some(n) = find_feature_issue(feature, issue) {
        err.note(&format!(
            "see issue #{} <https://github.com/rust-lang/rust/issues/{}> for more information",
            n, n,
        ));
    }

    // #23973: do not suggest `#![feature(...)]` if we are in beta/stable
    if sess.unstable_features.is_nightly_build() {
        err.help(&format!("add `#![feature({})]` to the crate attributes to enable", feature));
    }

    err
}

/// Info about a parsing session.
pub struct ParseSess {
    pub span_diagnostic: Handler,
    pub unstable_features: UnstableFeatures,
    pub config: CrateConfig,
    pub edition: Edition,
    pub missing_fragment_specifiers: Lock<FxHashMap<Span, NodeId>>,
    /// Places where raw identifiers were used. This is used for feature-gating raw identifiers.
    pub raw_identifier_spans: Lock<Vec<Span>>,
    /// Used to determine and report recursive module inclusions.
    pub included_mod_stack: Lock<Vec<PathBuf>>,
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
    /// All the type ascriptions expressions that have had a suggestion for likely path typo.
    pub type_ascription_path_suggestions: Lock<FxHashSet<Span>>,
}

impl ParseSess {
    pub fn new(file_path_mapping: FilePathMapping) -> Self {
        let sm = Lrc::new(SourceMap::new(file_path_mapping));
        let handler = Handler::with_tty_emitter(ColorConfig::Auto, true, None, Some(sm.clone()));
        ParseSess::with_span_handler(handler, sm)
    }

    pub fn with_span_handler(handler: Handler, source_map: Lrc<SourceMap>) -> Self {
        Self {
            span_diagnostic: handler,
            unstable_features: UnstableFeatures::from_environment(None),
            config: FxHashSet::default(),
            edition: ExpnId::root().expn_data().edition,
            missing_fragment_specifiers: Default::default(),
            raw_identifier_spans: Lock::new(Vec::new()),
            included_mod_stack: Lock::new(vec![]),
            source_map,
            buffered_lints: Lock::new(vec![]),
            ambiguous_block_expr_parse: Lock::new(FxHashMap::default()),
            gated_spans: GatedSpans::default(),
            symbol_gallery: SymbolGallery::default(),
            reached_eof: Lock::new(false),
            env_depinfo: Default::default(),
            type_ascription_path_suggestions: Default::default(),
        }
    }

    pub fn with_silent_emitter() -> Self {
        let sm = Lrc::new(SourceMap::new(FilePathMapping::empty()));
        let handler = Handler::with_emitter(false, None, Box::new(SilentEmitter));
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
    pub fn expr_parentheses_needed(
        &self,
        err: &mut DiagnosticBuilder<'_>,
        span: Span,
        alt_snippet: Option<String>,
    ) {
        if let Some(snippet) = self.source_map().span_to_snippet(span).ok().or(alt_snippet) {
            err.span_suggestion(
                span,
                "parentheses are required to parse this as an expression",
                format!("({})", snippet),
                Applicability::MachineApplicable,
            );
        }
    }
}
