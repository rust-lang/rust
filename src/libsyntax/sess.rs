//! Contains `ParseSess` which holds state living beyond what one `Parser` might.
//! It also serves as an input to the parser itself.

use crate::ast::{CrateConfig, NodeId};
use crate::early_buffered_lints::{BufferedEarlyLint, BufferedEarlyLintId};
use crate::source_map::{SourceMap, FilePathMapping};
use crate::feature_gate::UnstableFeatures;

use errors::{Applicability, emitter::SilentEmitter, Handler, ColorConfig, DiagnosticBuilder};
use rustc_data_structures::fx::{FxHashSet, FxHashMap};
use rustc_data_structures::sync::{Lrc, Lock, Once};
use syntax_pos::{Symbol, Span, MultiSpan};
use syntax_pos::edition::Edition;
use syntax_pos::hygiene::ExpnId;

use std::path::PathBuf;
use std::str;

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
        self.spans
            .borrow_mut()
            .entry(feature)
            .or_default()
            .push(span);
    }

    /// Ungate the last span under the given `feature`.
    /// Panics if the given `span` wasn't the last one.
    ///
    /// Using this is discouraged unless you have a really good reason to.
    pub fn ungate_last(&self, feature: Symbol, span: Span) {
        let removed_span = self.spans
            .borrow_mut()
            .entry(feature)
            .or_default()
            .pop()
            .unwrap();
        debug_assert_eq!(span, removed_span);
    }

    /// Is the provided `feature` gate ungated currently?
    ///
    /// Using this is discouraged unless you have a really good reason to.
    pub fn is_ungated(&self, feature: Symbol) -> bool {
        self.spans
            .borrow()
            .get(&feature)
            .map_or(true, |spans| spans.is_empty())
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

/// Info about a parsing session.
pub struct ParseSess {
    pub span_diagnostic: Handler,
    pub unstable_features: UnstableFeatures,
    pub config: CrateConfig,
    pub edition: Edition,
    pub missing_fragment_specifiers: Lock<FxHashSet<Span>>,
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
    pub injected_crate_name: Once<Symbol>,
    pub gated_spans: GatedSpans,
    /// The parser has reached `Eof` due to an unclosed brace. Used to silence unnecessary errors.
    pub reached_eof: Lock<bool>,
}

impl ParseSess {
    pub fn new(file_path_mapping: FilePathMapping) -> Self {
        let cm = Lrc::new(SourceMap::new(file_path_mapping));
        let handler = Handler::with_tty_emitter(
            ColorConfig::Auto,
            true,
            None,
            Some(cm.clone()),
        );
        ParseSess::with_span_handler(handler, cm)
    }

    pub fn with_span_handler(
        handler: Handler,
        source_map: Lrc<SourceMap>,
    ) -> Self {
        Self {
            span_diagnostic: handler,
            unstable_features: UnstableFeatures::from_environment(),
            config: FxHashSet::default(),
            edition: ExpnId::root().expn_data().edition,
            missing_fragment_specifiers: Lock::new(FxHashSet::default()),
            raw_identifier_spans: Lock::new(Vec::new()),
            included_mod_stack: Lock::new(vec![]),
            source_map,
            buffered_lints: Lock::new(vec![]),
            ambiguous_block_expr_parse: Lock::new(FxHashMap::default()),
            injected_crate_name: Once::new(),
            gated_spans: GatedSpans::default(),
            reached_eof: Lock::new(false),
        }
    }

    pub fn with_silent_emitter() -> Self {
        let cm = Lrc::new(SourceMap::new(FilePathMapping::empty()));
        let handler = Handler::with_emitter(false, None, Box::new(SilentEmitter));
        ParseSess::with_span_handler(handler, cm)
    }

    #[inline]
    pub fn source_map(&self) -> &SourceMap {
        &self.source_map
    }

    pub fn buffer_lint(
        &self,
        lint_id: BufferedEarlyLintId,
        span: impl Into<MultiSpan>,
        id: NodeId,
        msg: &str,
    ) {
        self.buffered_lints.with_lock(|buffered_lints| {
            buffered_lints.push(BufferedEarlyLint{
                span: span.into(),
                id,
                msg: msg.into(),
                lint_id,
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
