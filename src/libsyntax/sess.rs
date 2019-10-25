//! Contains `ParseSess` which holds state living beyond what one `Parser` might.
//! It also serves as an input to the parser itself.

use crate::ast::{CrateConfig, NodeId};
use crate::early_buffered_lints::{BufferedEarlyLint, BufferedEarlyLintId};
use crate::source_map::{SourceMap, FilePathMapping};
use crate::feature_gate::UnstableFeatures;

use errors::{Applicability, Handler, ColorConfig, DiagnosticBuilder};
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
crate struct GatedSpans {
    /// Spans collected for gating `let_chains`, e.g. `if a && let b = c {}`.
    crate let_chains: Lock<Vec<Span>>,
    /// Spans collected for gating `async_closure`, e.g. `async || ..`.
    crate async_closure: Lock<Vec<Span>>,
    /// Spans collected for gating `yield e?` expressions (`generators` gate).
    crate yields: Lock<Vec<Span>>,
    /// Spans collected for gating `or_patterns`, e.g. `Some(Foo | Bar)`.
    crate or_patterns: Lock<Vec<Span>>,
    /// Spans collected for gating `const_extern_fn`, e.g. `const extern fn foo`.
    crate const_extern_fn: Lock<Vec<Span>>,
    /// Spans collected for gating `trait_alias`, e.g. `trait Foo = Ord + Eq;`.
    pub trait_alias: Lock<Vec<Span>>,
    /// Spans collected for gating `associated_type_bounds`, e.g. `Iterator<Item: Ord>`.
    pub associated_type_bounds: Lock<Vec<Span>>,
    /// Spans collected for gating `crate_visibility_modifier`, e.g. `crate fn`.
    pub crate_visibility_modifier: Lock<Vec<Span>>,
    /// Spans collected for gating `const_generics`, e.g. `const N: usize`.
    pub const_generics: Lock<Vec<Span>>,
    /// Spans collected for gating `decl_macro`, e.g. `macro m() {}`.
    pub decl_macro: Lock<Vec<Span>>,
    /// Spans collected for gating `box_patterns`, e.g. `box 0`.
    pub box_patterns: Lock<Vec<Span>>,
    /// Spans collected for gating `exclusive_range_pattern`, e.g. `0..2`.
    pub exclusive_range_pattern: Lock<Vec<Span>>,
    /// Spans collected for gating `try_blocks`, e.g. `try { a? + b? }`.
    pub try_blocks: Lock<Vec<Span>>,
    /// Spans collected for gating `label_break_value`, e.g. `'label: { ... }`.
    pub label_break_value: Lock<Vec<Span>>,
    /// Spans collected for gating `box_syntax`, e.g. `box $expr`.
    pub box_syntax: Lock<Vec<Span>>,
    /// Spans collected for gating `type_ascription`, e.g. `42: usize`.
    pub type_ascription: Lock<Vec<Span>>,
}

/// Info about a parsing session.
pub struct ParseSess {
    pub span_diagnostic: Handler,
    crate unstable_features: UnstableFeatures,
    pub config: CrateConfig,
    pub edition: Edition,
    pub missing_fragment_specifiers: Lock<FxHashSet<Span>>,
    /// Places where raw identifiers were used. This is used for feature-gating raw identifiers.
    pub raw_identifier_spans: Lock<Vec<Span>>,
    /// Used to determine and report recursive module inclusions.
    pub(super) included_mod_stack: Lock<Vec<PathBuf>>,
    source_map: Lrc<SourceMap>,
    pub buffered_lints: Lock<Vec<BufferedEarlyLint>>,
    /// Contains the spans of block expressions that could have been incomplete based on the
    /// operation token that followed it, but that the parser cannot identify without further
    /// analysis.
    pub ambiguous_block_expr_parse: Lock<FxHashMap<Span, Span>>,
    pub injected_crate_name: Once<Symbol>,
    crate gated_spans: GatedSpans,
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

    pub fn with_span_handler(handler: Handler, source_map: Lrc<SourceMap>) -> Self {
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
        }
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
