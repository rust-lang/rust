//! Contains `ParseSess` which holds state living beyond what one `Parser` might.
//! It also serves as an input to the parser itself.

use std::sync::Arc;

use rustc_ast::attr::AttrIdGenerator;
use rustc_ast::node_id::NodeId;
use rustc_data_structures::fx::{FxHashMap, FxIndexMap};
use rustc_data_structures::sync::{AppendOnlyVec, DynSend, DynSync, Lock};
use rustc_errors::annotate_snippet_emitter_writer::AnnotateSnippetEmitter;
use rustc_errors::emitter::{EmitterWithNote, stderr_destination};
use rustc_errors::{
    BufferedEarlyLint, ColorConfig, DecorateDiagCompat, Diag, DiagCtxt, DiagCtxtHandle, Level,
    MultiSpan,
};
use rustc_span::edition::Edition;
use rustc_span::hygiene::ExpnId;
use rustc_span::source_map::{FilePathMapping, SourceMap};
use rustc_span::{Span, Symbol};

use crate::Session;
use crate::lint::{Lint, LintId};

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

/// Info about a parsing session.
pub struct ParseSess {
    dcx: DiagCtxt,
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
    /// Used to generate new `AttrId`s. Every `AttrId` is unique.
    pub attr_id_generator: AttrIdGenerator,
}

impl ParseSess {
    /// Used for testing.
    pub fn new() -> Self {
        let sm = Arc::new(SourceMap::new(FilePathMapping::empty()));
        let emitter = Box::new(
            AnnotateSnippetEmitter::new(stderr_destination(ColorConfig::Auto))
                .sm(Some(Arc::clone(&sm))),
        );
        let dcx = DiagCtxt::new(emitter);
        ParseSess::with_dcx(dcx, sm)
    }

    pub fn with_dcx(dcx: DiagCtxt, source_map: Arc<SourceMap>) -> Self {
        Self {
            dcx,
            edition: ExpnId::root().expn_data().edition,
            raw_identifier_spans: Default::default(),
            bad_unicode_identifiers: Lock::new(Default::default()),
            source_map,
            buffered_lints: Lock::new(vec![]),
            ambiguous_block_expr_parse: Lock::new(Default::default()),
            gated_spans: GatedSpans::default(),
            symbol_gallery: SymbolGallery::default(),
            attr_id_generator: AttrIdGenerator::new(),
        }
    }

    pub fn emitter_with_note(note: String) -> Self {
        let sm = Arc::new(SourceMap::new(FilePathMapping::empty()));
        let emitter = Box::new(AnnotateSnippetEmitter::new(stderr_destination(ColorConfig::Auto)));
        let dcx = DiagCtxt::new(Box::new(EmitterWithNote { emitter, note }));
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
        diagnostic: impl Into<DecorateDiagCompat>,
    ) {
        self.opt_span_buffer_lint(lint, Some(span.into()), node_id, diagnostic.into())
    }

    pub fn dyn_buffer_lint<
        F: for<'a> FnOnce(DiagCtxtHandle<'a>, Level) -> Diag<'a, ()> + DynSync + DynSend + 'static,
    >(
        &self,
        lint: &'static Lint,
        span: impl Into<MultiSpan>,
        node_id: NodeId,
        callback: F,
    ) {
        self.opt_span_buffer_lint(
            lint,
            Some(span.into()),
            node_id,
            DecorateDiagCompat(Box::new(|dcx, level, _| callback(dcx, level))),
        )
    }

    pub fn dyn_buffer_lint_sess<
        F: for<'a> FnOnce(DiagCtxtHandle<'a>, Level, &Session) -> Diag<'a, ()>
            + DynSync
            + DynSend
            + 'static,
    >(
        &self,
        lint: &'static Lint,
        span: impl Into<MultiSpan>,
        node_id: NodeId,
        callback: F,
    ) {
        self.opt_span_buffer_lint(
            lint,
            Some(span.into()),
            node_id,
            DecorateDiagCompat(Box::new(|dcx, level, sess| {
                let sess = sess.downcast_ref::<Session>().expect("expected a `Session`");
                callback(dcx, level, sess)
            })),
        )
    }

    pub(crate) fn opt_span_buffer_lint(
        &self,
        lint: &'static Lint,
        span: Option<MultiSpan>,
        node_id: NodeId,
        diagnostic: DecorateDiagCompat,
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

    pub fn dcx(&self) -> DiagCtxtHandle<'_> {
        self.dcx.handle()
    }
}
