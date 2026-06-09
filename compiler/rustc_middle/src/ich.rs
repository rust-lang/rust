use std::hash::Hash;

use rustc_data_structures::stable_hash::{
    RawDefId, RawDefPathHash, RawSpan, StableHash, StableHashControls, StableHashCtxt, StableHasher,
};
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_session::Session;
use rustc_session::cstore::Untracked;
use rustc_span::source_map::SourceMap;
use rustc_span::{CachingSourceMapView, DUMMY_SP, Pos, Span};

// Very often, we are hashing something that does not need the `CachingSourceMapView`, so we
// initialize it lazily.
enum CachingSourceMap<'a> {
    Unused(&'a SourceMap),
    InUse(CachingSourceMapView<'a>),
}

/// This is the context state available during incr. comp. hashing. It contains
/// enough information to transform `DefId`s and `HirId`s into stable `DefPath`s (i.e.,
/// a reference to the `TyCtxt`) and it holds a few caches for speeding up various
/// things (e.g., each `DefId`/`DefPath` is only hashed once).
pub struct StableHashState<'a> {
    untracked: &'a Untracked,
    // The value of `-Z incremental-ignore-spans`.
    // This field should only be used by `unstable_opts_incremental_ignore_span`
    incremental_ignore_spans: bool,
    caching_source_map: CachingSourceMap<'a>,
    stable_hash_controls: StableHashControls,
}

impl<'a> StableHashState<'a> {
    #[inline]
    pub fn new(sess: &'a Session, untracked: &'a Untracked) -> Self {
        let hash_spans_initial = !sess.opts.unstable_opts.incremental_ignore_spans;

        StableHashState {
            untracked,
            incremental_ignore_spans: sess.opts.unstable_opts.incremental_ignore_spans,
            caching_source_map: CachingSourceMap::Unused(sess.source_map()),
            stable_hash_controls: StableHashControls { hash_spans: hash_spans_initial },
        }
    }

    #[inline]
    pub fn while_hashing_spans<F: FnOnce(&mut Self)>(&mut self, hash_spans: bool, f: F) {
        let prev_hash_spans = self.stable_hash_controls.hash_spans;
        self.stable_hash_controls.hash_spans = hash_spans;
        f(self);
        self.stable_hash_controls.hash_spans = prev_hash_spans;
    }

    #[inline]
    fn source_map(&mut self) -> &mut CachingSourceMapView<'a> {
        match self.caching_source_map {
            CachingSourceMap::InUse(ref mut sm) => sm,
            CachingSourceMap::Unused(sm) => {
                self.caching_source_map = CachingSourceMap::InUse(CachingSourceMapView::new(sm));
                self.source_map() // this recursive call will hit the `InUse` case
            }
        }
    }

    #[inline]
    fn def_span(&self, def_id: LocalDefId) -> Span {
        self.untracked.source_span.get(def_id).unwrap_or(DUMMY_SP)
    }

    #[inline]
    pub fn stable_hash_controls(&self) -> StableHashControls {
        self.stable_hash_controls
    }
}

impl<'a> StableHashCtxt for StableHashState<'a> {
    /// Hashes a span in a stable way. We can't directly hash the span's `BytePos` fields (that
    /// would be similar to hashing pointers, since those are just offsets into the `SourceMap`).
    /// Instead, we hash the (file name, line, column) triple, which stays the same even if the
    /// containing `SourceFile` has moved within the `SourceMap`.
    ///
    /// Also note that we are hashing byte offsets for the column, not unicode codepoint offsets.
    /// For the purpose of the hash that's sufficient. Also, hashing filenames is expensive so we
    /// avoid doing it twice when the span starts and ends in the same file, which is almost always
    /// the case.
    ///
    /// IMPORTANT: changes to this method should be reflected in implementations of `SpanEncoder`.
    #[inline]
    fn stable_hash_span(&mut self, raw_span: RawSpan, hasher: &mut StableHasher) {
        const TAG_VALID_SPAN: u8 = 0;
        const TAG_INVALID_SPAN: u8 = 1;
        const TAG_RELATIVE_SPAN: u8 = 2;

        if !self.stable_hash_controls().hash_spans {
            return;
        }

        let span = Span::from_raw_span(raw_span);
        let span = span.data_untracked();
        span.ctxt.stable_hash(self, hasher);
        span.parent.stable_hash(self, hasher);

        if span.is_dummy() {
            Hash::hash(&TAG_INVALID_SPAN, hasher);
            return;
        }

        let parent = span.parent.map(|parent| self.def_span(parent).data_untracked());
        if let Some(parent) = parent
            && parent.contains(span)
        {
            // This span is enclosed in a definition: only hash the relative position. This catches
            // a subset of the cases from the `file.contains(parent.lo)`. But we can do this check
            // cheaply without the expensive `span_data_to_lines_and_cols` query.
            Hash::hash(&TAG_RELATIVE_SPAN, hasher);
            (span.lo - parent.lo).to_u32().stable_hash(self, hasher);
            (span.hi - parent.lo).to_u32().stable_hash(self, hasher);
            return;
        }

        // If this is not an empty or invalid span, we want to hash the last position that belongs
        // to it, as opposed to hashing the first position past it.
        let Some((file, line_lo, col_lo, line_hi, col_hi)) =
            self.source_map().span_data_to_lines_and_cols(&span)
        else {
            Hash::hash(&TAG_INVALID_SPAN, hasher);
            return;
        };

        if let Some(parent) = parent
            && file.contains(parent.lo)
        {
            // This span is relative to another span in the same file,
            // only hash the relative position.
            Hash::hash(&TAG_RELATIVE_SPAN, hasher);
            Hash::hash(&(span.lo.0.wrapping_sub(parent.lo.0)), hasher);
            Hash::hash(&(span.hi.0.wrapping_sub(parent.lo.0)), hasher);
            return;
        }

        Hash::hash(&TAG_VALID_SPAN, hasher);
        Hash::hash(&file.stable_id, hasher);

        // Hash both the length and the end location (line/column) of a span. If we hash only the
        // length, for example, then two otherwise equal spans with different end locations will
        // have the same hash. This can cause a problem during incremental compilation wherein a
        // previous result for a query that depends on the end location of a span will be
        // incorrectly reused when the end location of the span it depends on has changed (see
        // issue #74890). A similar analysis applies if some query depends specifically on the
        // length of the span, but we only hash the end location. So hash both.

        let col_lo_trunc = (col_lo.0 as u64) & 0xFF;
        let line_lo_trunc = ((line_lo as u64) & 0xFF_FF_FF) << 8;
        let col_hi_trunc = (col_hi.0 as u64) & 0xFF << 32;
        let line_hi_trunc = ((line_hi as u64) & 0xFF_FF_FF) << 40;
        let col_line = col_lo_trunc | line_lo_trunc | col_hi_trunc | line_hi_trunc;
        let len = (span.hi - span.lo).0;
        Hash::hash(&col_line, hasher);
        Hash::hash(&len, hasher);
    }

    #[inline]
    fn def_path_hash(&self, raw_def_id: RawDefId) -> RawDefPathHash {
        let def_id = DefId::from_raw_def_id(raw_def_id);
        if let Some(def_id) = def_id.as_local() {
            self.untracked.definitions.read().def_path_hash(def_id)
        } else {
            self.untracked.cstore.read().def_path_hash(def_id)
        }
        .to_raw_def_path_hash()
    }

    /// Assert that the provided `StableHashCtxt` is configured with the default
    /// `StableHashControls`. We should always have bailed out before getting to here with a
    /// non-default mode. With this check in place, we can avoid the need to maintain separate
    /// versions of `ExpnData` hashes for each permutation of `StableHashControls` settings.
    #[inline]
    fn assert_default_stable_hash_controls(&self, msg: &str) {
        let stable_hash_controls = self.stable_hash_controls;
        let StableHashControls { hash_spans } = stable_hash_controls;

        // Note that we require that `hash_spans` be the inverse of the global `-Z
        // incremental-ignore-spans` option. Normally, this option is disabled, in which case
        // `hash_spans` must be true.
        //
        // Span hashing can also be disabled without `-Z incremental-ignore-spans`. This is the
        // case for instance when building a hash for name mangling. Such configuration must not be
        // used for metadata.
        assert_eq!(
            hash_spans, !self.incremental_ignore_spans,
            "Attempted hashing of {msg} with non-default StableHashControls: {stable_hash_controls:?}"
        );
    }

    #[inline]
    fn stable_hash_controls(&self) -> StableHashControls {
        self.stable_hash_controls
    }
}
