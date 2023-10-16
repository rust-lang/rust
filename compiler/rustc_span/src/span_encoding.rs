use crate::def_id::{DefIndex, LocalDefId};
use crate::hygiene::SyntaxContext;
use crate::SPAN_TRACK;
use crate::{BytePos, SpanData};

use rustc_data_structures::fx::FxIndexSet;

/// A compressed span.
///
/// [`SpanData`] is 16 bytes, which is too big to stick everywhere. `Span` only
/// takes up 8 bytes, with less space for the length, parent and context. The
/// vast majority (99.9%+) of `SpanData` instances can be made to fit within
/// those 8 bytes. Any `SpanData` whose fields don't fit into a `Span` are
/// stored in a separate interner table, and the `Span` will index into that
/// table. Interning is rare enough that the cost is low, but common enough
/// that the code is exercised regularly.
///
/// An earlier version of this code used only 4 bytes for `Span`, but that was
/// slower because only 80--90% of spans could be stored inline (even less in
/// very large crates) and so the interner was used a lot more. That version of
/// the code also predated the storage of parents.
///
/// There are four different span forms.
///
/// Inline-context format (requires non-huge length, non-huge context, and no parent):
/// - `span.lo_or_index == span_data.lo`
/// - `span.len_with_tag_or_marker == len == span_data.hi - span_data.lo` (must be `<= MAX_LEN`)
/// - `span.ctxt_or_parent_or_marker == span_data.ctxt` (must be `<= MAX_CTXT`)
///
/// Inline-parent format (requires non-huge length, root context, and non-huge parent):
/// - `span.lo_or_index == span_data.lo`
/// - `span.len_with_tag_or_marker & !PARENT_TAG == len == span_data.hi - span_data.lo`
///   (must be `<= MAX_LEN`)
/// - `span.len_with_tag_or_marker` has top bit (`PARENT_TAG`) set
/// - `span.ctxt_or_parent_or_marker == span_data.parent` (must be `<= MAX_CTXT`)
///
/// Partially-interned format (requires non-huge context):
/// - `span.lo_or_index == index` (indexes into the interner table)
/// - `span.len_with_tag_or_marker == BASE_LEN_INTERNED_MARKER`
/// - `span.ctxt_or_parent_or_marker == span_data.ctxt` (must be `<= MAX_CTXT`)
///
/// Fully-interned format (all cases not covered above):
/// - `span.lo_or_index == index` (indexes into the interner table)
/// - `span.len_with_tag_or_marker == BASE_LEN_INTERNED_MARKER`
/// - `span.ctxt_or_parent_or_marker == CTXT_INTERNED_MARKER`
///
/// The partially-interned form requires looking in the interning table for
/// lo and length, but the context is stored inline as well as interned.
/// This is useful because context lookups are often done in isolation, and
/// inline lookups are quicker.
///
/// Notes about the choice of field sizes:
/// - `lo` is 32 bits in both `Span` and `SpanData`, which means that `lo`
///   values never cause interning. The number of bits needed for `lo`
///   depends on the crate size. 32 bits allows up to 4 GiB of code in a crate.
///   Having no compression on this field means there is no performance cliff
///   if a crate exceeds a particular size.
/// - `len` is ~15 bits in `Span` (a u16, minus 1 bit for PARENT_TAG) and 32
///   bits in `SpanData`, which means that large `len` values will cause
///   interning. The number of bits needed for `len` does not depend on the
///   crate size. The most common numbers of bits for `len` are from 0 to 7,
///   with a peak usually at 3 or 4, and then it drops off quickly from 8
///   onwards. 15 bits is enough for 99.99%+ of cases, but larger values
///   (sometimes 20+ bits) might occur dozens of times in a typical crate.
/// - `ctxt_or_parent_or_marker` is 16 bits in `Span` and two 32 bit fields in
///   `SpanData`, which means intering will happen if `ctxt` is large, if
///   `parent` is large, or if both values are non-zero. The number of bits
///   needed for `ctxt` values depend partly on the crate size and partly on
///   the form of the code. No crates in `rustc-perf` need more than 15 bits
///   for `ctxt_or_parent_or_marker`, but larger crates might need more than 16
///   bits. The number of bits needed for `parent` hasn't been measured,
///   because `parent` isn't currently used by default.
///
/// In order to reliably use parented spans in incremental compilation,
/// the dependency to the parent definition's span. This is performed
/// using the callback `SPAN_TRACK` to access the query engine.
///
#[derive(Clone, Copy, Eq, PartialEq, Hash)]
#[rustc_pass_by_value]
pub struct Span {
    lo_or_index: u32,
    len_with_tag_or_marker: u16,
    ctxt_or_parent_or_marker: u16,
}

// `MAX_LEN` is chosen so that `PARENT_TAG | MAX_LEN` is distinct from
// `BASE_LEN_INTERNED_MARKER`. (If `MAX_LEN` was 1 higher, this wouldn't be true.)
const MAX_LEN: u32 = 0b0111_1111_1111_1110;
const MAX_CTXT: u32 = 0b0111_1111_1111_1110;
const PARENT_TAG: u16 = 0b1000_0000_0000_0000;
const BASE_LEN_INTERNED_MARKER: u16 = 0b1111_1111_1111_1111;
const CTXT_INTERNED_MARKER: u16 = 0b1111_1111_1111_1111;

/// The dummy span has zero position, length, and context, and no parent.
pub const DUMMY_SP: Span =
    Span { lo_or_index: 0, len_with_tag_or_marker: 0, ctxt_or_parent_or_marker: 0 };

impl Span {
    #[inline]
    pub fn new(
        mut lo: BytePos,
        mut hi: BytePos,
        ctxt: SyntaxContext,
        parent: Option<LocalDefId>,
    ) -> Self {
        if lo > hi {
            std::mem::swap(&mut lo, &mut hi);
        }

        let (lo2, len, ctxt2) = (lo.0, hi.0 - lo.0, ctxt.as_u32());

        if len <= MAX_LEN {
            if ctxt2 <= MAX_CTXT && parent.is_none() {
                // Inline-context format.
                return Span {
                    lo_or_index: lo2,
                    len_with_tag_or_marker: len as u16,
                    ctxt_or_parent_or_marker: ctxt2 as u16,
                };
            } else if ctxt2 == SyntaxContext::root().as_u32()
                && let Some(parent) = parent
                && let parent2 = parent.local_def_index.as_u32()
                && parent2 <= MAX_CTXT
            {
                // Inline-parent format.
                return Span {
                    lo_or_index: lo2,
                    len_with_tag_or_marker: PARENT_TAG | len as u16,
                    ctxt_or_parent_or_marker: parent2 as u16,
                };
            }
        }

        // Partially-interned or fully-interned format.
        let index =
            with_span_interner(|interner| interner.intern(&SpanData { lo, hi, ctxt, parent }));
        let ctxt_or_parent_or_marker = if ctxt2 <= MAX_CTXT {
            ctxt2 as u16 // partially-interned
        } else {
            CTXT_INTERNED_MARKER // fully-interned
        };
        Span {
            lo_or_index: index,
            len_with_tag_or_marker: BASE_LEN_INTERNED_MARKER,
            ctxt_or_parent_or_marker,
        }
    }

    #[inline]
    pub fn data(self) -> SpanData {
        let data = self.data_untracked();
        if let Some(parent) = data.parent {
            (*SPAN_TRACK)(parent);
        }
        data
    }

    /// Internal function to translate between an encoded span and the expanded representation.
    /// This function must not be used outside the incremental engine.
    #[inline]
    pub fn data_untracked(self) -> SpanData {
        if self.len_with_tag_or_marker != BASE_LEN_INTERNED_MARKER {
            if self.len_with_tag_or_marker & PARENT_TAG == 0 {
                // Inline-context format.
                let len = self.len_with_tag_or_marker as u32;
                debug_assert!(len <= MAX_LEN);
                SpanData {
                    lo: BytePos(self.lo_or_index),
                    hi: BytePos(self.lo_or_index + len),
                    ctxt: SyntaxContext::from_u32(self.ctxt_or_parent_or_marker as u32),
                    parent: None,
                }
            } else {
                // Inline-parent format.
                let len = (self.len_with_tag_or_marker & !PARENT_TAG) as u32;
                debug_assert!(len <= MAX_LEN);
                let parent = LocalDefId {
                    local_def_index: DefIndex::from_u32(self.ctxt_or_parent_or_marker as u32),
                };
                SpanData {
                    lo: BytePos(self.lo_or_index),
                    hi: BytePos(self.lo_or_index + len),
                    ctxt: SyntaxContext::root(),
                    parent: Some(parent),
                }
            }
        } else {
            // Fully-interned or partially-interned format. In either case,
            // the interned value contains all the data, so we don't need to
            // distinguish them.
            let index = self.lo_or_index;
            with_span_interner(|interner| interner.spans[index as usize])
        }
    }

    /// Returns `true` if this is a dummy span with any hygienic context.
    #[inline]
    pub fn is_dummy(self) -> bool {
        if self.len_with_tag_or_marker != BASE_LEN_INTERNED_MARKER {
            // Inline-context or inline-parent format.
            let lo = self.lo_or_index;
            let len = (self.len_with_tag_or_marker & !PARENT_TAG) as u32;
            debug_assert!(len <= MAX_LEN);
            lo == 0 && len == 0
        } else {
            // Fully-interned or partially-interned format.
            let index = self.lo_or_index;
            let data = with_span_interner(|interner| interner.spans[index as usize]);
            data.lo == BytePos(0) && data.hi == BytePos(0)
        }
    }

    /// This function is used as a fast path when decoding the full `SpanData` is not necessary.
    /// It's a cut-down version of `data_untracked`.
    #[cfg_attr(not(test), rustc_diagnostic_item = "SpanCtxt")]
    #[inline]
    pub fn ctxt(self) -> SyntaxContext {
        if self.len_with_tag_or_marker != BASE_LEN_INTERNED_MARKER {
            if self.len_with_tag_or_marker & PARENT_TAG == 0 {
                // Inline-context format.
                SyntaxContext::from_u32(self.ctxt_or_parent_or_marker as u32)
            } else {
                // Inline-parent format. We know that the SyntaxContext is root.
                SyntaxContext::root()
            }
        } else {
            if self.ctxt_or_parent_or_marker != CTXT_INTERNED_MARKER {
                // Partially-interned format. This path avoids looking up the
                // interned value, and is the whole point of the
                // partially-interned format.
                SyntaxContext::from_u32(self.ctxt_or_parent_or_marker as u32)
            } else {
                // Fully-interned format.
                let index = self.lo_or_index;
                with_span_interner(|interner| interner.spans[index as usize].ctxt)
            }
        }
    }
}

#[derive(Default)]
pub struct SpanInterner {
    spans: FxIndexSet<SpanData>,
}

impl SpanInterner {
    fn intern(&mut self, span_data: &SpanData) -> u32 {
        let (index, _) = self.spans.insert_full(*span_data);
        index as u32
    }
}

// If an interner exists, return it. Otherwise, prepare a fresh one.
#[inline]
fn with_span_interner<T, F: FnOnce(&mut SpanInterner) -> T>(f: F) -> T {
    crate::with_session_globals(|session_globals| f(&mut session_globals.span_interner.lock()))
}
