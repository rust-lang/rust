use rustc_data_structures::fx::FxIndexSet;
// This code is very hot and uses lots of arithmetic, avoid overflow checks for performance.
// See https://github.com/rust-lang/rust/pull/119440#issuecomment-1874255727
use rustc_serialize::int_overflow::DebugStrictAdd;

use crate::def_id::{DefIndex, LocalDefId};
use crate::hygiene::SyntaxContext;
use crate::{BytePos, SPAN_TRACK, SpanData};

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
/// accesses to `lo` and `hi` must introduce a dependency to the parent definition's span.
/// This is performed using the callback `SPAN_TRACK` to access the query engine.
#[derive(Clone, Copy, Eq, PartialEq, Hash)]
#[rustc_pass_by_value]
pub struct Span {
    lo_or_index: u32,
    len_with_tag_or_marker: u16,
    ctxt_or_parent_or_marker: u16,
}

// Convenience structures for all span formats.
#[derive(Clone, Copy)]
struct InlineCtxt {
    lo: u32,
    len: u16,
    ctxt: u16,
}

#[derive(Clone, Copy)]
struct InlineParent {
    lo: u32,
    len_with_tag: u16,
    parent: u16,
}

#[derive(Clone, Copy)]
struct PartiallyInterned {
    index: u32,
    ctxt: u16,
}

#[derive(Clone, Copy)]
struct Interned {
    index: u32,
}

impl InlineCtxt {
    #[inline]
    fn data(self) -> SpanData {
        let len = self.len as u32;
        debug_assert!(len <= MAX_LEN);
        SpanData {
            lo: BytePos(self.lo),
            hi: BytePos(self.lo.debug_strict_add(len)),
            ctxt: SyntaxContext::from_u16(self.ctxt),
            parent: None,
        }
    }
    #[inline]
    fn span(lo: u32, len: u16, ctxt: u16) -> Span {
        Span { lo_or_index: lo, len_with_tag_or_marker: len, ctxt_or_parent_or_marker: ctxt }
    }
    #[inline]
    fn from_span(span: Span) -> InlineCtxt {
        let (lo, len, ctxt) =
            (span.lo_or_index, span.len_with_tag_or_marker, span.ctxt_or_parent_or_marker);
        InlineCtxt { lo, len, ctxt }
    }
}

impl InlineParent {
    #[inline]
    fn data(self) -> SpanData {
        let len = (self.len_with_tag & !PARENT_TAG) as u32;
        debug_assert!(len <= MAX_LEN);
        SpanData {
            lo: BytePos(self.lo),
            hi: BytePos(self.lo.debug_strict_add(len)),
            ctxt: SyntaxContext::root(),
            parent: Some(LocalDefId { local_def_index: DefIndex::from_u16(self.parent) }),
        }
    }
    #[inline]
    fn span(lo: u32, len: u16, parent: u16) -> Span {
        let (lo_or_index, len_with_tag_or_marker, ctxt_or_parent_or_marker) =
            (lo, PARENT_TAG | len, parent);
        Span { lo_or_index, len_with_tag_or_marker, ctxt_or_parent_or_marker }
    }
    #[inline]
    fn from_span(span: Span) -> InlineParent {
        let (lo, len_with_tag, parent) =
            (span.lo_or_index, span.len_with_tag_or_marker, span.ctxt_or_parent_or_marker);
        InlineParent { lo, len_with_tag, parent }
    }
}

impl PartiallyInterned {
    #[inline]
    fn data(self) -> SpanData {
        SpanData {
            ctxt: SyntaxContext::from_u16(self.ctxt),
            ..with_span_interner(|interner| interner.spans[self.index as usize])
        }
    }
    #[inline]
    fn span(index: u32, ctxt: u16) -> Span {
        let (lo_or_index, len_with_tag_or_marker, ctxt_or_parent_or_marker) =
            (index, BASE_LEN_INTERNED_MARKER, ctxt);
        Span { lo_or_index, len_with_tag_or_marker, ctxt_or_parent_or_marker }
    }
    #[inline]
    fn from_span(span: Span) -> PartiallyInterned {
        PartiallyInterned { index: span.lo_or_index, ctxt: span.ctxt_or_parent_or_marker }
    }
}

impl Interned {
    #[inline]
    fn data(self) -> SpanData {
        with_span_interner(|interner| interner.spans[self.index as usize])
    }
    #[inline]
    fn span(index: u32) -> Span {
        let (lo_or_index, len_with_tag_or_marker, ctxt_or_parent_or_marker) =
            (index, BASE_LEN_INTERNED_MARKER, CTXT_INTERNED_MARKER);
        Span { lo_or_index, len_with_tag_or_marker, ctxt_or_parent_or_marker }
    }
    #[inline]
    fn from_span(span: Span) -> Interned {
        Interned { index: span.lo_or_index }
    }
}

// This code is very hot, and converting span to an enum and matching on it doesn't optimize away
// properly. So we are using a macro emulating such a match, but expand it directly to an if-else
// chain.
macro_rules! match_span_kind {
    (
        $span:expr,
        InlineCtxt($span1:ident) => $arm1:expr,
        InlineParent($span2:ident) => $arm2:expr,
        PartiallyInterned($span3:ident) => $arm3:expr,
        Interned($span4:ident) => $arm4:expr,
    ) => {
        if $span.len_with_tag_or_marker != BASE_LEN_INTERNED_MARKER {
            if $span.len_with_tag_or_marker & PARENT_TAG == 0 {
                // Inline-context format.
                let $span1 = InlineCtxt::from_span($span);
                $arm1
            } else {
                // Inline-parent format.
                let $span2 = InlineParent::from_span($span);
                $arm2
            }
        } else if $span.ctxt_or_parent_or_marker != CTXT_INTERNED_MARKER {
            // Partially-interned format.
            let $span3 = PartiallyInterned::from_span($span);
            $arm3
        } else {
            // Interned format.
            let $span4 = Interned::from_span($span);
            $arm4
        }
    };
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

        // Small len and ctxt may enable one of fully inline formats (or may not).
        let (len, ctxt32) = (hi.0 - lo.0, ctxt.as_u32());
        if len <= MAX_LEN && ctxt32 <= MAX_CTXT {
            match parent {
                None => return InlineCtxt::span(lo.0, len as u16, ctxt32 as u16),
                Some(parent) => {
                    let parent32 = parent.local_def_index.as_u32();
                    if ctxt32 == 0 && parent32 <= MAX_CTXT {
                        return InlineParent::span(lo.0, len as u16, parent32 as u16);
                    }
                }
            }
        }

        // Otherwise small ctxt may enable the partially inline format.
        let index = |ctxt| {
            with_span_interner(|interner| interner.intern(&SpanData { lo, hi, ctxt, parent }))
        };
        if ctxt32 <= MAX_CTXT {
            // Interned ctxt should never be read, so it can use any value.
            PartiallyInterned::span(index(SyntaxContext::from_u32(u32::MAX)), ctxt32 as u16)
        } else {
            Interned::span(index(ctxt))
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
        match_span_kind! {
            self,
            InlineCtxt(span) => span.data(),
            InlineParent(span) => span.data(),
            PartiallyInterned(span) => span.data(),
            Interned(span) => span.data(),
        }
    }

    /// Returns `true` if this span comes from any kind of macro, desugaring or inlining.
    #[inline]
    pub fn from_expansion(self) -> bool {
        let ctxt = match_span_kind! {
            self,
            // All branches here, except `InlineParent`, actually return `span.ctxt_or_parent_or_marker`.
            // Since `Interned` is selected if the field contains `CTXT_INTERNED_MARKER` returning that value
            // as the context allows the compiler to optimize out the branch that selects between either
            // `Interned` and `PartiallyInterned`.
            //
            // Interned contexts can never be the root context and `CTXT_INTERNED_MARKER` has a different value
            // than the root context so this works for checking is this is an expansion.
            InlineCtxt(span) => SyntaxContext::from_u16(span.ctxt),
            InlineParent(_span) => SyntaxContext::root(),
            PartiallyInterned(span) => SyntaxContext::from_u16(span.ctxt),
            Interned(_span) => SyntaxContext::from_u16(CTXT_INTERNED_MARKER),
        };
        !ctxt.is_root()
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

    #[inline]
    pub fn map_ctxt(self, map: impl FnOnce(SyntaxContext) -> SyntaxContext) -> Span {
        let data = match_span_kind! {
            self,
            InlineCtxt(span) => {
                // This format occurs 1-2 orders of magnitude more often than others (#125017),
                // so it makes sense to micro-optimize it to avoid `span.data()` and `Span::new()`.
                let new_ctxt = map(SyntaxContext::from_u16(span.ctxt));
                let new_ctxt32 = new_ctxt.as_u32();
                return if new_ctxt32 <= MAX_CTXT {
                    // Any small new context including zero will preserve the format.
                    InlineCtxt::span(span.lo, span.len, new_ctxt32 as u16)
                } else {
                    span.data().with_ctxt(new_ctxt)
                };
            },
            InlineParent(span) => span.data(),
            PartiallyInterned(span) => span.data(),
            Interned(span) => span.data(),
        };

        data.with_ctxt(map(data.ctxt))
    }

    // Returns either syntactic context, if it can be retrieved without taking the interner lock,
    // or an index into the interner if it cannot.
    #[inline]
    fn inline_ctxt(self) -> Result<SyntaxContext, usize> {
        match_span_kind! {
            self,
            InlineCtxt(span) => Ok(SyntaxContext::from_u16(span.ctxt)),
            InlineParent(_span) => Ok(SyntaxContext::root()),
            PartiallyInterned(span) => Ok(SyntaxContext::from_u16(span.ctxt)),
            Interned(span) => Err(span.index as usize),
        }
    }

    /// This function is used as a fast path when decoding the full `SpanData` is not necessary.
    /// It's a cut-down version of `data_untracked`.
    #[cfg_attr(not(test), rustc_diagnostic_item = "SpanCtxt")]
    #[inline]
    pub fn ctxt(self) -> SyntaxContext {
        self.inline_ctxt()
            .unwrap_or_else(|index| with_span_interner(|interner| interner.spans[index].ctxt))
    }

    #[inline]
    pub fn eq_ctxt(self, other: Span) -> bool {
        match (self.inline_ctxt(), other.inline_ctxt()) {
            (Ok(ctxt1), Ok(ctxt2)) => ctxt1 == ctxt2,
            // If `inline_ctxt` returns `Ok` the context is <= MAX_CTXT.
            // If it returns `Err` the span is fully interned and the context is > MAX_CTXT.
            // As these do not overlap an `Ok` and `Err` result cannot have an equal context.
            (Ok(_), Err(_)) | (Err(_), Ok(_)) => false,
            (Err(index1), Err(index2)) => with_span_interner(|interner| {
                interner.spans[index1].ctxt == interner.spans[index2].ctxt
            }),
        }
    }

    #[inline]
    pub fn with_parent(self, parent: Option<LocalDefId>) -> Span {
        let data = match_span_kind! {
            self,
            InlineCtxt(span) => {
                // This format occurs 1-2 orders of magnitude more often than others (#126544),
                // so it makes sense to micro-optimize it to avoid `span.data()` and `Span::new()`.
                // Copypaste from `Span::new`, the small len & ctxt conditions are known to hold.
                match parent {
                    None => return self,
                    Some(parent) => {
                        let parent32 = parent.local_def_index.as_u32();
                        if span.ctxt == 0 && parent32 <= MAX_CTXT {
                            return InlineParent::span(span.lo, span.len, parent32 as u16);
                        }
                    }
                }
                span.data()
            },
            InlineParent(span) => span.data(),
            PartiallyInterned(span) => span.data(),
            Interned(span) => span.data(),
        };

        if let Some(old_parent) = data.parent {
            (*SPAN_TRACK)(old_parent);
        }
        data.with_parent(parent)
    }

    #[inline]
    pub fn parent(self) -> Option<LocalDefId> {
        let interned_parent =
            |index: u32| with_span_interner(|interner| interner.spans[index as usize].parent);
        match_span_kind! {
            self,
            InlineCtxt(_span) => None,
            InlineParent(span) => Some(LocalDefId { local_def_index: DefIndex::from_u16(span.parent) }),
            PartiallyInterned(span) => interned_parent(span.index),
            Interned(span) => interned_parent(span.index),
        }
    }
}

#[derive(Default)]
pub(crate) struct SpanInterner {
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
