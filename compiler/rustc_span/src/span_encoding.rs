// Spans are encoded using 1-bit tag and 2 different encoding formats (one for each tag value).
// One format is used for keeping span data inline,
// another contains index into an out-of-line span interner.
// The encoding format for inline spans were obtained by optimizing over crates in rustc/libstd.
// See https://internals.rust-lang.org/t/rfc-compiler-refactoring-spans/1357/28

use crate::def_id::LocalDefId;
use crate::hygiene::SyntaxContext;
use crate::SPAN_TRACK;
use crate::{BytePos, SpanData};

use rustc_data_structures::fx::FxIndexSet;

/// A compressed span.
///
/// Whereas [`SpanData`] is 12 bytes, which is a bit too big to stick everywhere, `Span`
/// is a form that only takes up 8 bytes, with less space for the length and
/// context. The vast majority (99.9%+) of `SpanData` instances will fit within
/// those 8 bytes; any `SpanData` whose fields don't fit into a `Span` are
/// stored in a separate interner table, and the `Span` will index into that
/// table. Interning is rare enough that the cost is low, but common enough
/// that the code is exercised regularly.
///
/// An earlier version of this code used only 4 bytes for `Span`, but that was
/// slower because only 80--90% of spans could be stored inline (even less in
/// very large crates) and so the interner was used a lot more.
///
/// Inline (compressed) format:
/// - `span.base_or_index == span_data.lo`
/// - `span.len_or_tag == len == span_data.hi - span_data.lo` (must be `<= MAX_LEN`)
/// - `span.ctxt == span_data.ctxt` (must be `<= MAX_CTXT`)
///
/// Interned format:
/// - `span.base_or_index == index` (indexes into the interner table)
/// - `span.len_or_tag == LEN_TAG` (high bit set, all other bits are zero)
/// - `span.ctxt == 0`
///
/// The inline form uses 0 for the tag value (rather than 1) so that we don't
/// need to mask out the tag bit when getting the length, and so that the
/// dummy span can be all zeroes.
///
/// Notes about the choice of field sizes:
/// - `base` is 32 bits in both `Span` and `SpanData`, which means that `base`
///   values never cause interning. The number of bits needed for `base`
///   depends on the crate size. 32 bits allows up to 4 GiB of code in a crate.
/// - `len` is 15 bits in `Span` (a u16, minus 1 bit for the tag) and 32 bits
///   in `SpanData`, which means that large `len` values will cause interning.
///   The number of bits needed for `len` does not depend on the crate size.
///   The most common numbers of bits for `len` are from 0 to 7, with a peak usually
///   at 3 or 4, and then it drops off quickly from 8 onwards. 15 bits is enough
///   for 99.99%+ of cases, but larger values (sometimes 20+ bits) might occur
///   dozens of times in a typical crate.
/// - `ctxt` is 16 bits in `Span` and 32 bits in `SpanData`, which means that
///   large `ctxt` values will cause interning. The number of bits needed for
///   `ctxt` values depend partly on the crate size and partly on the form of
///   the code. No crates in `rustc-perf` need more than 15 bits for `ctxt`,
///   but larger crates might need more than 16 bits.
///
/// In order to reliably use parented spans in incremental compilation,
/// the dependency to the parent definition's span. This is performed
/// using the callback `SPAN_TRACK` to access the query engine.
///
#[derive(Clone, Copy, Eq, PartialEq, Hash)]
// FIXME(@lcnr): Enable this attribute once the bootstrap
// compiler knows of `rustc_pass_by_value`.
//
// Right now, this lint would only trigger when compiling the
// stage 2 compiler, which is fairly annoying as there are
// a lot of places using `&Span` right now. After the next bootstrap bump,
// the lint will already trigger when using stage 1, which is a lot less annoying.
//
// #[cfg_attr(not(bootstrap), rustc_pass_by_value)]
pub struct Span {
    base_or_index: u32,
    len_or_tag: u16,
    ctxt_or_zero: u16,
}

const LEN_TAG: u16 = 0b1000_0000_0000_0000;
const MAX_LEN: u32 = 0b0111_1111_1111_1111;
const MAX_CTXT: u32 = 0b1111_1111_1111_1111;

/// Dummy span, both position and length are zero, syntax context is zero as well.
pub const DUMMY_SP: Span = Span { base_or_index: 0, len_or_tag: 0, ctxt_or_zero: 0 };

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

        let (base, len, ctxt2) = (lo.0, hi.0 - lo.0, ctxt.as_u32());

        if len <= MAX_LEN && ctxt2 <= MAX_CTXT && parent.is_none() {
            // Inline format.
            Span { base_or_index: base, len_or_tag: len as u16, ctxt_or_zero: ctxt2 as u16 }
        } else {
            // Interned format.
            let index =
                with_span_interner(|interner| interner.intern(&SpanData { lo, hi, ctxt, parent }));
            Span { base_or_index: index, len_or_tag: LEN_TAG, ctxt_or_zero: 0 }
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
        if self.len_or_tag != LEN_TAG {
            // Inline format.
            debug_assert!(self.len_or_tag as u32 <= MAX_LEN);
            SpanData {
                lo: BytePos(self.base_or_index),
                hi: BytePos(self.base_or_index + self.len_or_tag as u32),
                ctxt: SyntaxContext::from_u32(self.ctxt_or_zero as u32),
                parent: None,
            }
        } else {
            // Interned format.
            debug_assert!(self.ctxt_or_zero == 0);
            let index = self.base_or_index;
            with_span_interner(|interner| interner.spans[index as usize])
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
    crate::with_session_globals(|session_globals| f(&mut *session_globals.span_interner.lock()))
}
