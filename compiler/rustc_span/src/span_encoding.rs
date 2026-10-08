use std::cmp::Ordering;
use std::hash::Hash;

use rustc_data_structures::fx::FxIndexSet;
use rustc_data_structures::stable_hash::RawSpan;
// This code is very hot and uses lots of arithmetic, avoid overflow checks for performance.
// See https://github.com/rust-lang/rust/pull/119440#issuecomment-1874255727
use rustc_serialize::int_overflow::DebugStrictAdd;

use crate::def_id::{DefIndex, LocalDefId};
use crate::hygiene::SyntaxContext;
use crate::{BytePos, SPAN_TRACK, SpanData};

/// A compressed span.
///
/// [`SpanData`] is 16 bytes, which is too big to stick everywhere. `Span` is
/// only 8 bytes. A large majority of `SpanData` instances can be made to fit
/// within those 8 bytes. Any `SpanData` whose fields don't fit into a `Span`
/// are stored in a separate interner table, and the `Span` will index into
/// that table.
///
/// An earlier version of this code used only 4 bytes for `Span`, but that was
/// slower because many fewer spans could be stored inline and the interner was
/// used a lot more. That version of the code also predated the storage of
/// parents.
///
/// Experiments with uncompressed spans yielded worse performance in most cases
/// because memory usage and cache miss rates are significantly higher without
/// compression.
///
/// There are four different span formats and each packs data into the 8 bytes
/// in a different way. A variable-length prefix in the high bits identifies
/// which format is being used. It is an invariant that we always use the first
/// listed format whose requirements are satisfied. `len` is `hi - lo`.
///
/// - Inline-context format (requires 15-bit length, 16-bit context, and no parent):
///
///     `[ prefix=0 | len:15 | ctxt:16 | lo:32 ]`
///
/// - Inline-parent format (requires 14-bit length, root context, and 15-bit parent):
///
///     `[ prefix=110 | len:14 | parent:15 | lo:32 ]`
///
/// - Inline-pair format (requires 24-bit lo, 7-bit length, 15-bit context, 16-bit parent):
///
///     `[ prefix=10 | len:7 | ctxt:15 | parent:16 | lo:24 ]`
///
/// - Interned format (all cases not covered above):
///
///     `[ prefix=111 | ctxt:29 | index:32 ]`
///
/// Notes about this design.
///
/// - This configuration was the best one found after many measurements of
///   real-world crates, and replaced an earlier 8-byte design.
///
/// - The size of `lo` depends on the size of the crate and its dependencies.
///   Crates whose `lo` size exceeds 24-bits are extremely rare.
///
/// - The most common numbers of bits needed for `len` are from 0 to 7,
///   with a peak usually at 3 or 4, and then it drops off quickly from 8
///   onwards. Lengths larger than 14 bits are rare, but many crates will have
///   a small number of such lengths (sometimes 20+ bits).
///
/// - The number of bits needed for `ctxt` and `parent` values depend partly on
///   the crate size and partly on the form of the code. For both fields,
///   values needing more than 15 bits are very rare, and most crates won't hit
///   these limits.
///
/// - The ctxt is stored inline in every format (Every ctxt must fit in
///   29 bits; if we ever hit that we'd have many gigabytes of contexts, and
///   certainly would have OOM'd anyway.) This means reading the ctxt never
///   requires checking the interner. This is good because the ctxt is often
///   consulted by itself. It also means we don't need to store the ctxt in the
///   interner table, which we achieve by using `SpanDataNoCtxt`, which
///   minimizes the number of unique values that need to be interned.
///
/// - The use of `repr(packed(4))` means `Span` has an alignment of 4 bytes,
///   which keeps many types that contain spans (e.g. AST nodes) smaller.
///
/// In order to reliably use parented spans in incremental compilation,
/// accesses to `lo` and `hi` must introduce a dependency to the parent definition's span.
/// This is performed using the callback `SPAN_TRACK` to access the query engine.
#[derive(Clone, Copy, Eq, PartialEq, Hash)]
#[rustc_pass_by_value]
#[repr(packed(4))]
pub struct Span(u64);

// `SyntaxContext` (`SpanData::ctxt`) and `LocalDefId` (within `SpanData::parent`) aren't orderable.
// If you want to order spans just on `lo`/`hi`, use explicit comparisons involving `Span::lo_hi`.
// Or you can wrap your span within `OrdSpan` which impls `PartialOrd`/`Ord` via `Span::lo_hi`.
impl !PartialOrd for Span {}
impl !Ord for Span {}

/// Specifies a field within one of the span formats. Provides const operations that (a) let a span
/// format be fully specified, and (b) check/read/write the field for/from/to a span.
#[derive(Copy, Clone)]
struct SpanField {
    /// The field's bit position, where zero means it's in the lowest bits.
    shift: u32,
    /// The field's width, in bits.
    width: u32,
}

impl SpanField {
    /// Specifies the top field, i.e. the highest bits. Chained with `below`.
    #[inline]
    const fn top(width: u32) -> Self {
        Self { shift: 64 - width, width }
    }

    /// Specifies a field that comes below the field in `self`. Chainable.
    #[inline]
    const fn below(self, width: u32) -> Self {
        Self { shift: self.shift - width, width }
    }

    /// The mask for the (downshifted to 0..n) field.
    #[inline]
    const fn mask(self) -> u64 {
        (1 << self.width) - 1
    }

    /// Gets the field from a span-as-`u64` as a `u32`.
    #[inline]
    const fn get(self, u: u64) -> u32 {
        ((u >> self.shift) & self.mask()) as u32
    }

    /// Sets the field to `v` in an otherwise-zero span-as-`u64`.
    #[inline]
    const fn set(self, v: u32) -> u64 {
        debug_assert!(v <= self.mask() as u32);
        (v as u64) << self.shift
    }

    /// Returns non-zero if `v` is too big to fit in the field.
    #[inline]
    const fn too_big(self, v: u32) -> u32 {
        v & !(self.mask() as u32)
    }

    /// Asserts that a field uses the bottom-most bits.
    #[inline]
    const fn assert_is_bottom(self) {
        assert!(self.shift == 0);
    }
}

// Convenience structures for all span formats.
#[derive(Clone, Copy)]
struct InlineCtxt {
    lo: u32,
    len: u32,
    ctxt: u32,
}

#[derive(Clone, Copy)]
struct InlineParent {
    lo: u32,
    len: u32,
    parent: u32,
}

#[derive(Clone, Copy)]
struct InlinePair {
    lo: u32,
    len: u32,
    ctxt: u32,
    parent: u32,
}

#[derive(Clone, Copy)]
struct Interned {
    ctxt: u32,
    index: u32,
}

impl InlineCtxt {
    const PREFIX_VALUE: u32 = 0b0;
    const PREFIX: SpanField = SpanField::top(1);
    const LEN: SpanField = Self::PREFIX.below(15);
    const CTXT: SpanField = Self::LEN.below(16);
    const LO: SpanField = Self::CTXT.below(32);

    #[inline]
    const fn try_new_span(lo: u32, len: u32, ctxt: u32) -> Option<Span> {
        if Self::LEN.too_big(len) | Self::CTXT.too_big(ctxt) == 0 {
            Some(Span(
                Self::PREFIX.set(Self::PREFIX_VALUE)
                    | Self::LEN.set(len)
                    | Self::CTXT.set(ctxt)
                    | Self::LO.set(lo),
            ))
        } else {
            None
        }
    }

    #[inline]
    fn data(self) -> SpanData {
        SpanData {
            lo: BytePos(self.lo),
            hi: BytePos(self.lo.debug_strict_add(self.len)),
            ctxt: SyntaxContext::from_u32(self.ctxt),
            parent: None,
        }
    }

    #[inline]
    fn from_span(span: Span) -> InlineCtxt {
        let u = span.0;
        Self { lo: Self::LO.get(u), len: Self::LEN.get(u), ctxt: Self::CTXT.get(u) }
    }
}
const _: () = InlineCtxt::LO.assert_is_bottom();

impl InlineParent {
    const PREFIX_VALUE: u32 = 0b110;
    const PREFIX: SpanField = SpanField::top(3);
    const LEN: SpanField = Self::PREFIX.below(14);
    const PARENT: SpanField = Self::LEN.below(15);
    const LO: SpanField = Self::PARENT.below(32);

    #[inline]
    fn try_new_span(lo: u32, len: u32, ctxt: u32, parent: u32) -> Option<Span> {
        if Self::LEN.too_big(len) | ctxt | Self::PARENT.too_big(parent) == 0 {
            Some(Span(
                Self::PREFIX.set(Self::PREFIX_VALUE)
                    | Self::LEN.set(len)
                    | Self::PARENT.set(parent)
                    | Self::LO.set(lo),
            ))
        } else {
            None
        }
    }

    #[inline]
    fn data(self) -> SpanData {
        SpanData {
            lo: BytePos(self.lo),
            hi: BytePos(self.lo.debug_strict_add(self.len)),
            ctxt: SyntaxContext::root(),
            parent: Some(LocalDefId { local_def_index: DefIndex::from_u32(self.parent) }),
        }
    }

    #[inline]
    fn from_span(span: Span) -> InlineParent {
        let u = span.0;
        Self { lo: Self::LO.get(u), len: Self::LEN.get(u), parent: Self::PARENT.get(u) }
    }
}
const _: () = InlineParent::LO.assert_is_bottom();

impl InlinePair {
    const PREFIX_VALUE: u32 = 0b10;
    const PREFIX: SpanField = SpanField::top(2);
    const LEN: SpanField = Self::PREFIX.below(7);
    const CTXT: SpanField = Self::LEN.below(15);
    const PARENT: SpanField = Self::CTXT.below(16);
    const LO: SpanField = Self::PARENT.below(24);

    #[inline]
    fn try_new_span(lo: u32, len: u32, ctxt: u32, parent: u32) -> Option<Span> {
        if Self::LO.too_big(lo)
            | Self::LEN.too_big(len)
            | Self::CTXT.too_big(ctxt)
            | Self::PARENT.too_big(parent)
            == 0
        {
            Some(Span(
                Self::PREFIX.set(Self::PREFIX_VALUE)
                    | Self::LEN.set(len)
                    | Self::CTXT.set(ctxt)
                    | Self::PARENT.set(parent)
                    | Self::LO.set(lo),
            ))
        } else {
            None
        }
    }

    #[inline]
    fn data(self) -> SpanData {
        SpanData {
            lo: BytePos(self.lo),
            hi: BytePos(self.lo.debug_strict_add(self.len)),
            ctxt: SyntaxContext::from_u32(self.ctxt),
            parent: Some(LocalDefId { local_def_index: DefIndex::from_u32(self.parent) }),
        }
    }

    #[inline]
    fn from_span(span: Span) -> InlinePair {
        let u = span.0;
        Self {
            lo: Self::LO.get(u),
            len: Self::LEN.get(u),
            ctxt: Self::CTXT.get(u),
            parent: Self::PARENT.get(u),
        }
    }
}
const _: () = InlinePair::LO.assert_is_bottom();

impl Interned {
    const PREFIX_VALUE: u32 = 0b111;
    const PREFIX: SpanField = SpanField::top(3);
    const CTXT: SpanField = Self::PREFIX.below(29);
    const INDEX: SpanField = Self::CTXT.below(32);

    #[inline]
    fn new_span(ctxt: u32, index: u32) -> Span {
        Span(Self::PREFIX.set(Self::PREFIX_VALUE) | Self::CTXT.set(ctxt) | Self::INDEX.set(index))
    }

    #[inline]
    fn data(self) -> SpanData {
        let SpanDataNoCtxt { lo, hi, parent } =
            with_span_interner(|interner| interner.spans[self.index as usize]);
        SpanData { lo, hi, ctxt: SyntaxContext::from_u32(self.ctxt), parent }
    }

    #[inline]
    fn from_span(span: Span) -> Interned {
        let u = span.0;
        Self { ctxt: Self::CTXT.get(u), index: Self::INDEX.get(u) }
    }
}
const _: () = Interned::INDEX.assert_is_bottom();

// This code is very hot, and converting span to an enum and matching on it doesn't optimize away
// properly. So we are using a macro emulating such a match, but expand it directly to an if-else
// chain.
macro_rules! match_span_kind {
    (
        $span:expr,
        InlineCtxt($span1:ident) => $arm1:expr,
        InlineParent($span2:ident) => $arm2:expr,
        InlinePair($span3:ident) => $arm3:expr,
        Interned($span4:ident) => $arm4:expr,
    ) => {{
        let span64 = $span.0;
        if InlineCtxt::PREFIX.get(span64) == InlineCtxt::PREFIX_VALUE {
            let $span1 = InlineCtxt::from_span($span);
            $arm1
        } else if InlineParent::PREFIX.get(span64) == InlineParent::PREFIX_VALUE {
            let $span2 = InlineParent::from_span($span);
            $arm2
        } else if InlinePair::PREFIX.get(span64) == InlinePair::PREFIX_VALUE {
            let $span3 = InlinePair::from_span($span);
            $arm3
        } else {
            let $span4 = Interned::from_span($span);
            $arm4
        }
    }};
}

/// The dummy span has zero position, length, and context, and no parent.
pub const DUMMY_SP: Span = InlineCtxt::try_new_span(0, 0, 0).unwrap();

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
        Span::new_ordered(lo, hi, ctxt, parent)
    }

    #[inline]
    pub fn new_ordered(
        lo: BytePos,
        hi: BytePos,
        ctxt: SyntaxContext,
        parent: Option<LocalDefId>,
    ) -> Self {
        #[cold]
        #[inline(never)]
        fn interned(lo: BytePos, hi: BytePos, ctxt: u32, parent: Option<LocalDefId>) -> Span {
            // Interned.
            assert!(Interned::CTXT.too_big(ctxt) == 0); // this is a hard limit, with no fallback
            let index =
                with_span_interner(|interner| interner.intern(&SpanDataNoCtxt { lo, hi, parent }));
            Interned::new_span(ctxt, index)
        }

        let lo32 = lo.0;
        let (len, ctxt32) = (hi.0 - lo32, ctxt.as_u32());
        if let Some(parent) = parent {
            if let Some(span) = Span::try_new_span_with_parent(lo32, len, ctxt32, parent) {
                // InlineParent or InlinePair.
                return span;
            }
        } else if let Some(span) = InlineCtxt::try_new_span(lo32, len, ctxt32) {
            // InlineCtxt.
            return span;
        }

        interned(lo, hi, ctxt32, parent)
    }

    /// Tries to create an `InlineParent` or `InlinePair` format span. Returns `None` if the fields
    /// won't fit.
    #[inline]
    fn try_new_span_with_parent(lo: u32, len: u32, ctxt: u32, parent: LocalDefId) -> Option<Self> {
        let parent = parent.local_def_index.as_u32();
        InlineParent::try_new_span(lo, len, ctxt, parent)
            .or_else(|| InlinePair::try_new_span(lo, len, ctxt, parent))
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
            InlinePair(span) => span.data(),
            Interned(span) => span.data(),
        }
    }

    /// Returns `true` if this span comes from any kind of macro, desugaring or inlining.
    #[inline]
    pub fn from_expansion(self) -> bool {
        !self.ctxt().is_root()
    }

    /// Returns `true` if this is a dummy span with lo=0, hi=0. Any context value is allowed, which
    /// means it's not the same as an equality comparison with `DUMMY_SP`.
    #[inline]
    pub fn is_dummy(self) -> bool {
        match_span_kind! {
            self,
            InlineCtxt(span) => span.lo == 0 && span.len == 0,
            InlineParent(span) => span.lo == 0 && span.len == 0,
            InlinePair(span) => span.lo == 0 && span.len == 0,
            Interned(span) => {
                let data = with_span_interner(|interner| interner.spans[span.index as usize]);
                data.lo == BytePos(0) && data.hi == BytePos(0)
            },
        }
    }

    #[inline]
    pub fn map_ctxt(self, map: impl FnOnce(SyntaxContext) -> SyntaxContext) -> Span {
        let data = match_span_kind! {
            self,
            InlineCtxt(span) => {
                // This format occurs 1-2 orders of magnitude more often than others (#125017),
                // so it makes sense to micro-optimize it to avoid `span.data()` and `Span::new()`.
                let new_ctxt = map(SyntaxContext::from_u32(span.ctxt));
                let new_ctxt32 = new_ctxt.as_u32();
                return if let Some(span) = InlineCtxt::try_new_span(span.lo, span.len, new_ctxt32) {
                    span
                } else {
                    span.data().with_ctxt(new_ctxt)
                };
            },
            InlineParent(span) => span.data(),
            InlinePair(span) => span.data(),
            Interned(span) => span.data(),
        };

        data.with_ctxt(map(data.ctxt))
    }

    /// This function is used as a fast path when decoding the full `SpanData` is not necessary.
    /// It's a cut-down version of `data_untracked` and doesn't require taking the interner lock.
    #[cfg_attr(not(test), rustc_diagnostic_item = "SpanCtxt")]
    #[inline]
    pub fn ctxt(self) -> SyntaxContext {
        let ctxt32 = match_span_kind! {
            self,
            InlineCtxt(span) => span.ctxt,
            InlineParent(_span) => return SyntaxContext::root(),
            InlinePair(span) => span.ctxt,
            Interned(span) => span.ctxt,
        };
        SyntaxContext::from_u32(ctxt32)
    }

    // Allow `span_use_eq_ctxt` because this is `eq_ctxt`!
    //
    // FIXME(nnethercote): context equality used to be much more complex, but the span
    // representation changed and now simple equality is fine. This method and the
    // `span_use_eq_ctxt` lint and the `SpanCtxt` diagnostic item can be removed.
    #[allow(rustc::span_use_eq_ctxt)]
    #[inline]
    pub fn eq_ctxt(self, other: Span) -> bool {
        self.ctxt() == other.ctxt()
    }

    #[inline]
    pub fn with_parent(self, parent: Option<LocalDefId>) -> Span {
        let data = match_span_kind! {
            self,
            InlineCtxt(span) => {
                // This format occurs 1-2 orders of magnitude more often than others (#126544),
                // so it makes sense to micro-optimize it to avoid `span.data()`.
                match parent {
                    None => return self,
                    Some(parent)
                        if let Some(span) =
                            Span::try_new_span_with_parent(span.lo, span.len, span.ctxt, parent) =>
                    {
                        return span;
                    }
                    _ => span.data(),
                }
            },
            InlineParent(span) => span.data(),
            InlinePair(span) => span.data(),
            Interned(span) => span.data(),
        };

        if let Some(old_parent) = data.parent {
            (*SPAN_TRACK)(old_parent);
        }
        data.with_parent(parent)
    }

    #[inline]
    pub fn parent(self) -> Option<LocalDefId> {
        let to_parent = |parent| Some(LocalDefId { local_def_index: DefIndex::from_u32(parent) });
        match_span_kind! {
            self,
            InlineCtxt(_span) => None,
            InlineParent(span) => to_parent(span.parent),
            InlinePair(span) => to_parent(span.parent),
            Interned(span) => {
                with_span_interner(|interner| interner.spans[span.index as usize].parent)
            },
        }
    }

    #[inline]
    pub(crate) fn to_raw_span(self) -> RawSpan {
        RawSpan(self.0)
    }

    #[inline]
    pub fn from_raw_span(RawSpan(a): RawSpan) -> Span {
        Span(a)
    }
}

/// Span wrapper that provides equality and ordering based only on the `lo`/`hi` fields. Exists
/// because `Span` doesn't impl `PartialOrd`/`Ord` due to the `ctxt` and `parent` fields being
/// unorderable. Useful when storing things in source code order, e.g. in a `BTreeMap<OrdSpan, T>`.
#[derive(Clone, Copy, Debug)]
pub struct OrdSpan(pub Span);

impl PartialEq for OrdSpan {
    fn eq(&self, rhs: &Self) -> bool {
        // Ignores `ctxt` and `parent` because they are unorderable.
        self.0.lo_hi() == rhs.0.lo_hi()
    }
}

impl Eq for OrdSpan {}

impl PartialOrd for OrdSpan {
    fn partial_cmp(&self, rhs: &Self) -> Option<Ordering> {
        Some(self.cmp(rhs))
    }
}

impl Ord for OrdSpan {
    fn cmp(&self, rhs: &Self) -> Ordering {
        // Ignores `ctxt` and `parent` because they are unorderable.
        self.0.lo_hi().cmp(&rhs.0.lo_hi())
    }
}

// `OrdSpan` is typically stored in types that rely on ordering, such as `BTreeMap` or `SortedMap`.
// Hashing shouldn't be necessary.
impl !Hash for OrdSpan {}

/// `SpanData` minus the `ctxt` field. This is what we actually intern, because we can always store
/// ctxt inline in `Span`.
#[derive(Clone, Copy, Hash, PartialEq, Eq)]
struct SpanDataNoCtxt {
    lo: BytePos,
    hi: BytePos,
    parent: Option<LocalDefId>,
}

#[derive(Default)]
pub(crate) struct SpanInterner {
    spans: FxIndexSet<SpanDataNoCtxt>,
}

impl SpanInterner {
    fn intern(&mut self, span_data_no_ctxt: &SpanDataNoCtxt) -> u32 {
        let (index, _) = self.spans.insert_full(*span_data_no_ctxt);
        index as u32
    }
}

// If an interner exists, return it. Otherwise, prepare a fresh one.
#[inline]
fn with_span_interner<T, F: FnOnce(&mut SpanInterner) -> T>(f: F) -> T {
    crate::with_session_globals(|session_globals| f(&mut session_globals.span_interner.lock()))
}
