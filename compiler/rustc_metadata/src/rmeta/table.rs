use crate::rmeta::*;

use rustc_data_structures::fingerprint::Fingerprint;
use rustc_hir::def::{CtorKind, CtorOf};
use rustc_index::Idx;
use rustc_middle::ty::{ParameterizedOverTcx, UnusedGenericParams};
use rustc_serialize::opaque::FileEncoder;
use rustc_serialize::Encoder as _;
use rustc_span::hygiene::MacroKind;
use std::marker::PhantomData;
use std::num::NonZeroUsize;

/// Table elements in the rmeta format must have fixed size, but we also want to encode offsets in
/// the file inside of table elements. If we use 4-byte offsets, it is too easy for crates to run
/// over that limit; see #112934. Switching to an 8-byte offset increases the size of some rlibs by 30%.
/// So for the time being we are using 5 bytes, which lets us encode offsets as large as 1 TB. It
/// seems unlikely that anyone will need offsets larger than that, but if you do, simply adjust
/// this constant.
const OFFSET_SIZE_BYTES: usize = 5;

#[derive(Default)]
struct Offset(usize);

impl Offset {
    // We technically waste 1 byte per offset if the compiler is compiled for a 32-bit target, but
    // that waste keeps the format described in this module portable.
    #[cfg(target_pointer_width = "32")]
    const MAX: usize = usize::MAX;

    #[cfg(target_pointer_width = "64")]
    const MAX: usize = usize::MAX >> (8 - OFFSET_SIZE_BYTES);
}

#[derive(Debug)]
pub struct OffsetTooLarge;

impl TryFrom<usize> for Offset {
    type Error = OffsetTooLarge;

    #[inline]
    fn try_from(value: usize) -> Result<Self, Self::Error> {
        if value > Self::MAX { Err(OffsetTooLarge) } else { Ok(Self(value)) }
    }
}

impl From<Offset> for usize {
    #[inline]
    fn from(v: Offset) -> usize {
        v.0
    }
}

impl FixedSizeEncoding for Offset {
    type ByteArray = [u8; OFFSET_SIZE_BYTES];

    #[inline]
    fn from_bytes(b: &Self::ByteArray) -> Self {
        let mut buf = [0u8; std::mem::size_of::<usize>()];
        buf[..OFFSET_SIZE_BYTES].copy_from_slice(b);
        let inner = usize::from_le_bytes(buf);
        debug_assert!(inner <= Self::MAX);
        Self(inner)
    }

    #[inline]
    fn write_to_bytes(self, b: &mut Self::ByteArray) {
        b.copy_from_slice(&self.0.to_le_bytes()[..OFFSET_SIZE_BYTES]);
    }
}

impl IsDefault for Offset {
    fn is_default(&self) -> bool {
        self.0 == 0
    }
}

pub(super) trait IsDefault: Default {
    fn is_default(&self) -> bool;
}

impl<T> IsDefault for Option<T> {
    fn is_default(&self) -> bool {
        self.is_none()
    }
}

impl IsDefault for AttrFlags {
    fn is_default(&self) -> bool {
        self.is_empty()
    }
}

impl IsDefault for bool {
    fn is_default(&self) -> bool {
        !self
    }
}

impl IsDefault for u32 {
    fn is_default(&self) -> bool {
        *self == 0
    }
}

impl<T> IsDefault for LazyArray<T> {
    fn is_default(&self) -> bool {
        self.num_elems == 0
    }
}

impl IsDefault for DefPathHash {
    fn is_default(&self) -> bool {
        self.0 == Fingerprint::ZERO
    }
}

impl IsDefault for UnusedGenericParams {
    fn is_default(&self) -> bool {
        // UnusedGenericParams encodes the *un*usedness as a bitset.
        // This means that 0 corresponds to all bits used, which is indeed the default.
        let is_default = self.bits() == 0;
        debug_assert_eq!(is_default, self.all_used());
        is_default
    }
}

/// Helper trait, for encoding to, and decoding from, a fixed number of bytes.
/// Used mainly for Lazy positions and lengths.
/// Unchecked invariant: `Self::default()` should encode as `[0; BYTE_LEN]`,
/// but this has no impact on safety.
pub(super) trait FixedSizeEncoding: IsDefault {
    /// This should be `[u8; BYTE_LEN]`;
    /// Cannot use an associated `const BYTE_LEN: usize` instead due to const eval limitations.
    type ByteArray;

    fn from_bytes(b: &Self::ByteArray) -> Self;
    fn write_to_bytes(self, b: &mut Self::ByteArray);
}

impl FixedSizeEncoding for u32 {
    type ByteArray = [u8; 4];

    #[inline]
    fn from_bytes(b: &[u8; 4]) -> Self {
        Self::from_le_bytes(*b)
    }

    #[inline]
    fn write_to_bytes(self, b: &mut [u8; 4]) {
        *b = self.to_le_bytes();
    }
}

macro_rules! fixed_size_enum {
    ($ty:ty { $(($($pat:tt)*))* }) => {
        impl FixedSizeEncoding for Option<$ty> {
            type ByteArray = [u8;1];

            #[inline]
            fn from_bytes(b: &[u8;1]) -> Self {
                use $ty::*;
                if b[0] == 0 {
                    return None;
                }
                match b[0] - 1 {
                    $(${index()} => Some($($pat)*),)*
                    _ => panic!("Unexpected {} code: {:?}", stringify!($ty), b[0]),
                }
            }

            #[inline]
            fn write_to_bytes(self, b: &mut [u8;1]) {
                use $ty::*;
                b[0] = match self {
                    None => unreachable!(),
                    $(Some($($pat)*) => 1 + ${index()},)*
                }
            }
        }
    }
}

fixed_size_enum! {
    DefKind {
        ( Mod                                      )
        ( Struct                                   )
        ( Union                                    )
        ( Enum                                     )
        ( Variant                                  )
        ( Trait                                    )
        ( TyAlias                                  )
        ( ForeignTy                                )
        ( TraitAlias                               )
        ( AssocTy                                  )
        ( TyParam                                  )
        ( Fn                                       )
        ( Const                                    )
        ( ConstParam                               )
        ( AssocFn                                  )
        ( AssocConst                               )
        ( ExternCrate                              )
        ( Use                                      )
        ( ForeignMod                               )
        ( AnonConst                                )
        ( InlineConst                              )
        ( OpaqueTy                                 )
        ( ImplTraitPlaceholder                     )
        ( Field                                    )
        ( LifetimeParam                            )
        ( GlobalAsm                                )
        ( Impl { of_trait: false }                 )
        ( Impl { of_trait: true }                  )
        ( Closure                                  )
        ( Generator                                )
        ( Static(ast::Mutability::Not)             )
        ( Static(ast::Mutability::Mut)             )
        ( Ctor(CtorOf::Struct, CtorKind::Fn)       )
        ( Ctor(CtorOf::Struct, CtorKind::Const)    )
        ( Ctor(CtorOf::Variant, CtorKind::Fn)      )
        ( Ctor(CtorOf::Variant, CtorKind::Const)   )
        ( Macro(MacroKind::Bang)                   )
        ( Macro(MacroKind::Attr)                   )
        ( Macro(MacroKind::Derive)                 )
    }
}

fixed_size_enum! {
    ty::ImplPolarity {
        ( Positive    )
        ( Negative    )
        ( Reservation )
    }
}

fixed_size_enum! {
    hir::Constness {
        ( NotConst )
        ( Const    )
    }
}

fixed_size_enum! {
    hir::Defaultness {
        ( Final                        )
        ( Default { has_value: false } )
        ( Default { has_value: true }  )
    }
}

fixed_size_enum! {
    hir::IsAsync {
        ( NotAsync )
        ( Async    )
    }
}

fixed_size_enum! {
    ty::AssocItemContainer {
        ( TraitContainer )
        ( ImplContainer  )
    }
}

fixed_size_enum! {
    MacroKind {
        ( Attr   )
        ( Bang   )
        ( Derive )
    }
}

// We directly encode `DefPathHash` because a `LazyValue` would incur a 25% cost.
impl FixedSizeEncoding for DefPathHash {
    type ByteArray = [u8; 16];

    #[inline]
    fn from_bytes(b: &[u8; 16]) -> Self {
        DefPathHash(Fingerprint::from_le_bytes(*b))
    }

    #[inline]
    fn write_to_bytes(self, b: &mut [u8; 16]) {
        debug_assert!(!self.is_default());
        *b = self.0.to_le_bytes();
    }
}

// We directly encode RawDefId because using a `LazyValue` would incur a 50% overhead in the worst case.
impl FixedSizeEncoding for Option<RawDefId> {
    type ByteArray = [u8; 8];

    #[inline]
    fn from_bytes(b: &[u8; 8]) -> Self {
        let krate = u32::from_le_bytes(b[0..4].try_into().unwrap());
        if krate == 0 {
            return None;
        }
        let index = u32::from_le_bytes(b[4..8].try_into().unwrap());
        Some(RawDefId { krate: krate - 1, index })
    }

    #[inline]
    fn write_to_bytes(self, b: &mut [u8; 8]) {
        match self {
            None => unreachable!(),
            Some(RawDefId { krate, index }) => {
                // CrateNum is less than `CrateNum::MAX_AS_U32`.
                debug_assert!(krate < u32::MAX);
                b[0..4].copy_from_slice(&(1 + krate).to_le_bytes());
                b[4..8].copy_from_slice(&index.to_le_bytes());
            }
        }
    }
}

impl FixedSizeEncoding for AttrFlags {
    type ByteArray = [u8; 1];

    #[inline]
    fn from_bytes(b: &[u8; 1]) -> Self {
        AttrFlags::from_bits_truncate(b[0])
    }

    #[inline]
    fn write_to_bytes(self, b: &mut [u8; 1]) {
        debug_assert!(!self.is_default());
        b[0] = self.bits();
    }
}

impl FixedSizeEncoding for bool {
    type ByteArray = [u8; 1];

    #[inline]
    fn from_bytes(b: &[u8; 1]) -> Self {
        b[0] != 0
    }

    #[inline]
    fn write_to_bytes(self, b: &mut [u8; 1]) {
        debug_assert!(!self.is_default());
        b[0] = self as u8
    }
}

impl FixedSizeEncoding for UnusedGenericParams {
    type ByteArray = [u8; 4];

    #[inline]
    fn from_bytes(b: &[u8; 4]) -> Self {
        let x: u32 = u32::from_bytes(b);
        UnusedGenericParams::from_bits(x)
    }

    #[inline]
    fn write_to_bytes(self, b: &mut [u8; 4]) {
        self.bits().write_to_bytes(b);
    }
}

impl<T> FixedSizeEncoding for Option<LazyValue<T>> {
    type ByteArray = [u8; OFFSET_SIZE_BYTES];

    #[inline]
    fn from_bytes(b: &Self::ByteArray) -> Self {
        let position = NonZeroUsize::new(Offset::from_bytes(b).try_into().unwrap())?;
        Some(LazyValue::from_position(position))
    }

    #[inline]
    fn write_to_bytes(self, b: &mut Self::ByteArray) {
        match self {
            None => unreachable!(),
            Some(lazy) => {
                let position = lazy.position.get();
                let position: Offset = position.try_into().unwrap();
                position.write_to_bytes(b)
            }
        }
    }
}

impl<T> LazyArray<T> {
    #[inline]
    fn write_to_bytes_impl(self, b: &mut [u8; OFFSET_SIZE_BYTES * 2]) {
        let ([position_bytes, meta_bytes],[])= b.as_chunks_mut::<OFFSET_SIZE_BYTES>() else { panic!() };

        let position = self.position.get();
        let position: Offset = position.try_into().unwrap();
        position.write_to_bytes(position_bytes);

        let len = self.num_elems;
        let len: Offset = len.try_into().unwrap();
        len.write_to_bytes(meta_bytes);
    }

    fn from_bytes_impl(
        position_bytes: &[u8; OFFSET_SIZE_BYTES],
        meta_bytes: &[u8; OFFSET_SIZE_BYTES],
    ) -> Option<LazyArray<T>> {
        let position = NonZeroUsize::new(Offset::from_bytes(position_bytes).0)?;
        let len = Offset::from_bytes(meta_bytes).0;
        Some(LazyArray::from_position_and_num_elems(position, len))
    }
}

impl<T> FixedSizeEncoding for LazyArray<T> {
    type ByteArray = [u8; OFFSET_SIZE_BYTES * 2];

    #[inline]
    fn from_bytes(b: &Self::ByteArray) -> Self {
        let ([position_bytes, meta_bytes],[])= b.as_chunks::<OFFSET_SIZE_BYTES>() else { panic!() };
        if *meta_bytes == [0u8; OFFSET_SIZE_BYTES] {
            return Default::default();
        }
        LazyArray::from_bytes_impl(position_bytes, meta_bytes).unwrap()
    }

    #[inline]
    fn write_to_bytes(self, b: &mut Self::ByteArray) {
        assert!(!self.is_default());
        self.write_to_bytes_impl(b)
    }
}

impl<T> FixedSizeEncoding for Option<LazyArray<T>> {
    type ByteArray = [u8; OFFSET_SIZE_BYTES * 2];

    #[inline]
    fn from_bytes(b: &Self::ByteArray) -> Self {
        let ([position_bytes, meta_bytes],[])= b.as_chunks::<OFFSET_SIZE_BYTES>() else { panic!() };
        LazyArray::from_bytes_impl(position_bytes, meta_bytes)
    }

    #[inline]
    fn write_to_bytes(self, b: &mut Self::ByteArray) {
        match self {
            None => unreachable!(),
            Some(lazy) => lazy.write_to_bytes_impl(b),
        }
    }
}

/// Helper for constructing a table's serialization (also see `Table`).
pub(super) struct TableBuilder<I: Idx, T: FixedSizeEncoding> {
    blocks: IndexVec<I, T::ByteArray>,
    _marker: PhantomData<T>,
}

impl<I: Idx, T: FixedSizeEncoding> Default for TableBuilder<I, T> {
    fn default() -> Self {
        TableBuilder { blocks: Default::default(), _marker: PhantomData }
    }
}

impl<I: Idx, const N: usize, T> TableBuilder<I, Option<T>>
where
    Option<T>: FixedSizeEncoding<ByteArray = [u8; N]>,
{
    pub(crate) fn set_some(&mut self, i: I, value: T) {
        self.set(i, Some(value))
    }
}

impl<I: Idx, const N: usize, T: FixedSizeEncoding<ByteArray = [u8; N]>> TableBuilder<I, T> {
    /// Sets the table value if it is not default.
    /// ATTENTION: For optimization default values are simply ignored by this function, because
    /// right now metadata tables never need to reset non-default values to default. If such need
    /// arises in the future then a new method (e.g. `clear` or `reset`) will need to be introduced
    /// for doing that explicitly.
    pub(crate) fn set(&mut self, i: I, value: T) {
        if !value.is_default() {
            // FIXME(eddyb) investigate more compact encodings for sparse tables.
            // On the PR @michaelwoerister mentioned:
            // > Space requirements could perhaps be optimized by using the HAMT `popcnt`
            // > trick (i.e. divide things into buckets of 32 or 64 items and then
            // > store bit-masks of which item in each bucket is actually serialized).
            let block = self.blocks.ensure_contains_elem(i, || [0; N]);
            value.write_to_bytes(block);
        }
    }

    pub(crate) fn encode(&self, buf: &mut FileEncoder) -> LazyTable<I, T> {
        let pos = buf.position();
        for block in &self.blocks {
            buf.emit_raw_bytes(block);
        }
        let num_bytes = self.blocks.len() * N;
        LazyTable::from_position_and_encoded_size(
            NonZeroUsize::new(pos as usize).unwrap(),
            num_bytes,
        )
    }
}

impl<I: Idx, const N: usize, T: FixedSizeEncoding<ByteArray = [u8; N]> + ParameterizedOverTcx>
    LazyTable<I, T>
where
    for<'tcx> T::Value<'tcx>: FixedSizeEncoding<ByteArray = [u8; N]>,
{
    /// Given the metadata, extract out the value at a particular index (if any).
    #[inline(never)]
    pub(super) fn get<'a, 'tcx, M: Metadata<'a, 'tcx>>(&self, metadata: M, i: I) -> T::Value<'tcx> {
        trace!("LazyTable::lookup: index={:?} len={:?}", i, self.encoded_size);

        let start = self.position.get();
        let bytes = &metadata.blob()[start..start + self.encoded_size];
        let (bytes, []) = bytes.as_chunks::<N>() else { panic!() };
        bytes.get(i.index()).map_or_else(Default::default, FixedSizeEncoding::from_bytes)
    }

    /// Size of the table in entries, including possible gaps.
    pub(super) fn size(&self) -> usize {
        self.encoded_size / N
    }
}
