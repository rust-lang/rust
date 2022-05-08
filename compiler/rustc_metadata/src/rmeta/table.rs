use crate::rmeta::*;

use rustc_data_structures::fingerprint::Fingerprint;
use rustc_hir::def::{CtorKind, CtorOf};
use rustc_index::vec::Idx;
use rustc_serialize::opaque::Encoder;
use rustc_serialize::Encoder as _;
use rustc_span::hygiene::MacroKind;
use std::convert::TryInto;
use std::marker::PhantomData;
use std::num::NonZeroUsize;
use tracing::debug;

/// Helper trait, for encoding to, and decoding from, a fixed number of bytes.
/// Used mainly for Lazy positions and lengths.
/// Unchecked invariant: `Self::default()` should encode as `[0; BYTE_LEN]`,
/// but this has no impact on safety.
pub(super) trait FixedSizeEncoding: Default {
    /// This should be `[u8; BYTE_LEN]`;
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
                    _ => panic!("Unexpected ImplPolarity code: {:?}", b[0]),
                }
            }

            #[inline]
            fn write_to_bytes(self, b: &mut [u8;1]) {
                use $ty::*;
                b[0] = match self {
                    None => 0,
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
        ( Field                                    )
        ( LifetimeParam                            )
        ( GlobalAsm                                )
        ( Impl                                     )
        ( Closure                                  )
        ( Generator                                )
        ( Static(ast::Mutability::Not)             )
        ( Static(ast::Mutability::Mut)             )
        ( Ctor(CtorOf::Struct, CtorKind::Fn)       )
        ( Ctor(CtorOf::Struct, CtorKind::Const)    )
        ( Ctor(CtorOf::Struct, CtorKind::Fictive)  )
        ( Ctor(CtorOf::Variant, CtorKind::Fn)      )
        ( Ctor(CtorOf::Variant, CtorKind::Const)   )
        ( Ctor(CtorOf::Variant, CtorKind::Fictive) )
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

// We directly encode `DefPathHash` because a `Lazy` would encur a 25% cost.
impl FixedSizeEncoding for Option<DefPathHash> {
    type ByteArray = [u8; 16];

    #[inline]
    fn from_bytes(b: &[u8; 16]) -> Self {
        Some(DefPathHash(Fingerprint::from_le_bytes(*b)))
    }

    #[inline]
    fn write_to_bytes(self, b: &mut [u8; 16]) {
        let Some(DefPathHash(fingerprint)) = self else {
            panic!("Trying to encode absent DefPathHash.")
        };
        *b = fingerprint.to_le_bytes();
    }
}

// We directly encode RawDefId because using a `Lazy` would incur a 50% overhead in the worst case.
impl FixedSizeEncoding for Option<RawDefId> {
    type ByteArray = [u8; 8];

    #[inline]
    fn from_bytes(b: &[u8; 8]) -> Self {
        let krate = u32::from_le_bytes(b[0..4].try_into().unwrap());
        let index = u32::from_le_bytes(b[4..8].try_into().unwrap());
        if krate == 0 {
            return None;
        }
        Some(RawDefId { krate: krate - 1, index })
    }

    #[inline]
    fn write_to_bytes(self, b: &mut [u8; 8]) {
        match self {
            None => *b = [0; 8],
            Some(RawDefId { krate, index }) => {
                // CrateNum is less than `CrateNum::MAX_AS_U32`.
                debug_assert!(krate < u32::MAX);
                b[0..4].copy_from_slice(&(1 + krate).to_le_bytes());
                b[4..8].copy_from_slice(&index.to_le_bytes());
            }
        }
    }
}

impl FixedSizeEncoding for Option<()> {
    type ByteArray = [u8; 1];

    #[inline]
    fn from_bytes(b: &[u8; 1]) -> Self {
        (b[0] != 0).then(|| ())
    }

    #[inline]
    fn write_to_bytes(self, b: &mut [u8; 1]) {
        b[0] = self.is_some() as u8
    }
}

// NOTE(eddyb) there could be an impl for `usize`, which would enable a more
// generic `Lazy<T>` impl, but in the general case we might not need / want to
// fit every `usize` in `u32`.
impl<T> FixedSizeEncoding for Option<Lazy<T>> {
    type ByteArray = [u8; 4];

    #[inline]
    fn from_bytes(b: &[u8; 4]) -> Self {
        let position = NonZeroUsize::new(u32::from_bytes(b) as usize)?;
        Some(Lazy::from_position(position))
    }

    #[inline]
    fn write_to_bytes(self, b: &mut [u8; 4]) {
        let position = self.map_or(0, |lazy| lazy.position.get());
        let position: u32 = position.try_into().unwrap();
        position.write_to_bytes(b)
    }
}

impl<T> FixedSizeEncoding for Option<Lazy<[T]>> {
    type ByteArray = [u8; 8];

    #[inline]
    fn from_bytes(b: &[u8; 8]) -> Self {
        let ([ref position_bytes, ref meta_bytes],[])= b.as_chunks::<4>() else { panic!() };
        let position = NonZeroUsize::new(u32::from_bytes(position_bytes) as usize)?;
        let len = u32::from_bytes(meta_bytes) as usize;
        Some(Lazy::from_position_and_meta(position, len))
    }

    #[inline]
    fn write_to_bytes(self, b: &mut [u8; 8]) {
        let ([ref mut position_bytes, ref mut meta_bytes],[])= b.as_chunks_mut::<4>() else { panic!() };

        let position = self.map_or(0, |lazy| lazy.position.get());
        let position: u32 = position.try_into().unwrap();
        position.write_to_bytes(position_bytes);

        let len = self.map_or(0, |lazy| lazy.meta);
        let len: u32 = len.try_into().unwrap();
        len.write_to_bytes(meta_bytes);
    }
}

/// Random-access table (i.e. offering constant-time `get`/`set`), similar to
/// `Vec<Option<T>>`, but without requiring encoding or decoding all the values
/// eagerly and in-order.
/// A total of `(max_idx + 1)` times `Option<T> as FixedSizeEncoding>::ByteArray`
/// are used for a table, where `max_idx` is the largest index passed to
/// `TableBuilder::set`.
pub(super) struct Table<I: Idx, T>
where
    Option<T>: FixedSizeEncoding,
{
    _marker: PhantomData<(fn(&I), T)>,
    // NOTE(eddyb) this makes `Table` not implement `Sized`, but no
    // value of `Table` is ever created (it's always behind `Lazy`).
    _bytes: [u8],
}

/// Helper for constructing a table's serialization (also see `Table`).
pub(super) struct TableBuilder<I: Idx, T>
where
    Option<T>: FixedSizeEncoding,
{
    blocks: IndexVec<I, <Option<T> as FixedSizeEncoding>::ByteArray>,
    _marker: PhantomData<T>,
}

impl<I: Idx, T> Default for TableBuilder<I, T>
where
    Option<T>: FixedSizeEncoding,
{
    fn default() -> Self {
        TableBuilder { blocks: Default::default(), _marker: PhantomData }
    }
}

impl<I: Idx, T> TableBuilder<I, T>
where
    Option<T>: FixedSizeEncoding,
{
    pub(crate) fn set<const N: usize>(&mut self, i: I, value: T)
    where
        Option<T>: FixedSizeEncoding<ByteArray = [u8; N]>,
    {
        // FIXME(eddyb) investigate more compact encodings for sparse tables.
        // On the PR @michaelwoerister mentioned:
        // > Space requirements could perhaps be optimized by using the HAMT `popcnt`
        // > trick (i.e. divide things into buckets of 32 or 64 items and then
        // > store bit-masks of which item in each bucket is actually serialized).
        self.blocks.ensure_contains_elem(i, || [0; N]);
        Some(value).write_to_bytes(&mut self.blocks[i]);
    }

    pub(crate) fn encode<const N: usize>(&self, buf: &mut Encoder) -> Lazy<Table<I, T>>
    where
        Option<T>: FixedSizeEncoding<ByteArray = [u8; N]>,
    {
        let pos = buf.position();
        for block in &self.blocks {
            buf.emit_raw_bytes(block).unwrap();
        }
        let num_bytes = self.blocks.len() * N;
        Lazy::from_position_and_meta(NonZeroUsize::new(pos as usize).unwrap(), num_bytes)
    }
}

impl<I: Idx, T> LazyMeta for Table<I, T>
where
    Option<T>: FixedSizeEncoding,
{
    /// Number of bytes in the data stream.
    type Meta = usize;
}

impl<I: Idx, T> Lazy<Table<I, T>>
where
    Option<T>: FixedSizeEncoding,
{
    /// Given the metadata, extract out the value at a particular index (if any).
    #[inline(never)]
    pub(super) fn get<'a, 'tcx, M: Metadata<'a, 'tcx>, const N: usize>(
        &self,
        metadata: M,
        i: I,
    ) -> Option<T>
    where
        Option<T>: FixedSizeEncoding<ByteArray = [u8; N]>,
    {
        debug!("Table::lookup: index={:?} len={:?}", i, self.meta);

        let start = self.position.get();
        let bytes = &metadata.blob()[start..start + self.meta];
        let (bytes, []) = bytes.as_chunks::<N>() else { panic!() };
        let bytes = bytes.get(i.index())?;
        FixedSizeEncoding::from_bytes(bytes)
    }

    /// Size of the table in entries, including possible gaps.
    pub(super) fn size<const N: usize>(&self) -> usize
    where
        Option<T>: FixedSizeEncoding<ByteArray = [u8; N]>,
    {
        self.meta / N
    }
}
