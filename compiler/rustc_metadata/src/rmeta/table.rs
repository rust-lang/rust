use crate::rmeta::*;

use rustc_index::vec::Idx;
use rustc_serialize::opaque::Encoder;
use rustc_serialize::Encoder as _;
use std::convert::TryInto;
use std::marker::PhantomData;
use std::num::NonZeroUsize;
use tracing::debug;

/// Helper trait, for encoding to, and decoding from, a fixed number of bytes.
/// Used mainly for Lazy positions and lengths.
/// Unchecked invariant: `Self::default()` should encode as `[0; BYTE_LEN]`,
/// but this has no impact on safety.
pub(super) trait FixedSizeEncoding: Default {
    const BYTE_LEN: usize;

    // FIXME(eddyb) convert to and from `[u8; Self::BYTE_LEN]` instead,
    // once that starts being allowed by the compiler (i.e. lazy normalization).
    fn from_bytes(b: &[u8]) -> Self;
    fn write_to_bytes(self, b: &mut [u8]);

    // FIXME(eddyb) make these generic functions, or at least defaults here.
    // (same problem as above, needs `[u8; Self::BYTE_LEN]`)
    // For now, a macro (`fixed_size_encoding_byte_len_and_defaults`) is used.

    /// Read a `Self` value (encoded as `Self::BYTE_LEN` bytes),
    /// from `&b[i * Self::BYTE_LEN..]`, returning `None` if `i`
    /// is not in bounds, or `Some(Self::from_bytes(...))` otherwise.
    fn maybe_read_from_bytes_at(b: &[u8], i: usize) -> Option<Self>;
    /// Write a `Self` value (encoded as `Self::BYTE_LEN` bytes),
    /// at `&mut b[i * Self::BYTE_LEN..]`, using `Self::write_to_bytes`.
    fn write_to_bytes_at(self, b: &mut [u8], i: usize);
}

// HACK(eddyb) this shouldn't be needed (see comments on the methods above).
macro_rules! fixed_size_encoding_byte_len_and_defaults {
    ($byte_len:expr) => {
        const BYTE_LEN: usize = $byte_len;
        fn maybe_read_from_bytes_at(b: &[u8], i: usize) -> Option<Self> {
            const BYTE_LEN: usize = $byte_len;
            // HACK(eddyb) ideally this would be done with fully safe code,
            // but slicing `[u8]` with `i * N..` is optimized worse, due to the
            // possibility of `i * N` overflowing, than indexing `[[u8; N]]`.
            let b = unsafe {
                std::slice::from_raw_parts(b.as_ptr() as *const [u8; BYTE_LEN], b.len() / BYTE_LEN)
            };
            b.get(i).map(|b| FixedSizeEncoding::from_bytes(b))
        }
        fn write_to_bytes_at(self, b: &mut [u8], i: usize) {
            const BYTE_LEN: usize = $byte_len;
            // HACK(eddyb) ideally this would be done with fully safe code,
            // see similar comment in `read_from_bytes_at` for why it can't yet.
            let b = unsafe {
                std::slice::from_raw_parts_mut(
                    b.as_mut_ptr() as *mut [u8; BYTE_LEN],
                    b.len() / BYTE_LEN,
                )
            };
            self.write_to_bytes(&mut b[i]);
        }
    };
}

impl FixedSizeEncoding for u32 {
    fixed_size_encoding_byte_len_and_defaults!(4);

    fn from_bytes(b: &[u8]) -> Self {
        let mut bytes = [0; Self::BYTE_LEN];
        bytes.copy_from_slice(&b[..Self::BYTE_LEN]);
        Self::from_le_bytes(bytes)
    }

    fn write_to_bytes(self, b: &mut [u8]) {
        b[..Self::BYTE_LEN].copy_from_slice(&self.to_le_bytes());
    }
}

// NOTE(eddyb) there could be an impl for `usize`, which would enable a more
// generic `Lazy<T>` impl, but in the general case we might not need / want to
// fit every `usize` in `u32`.
impl<T> FixedSizeEncoding for Option<Lazy<T>> {
    fixed_size_encoding_byte_len_and_defaults!(u32::BYTE_LEN);

    fn from_bytes(b: &[u8]) -> Self {
        Some(Lazy::from_position(NonZeroUsize::new(u32::from_bytes(b) as usize)?))
    }

    fn write_to_bytes(self, b: &mut [u8]) {
        let position = self.map_or(0, |lazy| lazy.position.get());
        let position: u32 = position.try_into().unwrap();

        position.write_to_bytes(b)
    }
}

impl<T> FixedSizeEncoding for Option<Lazy<[T]>> {
    fixed_size_encoding_byte_len_and_defaults!(u32::BYTE_LEN * 2);

    fn from_bytes(b: &[u8]) -> Self {
        Some(Lazy::from_position_and_meta(
            <Option<Lazy<T>>>::from_bytes(b)?.position,
            u32::from_bytes(&b[u32::BYTE_LEN..]) as usize,
        ))
    }

    fn write_to_bytes(self, b: &mut [u8]) {
        self.map(|lazy| Lazy::<T>::from_position(lazy.position)).write_to_bytes(b);

        let len = self.map_or(0, |lazy| lazy.meta);
        let len: u32 = len.try_into().unwrap();

        len.write_to_bytes(&mut b[u32::BYTE_LEN..]);
    }
}

/// Random-access table (i.e. offering constant-time `get`/`set`), similar to
/// `Vec<Option<T>>`, but without requiring encoding or decoding all the values
/// eagerly and in-order.
/// A total of `(max_idx + 1) * <Option<T> as FixedSizeEncoding>::BYTE_LEN` bytes
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
    // FIXME(eddyb) use `IndexVec<I, [u8; <Option<T>>::BYTE_LEN]>` instead of
    // `Vec<u8>`, once that starts working (i.e. lazy normalization).
    // Then again, that has the downside of not allowing `TableBuilder::encode` to
    // obtain a `&[u8]` entirely in safe code, for writing the bytes out.
    bytes: Vec<u8>,
    _marker: PhantomData<(fn(&I), T)>,
}

impl<I: Idx, T> Default for TableBuilder<I, T>
where
    Option<T>: FixedSizeEncoding,
{
    fn default() -> Self {
        TableBuilder { bytes: vec![], _marker: PhantomData }
    }
}

impl<I: Idx, T> TableBuilder<I, T>
where
    Option<T>: FixedSizeEncoding,
{
    pub(crate) fn set(&mut self, i: I, value: T) {
        // FIXME(eddyb) investigate more compact encodings for sparse tables.
        // On the PR @michaelwoerister mentioned:
        // > Space requirements could perhaps be optimized by using the HAMT `popcnt`
        // > trick (i.e. divide things into buckets of 32 or 64 items and then
        // > store bit-masks of which item in each bucket is actually serialized).
        let i = i.index();
        let needed = (i + 1) * <Option<T>>::BYTE_LEN;
        if self.bytes.len() < needed {
            self.bytes.resize(needed, 0);
        }

        Some(value).write_to_bytes_at(&mut self.bytes, i);
    }

    pub(crate) fn encode(&self, buf: &mut Encoder) -> Lazy<Table<I, T>> {
        let pos = buf.position();
        buf.emit_raw_bytes(&self.bytes).unwrap();
        Lazy::from_position_and_meta(NonZeroUsize::new(pos as usize).unwrap(), self.bytes.len())
    }
}

impl<I: Idx, T> LazyMeta for Table<I, T>
where
    Option<T>: FixedSizeEncoding,
{
    type Meta = usize;

    fn min_size(len: usize) -> usize {
        len
    }
}

impl<I: Idx, T> Lazy<Table<I, T>>
where
    Option<T>: FixedSizeEncoding,
{
    /// Given the metadata, extract out the value at a particular index (if any).
    #[inline(never)]
    pub(super) fn get<'a, 'tcx, M: Metadata<'a, 'tcx>>(&self, metadata: M, i: I) -> Option<T> {
        debug!("Table::lookup: index={:?} len={:?}", i, self.meta);

        let start = self.position.get();
        let bytes = &metadata.blob()[start..start + self.meta];
        <Option<T>>::maybe_read_from_bytes_at(bytes, i.index())?
    }

    /// Size of the table in entries, including possible gaps.
    pub(super) fn size(&self) -> usize {
        self.meta / <Option<T>>::BYTE_LEN
    }
}
