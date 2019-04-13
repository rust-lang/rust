use crate::decoder::Metadata;
use crate::schema::*;

use rustc::hir::def_id::{DefId, DefIndex};
use rustc_serialize::{Encodable, opaque::Encoder};
use std::convert::TryInto;
use std::marker::PhantomData;
use std::num::NonZeroUsize;
use log::debug;

/// Helper trait, for encoding to, and decoding from, a fixed number of bytes.
/// Used mainly for Lazy positions and lengths.
/// Unchecked invariant: `Self::default()` should encode as `[0; BYTE_LEN]`,
/// but this has no impact on safety.
crate trait FixedSizeEncoding: Default {
    const BYTE_LEN: usize;

    // FIXME(eddyb) convert to and from `[u8; Self::BYTE_LEN]` instead,
    // once that starts being allowed by the compiler (i.e. lazy normalization).
    fn from_bytes(b: &[u8]) -> Self;
    fn write_to_bytes(self, b: &mut [u8]);

    // FIXME(eddyb) make these generic functions, or at least defaults here.
    // (same problem as above, needs `[u8; Self::BYTE_LEN]`)
    // For now, a macro (`fixed_size_encoding_byte_len_and_defaults`) is used.
    fn read_from_bytes_at(b: &[u8], i: usize) -> Self;
    fn write_to_bytes_at(self, b: &mut [u8], i: usize);
}

// HACK(eddyb) this shouldn't be needed (see comments on the methods above).
macro_rules! fixed_size_encoding_byte_len_and_defaults {
    ($byte_len:expr) => {
        const BYTE_LEN: usize = $byte_len;
        fn read_from_bytes_at(b: &[u8], i: usize) -> Self {
            const BYTE_LEN: usize = $byte_len;
            // HACK(eddyb) ideally this would be done with fully safe code,
            // but slicing `[u8]` with `i * N..` is optimized worse, due to the
            // possibility of `i * N` overflowing, than indexing `[[u8; N]]`.
            let b = unsafe {
                std::slice::from_raw_parts(
                    b.as_ptr() as *const [u8; BYTE_LEN],
                    b.len() / BYTE_LEN,
                )
            };
            FixedSizeEncoding::from_bytes(&b[i])
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
    }
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
impl<T: Encodable> FixedSizeEncoding for Option<Lazy<T>> {
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

impl<T: Encodable> FixedSizeEncoding for Option<Lazy<[T]>> {
    fixed_size_encoding_byte_len_and_defaults!(u32::BYTE_LEN * 2);

    fn from_bytes(b: &[u8]) -> Self {
        Some(Lazy::from_position_and_meta(
            <Option<Lazy<T>>>::from_bytes(b)?.position,
            u32::from_bytes(&b[u32::BYTE_LEN..]) as usize,
        ))
    }

    fn write_to_bytes(self, b: &mut [u8]) {
        self.map(|lazy| Lazy::<T>::from_position(lazy.position))
            .write_to_bytes(b);

        let len = self.map_or(0, |lazy| lazy.meta);
        let len: u32 = len.try_into().unwrap();

        len.write_to_bytes(&mut b[u32::BYTE_LEN..]);
    }
}

/// Random-access table, similar to `Vec<Option<T>>`, but without requiring
/// encoding or decoding all the values eagerly and in-order.
// FIXME(eddyb) replace `Vec` with `[_]` here, such that `Box<Table<T>>` would be used
// when building it, and `Lazy<Table<T>>` or `&Table<T>` when reading it.
// Sadly, that doesn't work for `DefPerTable`, which is `(Table<T>, Table<T>)`,
// and so would need two lengths in its metadata, which is not supported yet.
crate struct Table<T> where Option<T>: FixedSizeEncoding {
    // FIXME(eddyb) store `[u8; <Option<T>>::BYTE_LEN]` instead of `u8` in `Vec`,
    // once that starts being allowed by the compiler (i.e. lazy normalization).
    bytes: Vec<u8>,
    _marker: PhantomData<T>,
}

impl<T> Table<T> where Option<T>: FixedSizeEncoding {
    crate fn new(len: usize) -> Self {
        Table {
            // FIXME(eddyb) only allocate and encode as many entries as needed.
            bytes: vec![0; len * <Option<T>>::BYTE_LEN],
            _marker: PhantomData,
        }
    }

    crate fn set(&mut self, i: usize, value: T) {
        Some(value).write_to_bytes_at(&mut self.bytes, i);
    }

    crate fn encode(&self, buf: &mut Encoder) -> Lazy<Self> {
        let pos = buf.position();
        buf.emit_raw_bytes(&self.bytes);
        Lazy::from_position_and_meta(
            NonZeroUsize::new(pos as usize).unwrap(),
            self.bytes.len(),
        )
    }
}

impl<T> LazyMeta for Table<T> where Option<T>: FixedSizeEncoding {
    type Meta = usize;

    fn min_size(len: usize) -> usize {
        len
    }
}

impl<T> Lazy<Table<T>> where Option<T>: FixedSizeEncoding {
    /// Given the metadata, extract out the value at a particular index (if any).
    #[inline(never)]
    crate fn get<'a, 'tcx, M: Metadata<'a, 'tcx>>(
        &self,
        metadata: M,
        i: usize,
    ) -> Option<T> {
        debug!("Table::lookup: index={:?} len={:?}", i, self.meta);

        let bytes = &metadata.raw_bytes()[self.position.get()..][..self.meta];
        <Option<T>>::read_from_bytes_at(bytes, i)
    }
}

/// Per-definition table, similar to `Table` but keyed on `DefIndex`.
// FIXME(eddyb) replace by making `Table` behave like `IndexVec`,
// and by using `newtype_index!` to define `DefIndex`.
crate struct PerDefTable<T>(Table<T>) where Option<T>: FixedSizeEncoding;

impl<T> PerDefTable<T> where Option<T>: FixedSizeEncoding {
    crate fn new(def_index_count: usize) -> Self {
        PerDefTable(Table::new(def_index_count))
    }

    crate fn set(&mut self, def_id: DefId, value: T) {
        assert!(def_id.is_local());
        self.0.set(def_id.index.index(), value);
    }

    crate fn encode(&self, buf: &mut Encoder) -> Lazy<Self> {
        let lazy = self.0.encode(buf);
        Lazy::from_position_and_meta(lazy.position, lazy.meta)
    }
}

impl<T> LazyMeta for PerDefTable<T> where Option<T>: FixedSizeEncoding {
    type Meta = <Table<T> as LazyMeta>::Meta;

    fn min_size(meta: Self::Meta) -> usize {
        Table::<T>::min_size(meta)
    }
}

impl<T> Lazy<PerDefTable<T>> where Option<T>: FixedSizeEncoding {
    fn as_table(&self) -> Lazy<Table<T>> {
        Lazy::from_position_and_meta(self.position, self.meta)
    }

    /// Given the metadata, extract out the value at a particular DefIndex (if any).
    #[inline(never)]
    crate fn get<'a, 'tcx, M: Metadata<'a, 'tcx>>(
        &self,
        metadata: M,
        def_index: DefIndex,
    ) -> Option<T> {
        self.as_table().get(metadata, def_index.index())
    }
}
