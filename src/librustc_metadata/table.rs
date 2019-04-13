use crate::schema::*;

use rustc::hir::def_id::{DefId, DefIndex};
use rustc_serialize::{Encodable, opaque::Encoder};
use std::convert::TryInto;
use std::marker::PhantomData;
use std::num::NonZeroUsize;
use log::debug;

/// Helper trait, for encoding to, and decoding from, a fixed number of bytes.
trait FixedSizeEncoding {
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
            Self::from_bytes(&b[i])
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

/// Random-access position table, allowing encoding in an arbitrary order
/// (e.g. while visiting the definitions of a crate), and on-demand decoding
/// of specific indices (e.g. queries for per-definition data).
/// Similar to `Vec<Lazy<T>>`, but with zero-copy decoding.
// FIXME(eddyb) newtype `[u8]` here, such that `Box<Table<T>>` would be used
// when building it, and `Lazy<Table<T>>` or `&Table<T>` when reading it.
// Sadly, that doesn't work for `DefPerTable`, which is `(Table<T>, Table<T>)`,
// and so would need two lengths in its metadata, which is not supported yet.
crate struct Table<T: LazyMeta<Meta = ()>> {
    bytes: Vec<u8>,
    _marker: PhantomData<T>,
}

impl<T: LazyMeta<Meta = ()>> Table<T> {
    crate fn new(len: usize) -> Self {
        Table {
            bytes: vec![0; len * 4],
            _marker: PhantomData,
        }
    }

    crate fn record(&mut self, i: usize, entry: Lazy<T>) {
        let position: u32 = entry.position.get().try_into().unwrap();

        assert!(u32::read_from_bytes_at(&self.bytes, i) == 0,
                "recorded position for index {:?} twice, first at {:?} and now at {:?}",
                i,
                u32::read_from_bytes_at(&self.bytes, i),
                position);

        position.write_to_bytes_at(&mut self.bytes, i)
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

impl<T: LazyMeta<Meta = ()>> LazyMeta for Table<T> {
    type Meta = usize;

    fn min_size(len: usize) -> usize {
        len
    }
}

impl<T: Encodable> Lazy<Table<T>> {
    /// Given the metadata, extract out the offset of a particular index (if any).
    #[inline(never)]
    crate fn lookup(&self, bytes: &[u8], i: usize) -> Option<Lazy<T>> {
        debug!("Table::lookup: index={:?} len={:?}", i, self.meta);

        let bytes = &bytes[self.position.get()..][..self.meta];
        let position = u32::read_from_bytes_at(bytes, i);
        debug!("Table::lookup: position={:?}", position);

        NonZeroUsize::new(position as usize).map(Lazy::from_position)
    }
}


/// Per-definition table, similar to `Table` but keyed on `DefIndex`.
// FIXME(eddyb) replace by making `Table` behave like `IndexVec`,
// and by using `newtype_index!` to define `DefIndex`.
crate struct PerDefTable<T: LazyMeta<Meta = ()>>(Table<T>);

impl<T: LazyMeta<Meta = ()>> PerDefTable<T> {
    crate fn new(def_index_count: usize) -> Self {
        PerDefTable(Table::new(def_index_count))
    }

    crate fn record(&mut self, def_id: DefId, entry: Lazy<T>) {
        assert!(def_id.is_local());
        self.0.record(def_id.index.index(), entry);
    }

    crate fn encode(&self, buf: &mut Encoder) -> Lazy<Self> {
        let lazy = self.0.encode(buf);
        Lazy::from_position_and_meta(lazy.position, lazy.meta)
    }
}

impl<T: LazyMeta<Meta = ()>> LazyMeta for PerDefTable<T> {
    type Meta = <Table<T> as LazyMeta>::Meta;

    fn min_size(meta: Self::Meta) -> usize {
        Table::<T>::min_size(meta)
    }
}

impl<T: Encodable> Lazy<PerDefTable<T>> {
    fn as_table(&self) -> Lazy<Table<T>> {
        Lazy::from_position_and_meta(self.position, self.meta)
    }

    /// Given the metadata, extract out the offset of a particular DefIndex (if any).
    #[inline(never)]
    crate fn lookup(&self, bytes: &[u8], def_index: DefIndex) -> Option<Lazy<T>> {
        self.as_table().lookup(bytes, def_index.index())
    }
}
