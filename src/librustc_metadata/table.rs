use crate::schema::*;

use rustc::hir::def_id::{DefId, DefIndex};
use rustc_serialize::opaque::Encoder;
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
crate struct Table<T> {
    positions: Vec<u8>,
    _marker: PhantomData<T>,
}

impl<T> Table<T> {
    crate fn new(max_index: usize) -> Self {
        Table {
            positions: vec![0; max_index * 4],
            _marker: PhantomData,
        }
    }

    crate fn record(&mut self, def_id: DefId, entry: Lazy<T>) {
        assert!(def_id.is_local());
        self.record_index(def_id.index, entry);
    }

    crate fn record_index(&mut self, item: DefIndex, entry: Lazy<T>) {
        let position: u32 = entry.position.get().try_into().unwrap();
        let array_index = item.index();

        let positions = &mut self.positions;
        assert!(u32::read_from_bytes_at(positions, array_index) == 0,
                "recorded position for item {:?} twice, first at {:?} and now at {:?}",
                item,
                u32::read_from_bytes_at(positions, array_index),
                position);

        position.write_to_bytes_at(positions, array_index)
    }

    crate fn encode(&self, buf: &mut Encoder) -> Lazy<[Self]> {
        let pos = buf.position();
        buf.emit_raw_bytes(&self.positions);
        Lazy::from_position_and_meta(
            NonZeroUsize::new(pos as usize).unwrap(),
            self.positions.len() / 4,
        )
    }
}

impl<T> Lazy<[Table<T>]> {
    /// Given the metadata, extract out the offset of a particular
    /// DefIndex (if any).
    #[inline(never)]
    crate fn lookup(&self, bytes: &[u8], def_index: DefIndex) -> Option<Lazy<T>> {
        debug!("Table::lookup: index={:?} len={:?}",
               def_index,
               self.meta);

        let bytes = &bytes[self.position.get()..][..self.meta * 4];
        let position = u32::read_from_bytes_at(bytes, def_index.index());
        debug!("Table::lookup: position={:?}", position);
        NonZeroUsize::new(position as usize).map(Lazy::from_position)
    }
}
