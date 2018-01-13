// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use schema::*;

use rustc::hir::def_id::{DefId, DefIndex, DefIndexAddressSpace};
use std::io::{Cursor, Write};
use std::slice;
use std::u32;

/// While we are generating the metadata, we also track the position
/// of each DefIndex. It is not required that all definitions appear
/// in the metadata, nor that they are serialized in order, and
/// therefore we first allocate the vector here and fill it with
/// `u32::MAX`. Whenever an index is visited, we fill in the
/// appropriate spot by calling `record_position`. We should never
/// visit the same index twice.
pub struct Index {
    positions: [Vec<u32>; 2]
}

impl Index {
    pub fn new((max_index_lo, max_index_hi): (usize, usize)) -> Index {
        Index {
            positions: [vec![u32::MAX; max_index_lo],
                        vec![u32::MAX; max_index_hi]],
        }
    }

    pub fn record(&mut self, def_id: DefId, entry: Lazy<Entry>) {
        assert!(def_id.is_local());
        self.record_index(def_id.index, entry);
    }

    pub fn record_index(&mut self, item: DefIndex, entry: Lazy<Entry>) {
        assert!(entry.position < (u32::MAX as usize));
        let position = entry.position as u32;
        let space_index = item.address_space().index();
        let array_index = item.as_array_index();

        assert!(self.positions[space_index][array_index] == u32::MAX,
                "recorded position for item {:?} twice, first at {:?} and now at {:?}",
                item,
                self.positions[space_index][array_index],
                position);

        self.positions[space_index][array_index] = position.to_le();
    }

    pub fn write_index(&self, buf: &mut Cursor<Vec<u8>>) -> LazySeq<Index> {
        let pos = buf.position();

        // First we write the length of the lower range ...
        buf.write_all(words_to_bytes(&[(self.positions[0].len() as u32).to_le()])).unwrap();
        // ... then the values in the lower range ...
        buf.write_all(words_to_bytes(&self.positions[0][..])).unwrap();
        // ... then the values in the higher range.
        buf.write_all(words_to_bytes(&self.positions[1][..])).unwrap();
        LazySeq::with_position_and_length(pos as usize,
            self.positions[0].len() + self.positions[1].len() + 1)
    }
}

impl<'tcx> LazySeq<Index> {
    /// Given the metadata, extract out the offset of a particular
    /// DefIndex (if any).
    #[inline(never)]
    pub fn lookup(&self, bytes: &[u8], def_index: DefIndex) -> Option<Lazy<Entry<'tcx>>> {
        let words = &bytes_to_words(&bytes[self.position..])[..self.len];

        debug!("Index::lookup: index={:?} words.len={:?}",
               def_index,
               words.len());

        let positions = match def_index.address_space() {
            DefIndexAddressSpace::Low => &words[1..],
            DefIndexAddressSpace::High => {
                // This is a DefIndex in the higher range, so find out where
                // that starts:
                let lo_count = u32::from_le(words[0].get()) as usize;
                &words[lo_count + 1 .. ]
            }
        };

        let array_index = def_index.as_array_index();
        let position = u32::from_le(positions[array_index].get());
        if position == u32::MAX {
            debug!("Index::lookup: position=u32::MAX");
            None
        } else {
            debug!("Index::lookup: position={:?}", position);
            Some(Lazy::with_position(position as usize))
        }
    }
}

#[repr(packed)]
#[derive(Copy)]
struct Unaligned<T>(T);

// The derived Clone impl is unsafe for this packed struct since it needs to pass a reference to
// the field to `T::clone`, but this reference may not be properly aligned.
impl<T: Copy> Clone for Unaligned<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Unaligned<T> {
    fn get(self) -> T { self.0 }
}

fn bytes_to_words(b: &[u8]) -> &[Unaligned<u32>] {
    unsafe { slice::from_raw_parts(b.as_ptr() as *const Unaligned<u32>, b.len() / 4) }
}

fn words_to_bytes(w: &[u32]) -> &[u8] {
    unsafe { slice::from_raw_parts(w.as_ptr() as *const u8, w.len() * 4) }
}
