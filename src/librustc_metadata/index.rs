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

use rustc::hir::def_id::{DefId, DefIndex};
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
    positions: Vec<u32>,
}

impl Index {
    pub fn new(max_index: usize) -> Index {
        Index { positions: vec![u32::MAX; max_index] }
    }

    pub fn record(&mut self, def_id: DefId, entry: Lazy<Entry>) {
        assert!(def_id.is_local());
        self.record_index(def_id.index, entry);
    }

    pub fn record_index(&mut self, item: DefIndex, entry: Lazy<Entry>) {
        let item = item.as_usize();

        assert!(entry.position < (u32::MAX as usize));
        let position = entry.position as u32;

        assert!(self.positions[item] == u32::MAX,
                "recorded position for item {:?} twice, first at {:?} and now at {:?}",
                item,
                self.positions[item],
                position);

        self.positions[item] = position.to_le();
    }

    pub fn write_index(&self, buf: &mut Cursor<Vec<u8>>) -> LazySeq<Index> {
        let pos = buf.position();
        buf.write_all(words_to_bytes(&self.positions)).unwrap();
        LazySeq::with_position_and_length(pos as usize, self.positions.len())
    }
}

impl<'tcx> LazySeq<Index> {
    /// Given the metadata, extract out the offset of a particular
    /// DefIndex (if any).
    #[inline(never)]
    pub fn lookup(&self, bytes: &[u8], def_index: DefIndex) -> Option<Lazy<Entry<'tcx>>> {
        let words = &bytes_to_words(&bytes[self.position..])[..self.len];
        let index = def_index.as_usize();

        debug!("Index::lookup: index={:?} words.len={:?}",
               index,
               words.len());

        let position = u32::from_le(words[index].get());
        if position == u32::MAX {
            debug!("Index::lookup: position=u32::MAX");
            None
        } else {
            debug!("Index::lookup: position={:?}", position);
            Some(Lazy::with_position(position as usize))
        }
    }

    pub fn iter_enumerated<'a>(&self,
                               bytes: &'a [u8])
                               -> impl Iterator<Item = (DefIndex, Lazy<Entry<'tcx>>)> + 'a {
        let words = &bytes_to_words(&bytes[self.position..])[..self.len];
        words.iter().map(|word| word.get()).enumerate().filter_map(|(index, position)| {
            if position == u32::MAX {
                None
            } else {
                let position = u32::from_le(position) as usize;
                Some((DefIndex::new(index), Lazy::with_position(position)))
            }
        })
    }
}

#[repr(packed)]
#[derive(Copy, Clone)]
struct Unaligned<T>(T);

impl<T> Unaligned<T> {
    fn get(self) -> T { self.0 }
}

fn bytes_to_words(b: &[u8]) -> &[Unaligned<u32>] {
    unsafe { slice::from_raw_parts(b.as_ptr() as *const Unaligned<u32>, b.len() / 4) }
}

fn words_to_bytes(w: &[u32]) -> &[u8] {
    unsafe { slice::from_raw_parts(w.as_ptr() as *const u8, w.len() * 4) }
}
