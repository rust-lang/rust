// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use middle::def_id::{DefId, DefIndex};
use rbml;
use std::io::{Cursor, Write};
use std::slice;
use std::u32;

/// As part of the metadata, we generate an index that stores, for
/// each DefIndex, the position of the corresponding RBML document (if
/// any).  This is just a big `[u32]` slice, where an entry of
/// `u32::MAX` indicates that there is no RBML document. This little
/// struct just stores the offsets within the metadata of the start
/// and end of this slice. These are actually part of an RBML
/// document, but for looking things up in the metadata, we just
/// discard the RBML positioning and jump directly to the data.
pub struct Index {
    data_start: usize,
    data_end: usize,
}

impl Index {
    /// Given the RBML doc representing the index, save the offests
    /// for later.
    pub fn from_rbml(index: rbml::Doc) -> Index {
        Index { data_start: index.start, data_end: index.end }
    }

    /// Given the metadata, extract out the offset of a particular
    /// DefIndex (if any).
    #[inline(never)]
    pub fn lookup_item(&self, bytes: &[u8], def_index: DefIndex) -> Option<u32> {
        let words = bytes_to_words(&bytes[self.data_start..self.data_end]);
        let index = def_index.as_usize();

        debug!("lookup_item: index={:?} words.len={:?}",
               index, words.len());

        let position = u32::from_be(words[index]);
        if position == u32::MAX {
            debug!("lookup_item: position=u32::MAX");
            None
        } else {
            debug!("lookup_item: position={:?}", position);
            Some(position)
        }
    }
}

/// While we are generating the metadata, we also track the position
/// of each DefIndex. It is not required that all definitions appear
/// in the metadata, nor that they are serialized in order, and
/// therefore we first allocate the vector here and fill it with
/// `u32::MAX`. Whenever an index is visited, we fill in the
/// appropriate spot by calling `record_position`. We should never
/// visit the same index twice.
pub struct IndexData {
    positions: Vec<u32>,
}

impl IndexData {
    pub fn new(max_index: usize) -> IndexData {
        IndexData {
            positions: vec![u32::MAX; max_index]
        }
    }

    pub fn record(&mut self, def_id: DefId, position: u64) {
        assert!(def_id.is_local());
        self.record_index(def_id.index, position)
    }

    pub fn record_index(&mut self, item: DefIndex, position: u64) {
        let item = item.as_usize();

        assert!(position < (u32::MAX as u64));
        let position = position as u32;

        assert!(self.positions[item] == u32::MAX,
                "recorded position for item {:?} twice, first at {:?} and now at {:?}",
                item, self.positions[item], position);

        self.positions[item] = position;
    }

    pub fn write_index(&self, buf: &mut Cursor<Vec<u8>>) {
        for &position in &self.positions {
            write_be_u32(buf, position);
        }
    }
}

/// A dense index with integer keys. Different API from IndexData (should
/// these be merged?)
pub struct DenseIndex {
    start: usize,
    end: usize
}

impl DenseIndex {
    pub fn lookup(&self, buf: &[u8], ix: u32) -> Option<u32> {
        let data = bytes_to_words(&buf[self.start..self.end]);
        data.get(ix as usize).map(|d| u32::from_be(*d))
    }
    pub fn from_buf(buf: &[u8], start: usize, end: usize) -> Self {
        assert!((end-start)%4 == 0 && start <= end && end <= buf.len());
        DenseIndex {
            start: start,
            end: end
        }
    }
}

pub fn write_dense_index(entries: Vec<u32>, buf: &mut Cursor<Vec<u8>>) {
    let elen = entries.len();
    assert!(elen < u32::MAX as usize);

    for entry in entries {
        write_be_u32(buf, entry);
    }

    info!("write_dense_index: {} entries", elen);
}

fn write_be_u32<W: Write>(w: &mut W, u: u32) {
    let _ = w.write_all(&[
        (u >> 24) as u8,
        (u >> 16) as u8,
        (u >>  8) as u8,
        (u >>  0) as u8,
    ]);
}

fn bytes_to_words(b: &[u8]) -> &[u32] {
    assert!(b.len() % 4 == 0);
    unsafe { slice::from_raw_parts(b.as_ptr() as *const u32, b.len()/4) }
}
