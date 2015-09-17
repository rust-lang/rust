// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::io::{Cursor, Write};
use std::slice;
use std::u32;
use syntax::ast::NodeId;

#[derive(Copy, Clone, PartialEq, PartialOrd, Eq, Ord)]
pub struct IndexEntry {
    pub node: NodeId,
    pub pos: u64
}

#[derive(Debug)]
pub struct IndexArrayEntry {
    bits: u32,
    first_pos: u32
}

impl IndexArrayEntry {
    fn encode_to<W: Write>(&self, b: &mut W) {
        write_be_u32(b, self.bits);
        write_be_u32(b, self.first_pos);
    }

    fn decode_from(b: &[u32]) -> Self {
        IndexArrayEntry {
            bits: u32::from_be(b[0]),
            first_pos: u32::from_be(b[1])
        }
    }
}

/// The Item Index
///
/// This index maps the NodeId of each item to its location in the
/// metadata.
///
/// The index is a sparse bit-vector consisting of a index-array
/// and a position-array. Each entry in the index-array handles 32 nodes.
/// The first word is a bit-array consisting of the nodes that hold items,
/// the second is the index of the first of the items in the position-array.
/// If there is a large set of non-item trailing nodes, they can be omitted
/// from the index-array.
///
/// The index is serialized as an array of big-endian 32-bit words.
/// The first word is the number of items in the position-array.
/// Then, for each item, its position in the metadata follows.
/// After that the index-array is stored.
///
/// struct index {
///     u32 item_count;
///     u32 items[self.item_count];
///     struct { u32 bits; u32 offset; } positions[..];
/// }
pub struct Index {
    position_start: usize,
    index_start: usize,
    index_end: usize,
}

pub fn write_index(mut entries: Vec<IndexEntry>, buf: &mut Cursor<Vec<u8>>) {
    assert!(entries.len() < u32::MAX as usize);
    entries.sort();

    let mut last_entry = IndexArrayEntry { bits: 0, first_pos: 0 };

    write_be_u32(buf, entries.len() as u32);
    for &IndexEntry { pos, .. } in &entries {
        assert!(pos < u32::MAX as u64);
        write_be_u32(buf, pos as u32);
    }

    let mut pos_in_index_array = 0;
    for (i, &IndexEntry { node, .. }) in entries.iter().enumerate() {
        let (x, s) = (node / 32 as u32, node % 32 as u32);
        while x > pos_in_index_array {
            pos_in_index_array += 1;
            last_entry.encode_to(buf);
            last_entry = IndexArrayEntry { bits: 0, first_pos: i as u32 };
        }
        last_entry.bits |= 1<<s;
    }
    last_entry.encode_to(buf);

    info!("write_index: {} items, {} array entries",
          entries.len(), pos_in_index_array);
}

impl Index {
    fn lookup_index(&self, index: &[u32], i: u32) -> Option<IndexArrayEntry> {
        let ix = (i as usize)*2;
        if ix >= index.len() {
            None
        } else {
            Some(IndexArrayEntry::decode_from(&index[ix..ix+2]))
        }
    }

    fn item_from_pos(&self, positions: &[u32], pos: u32) -> u32 {
        u32::from_be(positions[pos as usize])
    }

    #[inline(never)]
    pub fn lookup_item(&self, buf: &[u8], node: NodeId) -> Option<u32> {
        let index = bytes_to_words(&buf[self.index_start..self.index_end]);
        let positions = bytes_to_words(&buf[self.position_start..self.index_start]);
        let (x, s) = (node / 32 as u32, node % 32 as u32);
        let result = match self.lookup_index(index, x) {
            Some(IndexArrayEntry { bits, first_pos }) => {
                let bit = 1<<s;
                if bits & bit == 0 {
                    None
                } else {
                    let prev_nodes_for_entry = (bits&(bit-1)).count_ones();
                    Some(self.item_from_pos(
                        positions,
                        first_pos+prev_nodes_for_entry))
                }
            }
            None => None // trailing zero
        };
        debug!("lookup_item({:?}) = {:?}", node, result);
        result
    }

    pub fn from_buf(buf: &[u8], start: usize, end: usize) -> Self {
        let buf = bytes_to_words(&buf[start..end]);
        let position_count = buf[0].to_be() as usize;
        let position_len = position_count*4;
        info!("loaded index - position: {}-{}-{}", start, start+position_len, end);
        debug!("index contents are {:?}",
               buf.iter().map(|b| format!("{:08x}", b)).collect::<Vec<_>>().concat());
        assert!(end-4-start >= position_len);
        assert_eq!((end-4-start-position_len)%8, 0);
        Index {
            position_start: start+4,
            index_start: start+position_len+4,
            index_end: end
        }
    }
}

/// A dense index with integer keys
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

#[test]
fn test_index() {
    let entries = vec![
        IndexEntry { node: 0, pos: 17 },
        IndexEntry { node: 31, pos: 29 },
        IndexEntry { node: 32, pos: 1175 },
        IndexEntry { node: 191, pos: 21 },
        IndexEntry { node: 128, pos: 34 },
        IndexEntry { node: 145, pos: 70 },
        IndexEntry { node: 305, pos: 93214 },
        IndexEntry { node: 138, pos: 64 },
        IndexEntry { node: 129, pos: 53 },
        IndexEntry { node: 192, pos: 33334 },
        IndexEntry { node: 200, pos: 80123 },
    ];
    let mut c = Cursor::new(vec![]);
    write_index(entries.clone(), &mut c);
    let mut buf = c.into_inner();
    let expected: &[u8] = &[
        0, 0, 0, 11, // # entries
        // values:
        0,0,0,17, 0,0,0,29, 0,0,4,151, 0,0,0,34,
        0,0,0,53, 0,0,0,64, 0,0,0,70, 0,0,0,21,
        0,0,130,54, 0,1,56,251, 0,1,108,30,
        // index:
        128,0,0,1,0,0,0,0, 0,0,0,1,0,0,0,2,
        0,0,0,0,0,0,0,3,   0,0,0,0,0,0,0,3,
        0,2,4,3,0,0,0,3,   128,0,0,0,0,0,0,7,
        0,0,1,1,0,0,0,8,   0,0,0,0,0,0,0,10,
        0,0,0,0,0,0,0,10,  0,2,0,0,0,0,0,10
    ];
    assert_eq!(buf, expected);

    // insert some junk padding
    for i in 0..17 { buf.insert(0, i); buf.push(i) }
    let index = Index::from_buf(&buf, 17, buf.len()-17);

    // test round-trip
    for i in 0..4096 {
        assert_eq!(index.lookup_item(&buf, i),
                   entries.iter().find(|e| e.node == i).map(|n| n.pos as u32));
    }
}
