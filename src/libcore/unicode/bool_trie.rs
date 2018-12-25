/// BoolTrie is a trie for representing a set of Unicode codepoints. It is
/// implemented with postfix compression (sharing of identical child nodes),
/// which gives both compact size and fast lookup.
///
/// The space of Unicode codepoints is divided into 3 subareas, each
/// represented by a trie with different depth. In the first (0..0x800), there
/// is no trie structure at all; each u64 entry corresponds to a bitvector
/// effectively holding 64 bool values.
///
/// In the second (0x800..0x10000), each child of the root node represents a
/// 64-wide subrange, but instead of storing the full 64-bit value of the leaf,
/// the trie stores an 8-bit index into a shared table of leaf values. This
/// exploits the fact that in reasonable sets, many such leaves can be shared.
///
/// In the third (0x10000..0x110000), each child of the root node represents a
/// 4096-wide subrange, and the trie stores an 8-bit index into a 64-byte slice
/// of a child tree. Each of these 64 bytes represents an index into the table
/// of shared 64-bit leaf values. This exploits the sparse structure in the
/// non-BMP range of most Unicode sets.
pub struct BoolTrie {
    // 0..0x800 (corresponding to 1 and 2 byte utf-8 sequences)
    pub r1: [u64; 32],   // leaves

    // 0x800..0x10000 (corresponding to 3 byte utf-8 sequences)
    pub r2: [u8; 992],      // first level
    pub r3: &'static [u64],  // leaves

    // 0x10000..0x110000 (corresponding to 4 byte utf-8 sequences)
    pub r4: [u8; 256],       // first level
    pub r5: &'static [u8],   // second level
    pub r6: &'static [u64],  // leaves
}
impl BoolTrie {
    pub fn lookup(&self, c: char) -> bool {
        let c = c as u32;
        if c < 0x800 {
            trie_range_leaf(c, self.r1[(c >> 6) as usize])
        } else if c < 0x10000 {
            let child = self.r2[(c >> 6) as usize - 0x20];
            trie_range_leaf(c, self.r3[child as usize])
        } else {
            let child = self.r4[(c >> 12) as usize - 0x10];
            let leaf = self.r5[((child as usize) << 6) + ((c >> 6) as usize & 0x3f)];
            trie_range_leaf(c, self.r6[leaf as usize])
        }
    }
}

pub struct SmallBoolTrie {
    pub(crate) r1: &'static [u8],  // first level
    pub(crate) r2: &'static [u64],  // leaves
}

impl SmallBoolTrie {
    pub fn lookup(&self, c: char) -> bool {
        let c = c as u32;
        match self.r1.get((c >> 6) as usize) {
            Some(&child) => trie_range_leaf(c, self.r2[child as usize]),
            None => false,
        }
    }
}

fn trie_range_leaf(c: u32, bitmap_chunk: u64) -> bool {
    ((bitmap_chunk >> (c & 63)) & 1) != 0
}
