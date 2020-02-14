//! This implements the core logic of the compression scheme used to compactly
//! encode the Unicode character classes.
//!
//! The primary idea is that we 'flatten' the Unicode ranges into an enormous
//! bitset. To represent any arbitrary codepoint in a raw bitset, we would need
//! over 17 kilobytes of data per character set -- way too much for our
//! purposes.
//!
//! We have two primary goals with the encoding: we want to be compact, because
//! these tables often end up in ~every Rust program (especially the
//! grapheme_extend table, used for str debugging), including those for embedded
//! targets (where space is important). We also want to be relatively fast,
//! though this is more of a nice to have rather than a key design constraint.
//! In practice, due to modern processor design these two are closely related.
//!
//! The encoding scheme here compresses the bitset by first deduplicating the
//! "words" (64 bits on all platforms). In practice very few words are present
//! in most data sets.
//!
//! This gives us an array that maps `u8 -> word` (if we ever went beyond 256
//! words, we could go to u16 -> word or have some dual compression scheme
//! mapping into two separate sets; currently this is not dealt with).
//!
//! With that scheme, we now have a single byte for every 64 codepoints. We
//! further group these by 16 (arbitrarily chosen), and again deduplicate and
//! store in an array (u8 -> [u8; 16]).
//!
//! The indices into this array represent ranges of 64*16 = 1024 codepoints.
//!
//! This already reduces the top-level array to at most 1,086 bytes, but in
//! practice we usually can encode in far fewer (the first couple Unicode planes
//! are dense).
//!
//! The last byte of this top-level array is pulled out to a separate static
//! and trailing zeros are dropped; this is simply because grapheme_extend and
//! case_ignorable have a single entry in the 896th entry, so this shrinks them
//! down considerably.

use crate::fmt_list;
use std::collections::{BTreeSet, HashMap};
use std::convert::TryFrom;
use std::fmt::Write;
use std::ops::Range;

pub struct RawEmitter {
    pub file: String,
    pub bytes_used: usize,
}

impl RawEmitter {
    pub fn new() -> RawEmitter {
        RawEmitter { file: String::new(), bytes_used: 0 }
    }

    fn blank_line(&mut self) {
        if self.file.is_empty() || self.file.ends_with("\n\n") {
            return;
        }
        writeln!(&mut self.file, "").unwrap();
    }

    fn emit_bitset(&mut self, words: &[u64]) {
        let unique_words =
            words.iter().cloned().collect::<BTreeSet<_>>().into_iter().collect::<Vec<_>>();
        if unique_words.len() > u8::max_value() as usize {
            panic!("cannot pack {} into 8 bits", unique_words.len());
        }

        let word_indices = unique_words
            .iter()
            .cloned()
            .enumerate()
            .map(|(idx, word)| (word, u8::try_from(idx).unwrap()))
            .collect::<HashMap<_, _>>();

        let mut idx = words.iter().map(|w| word_indices[w]).collect::<Vec<u8>>();
        let chunk_length = 16;
        for _ in 0..(chunk_length - (idx.len() % chunk_length)) {
            assert_eq!(unique_words[0], 0, "first word is all zeros");
            // pad out bitset index with zero words so we have all chunks of 16
            idx.push(0);
        }

        let mut chunks = BTreeSet::new();
        for chunk in idx.chunks(chunk_length) {
            chunks.insert(chunk);
        }
        let chunk_map = chunks
            .clone()
            .into_iter()
            .enumerate()
            .map(|(idx, chunk)| (chunk, idx))
            .collect::<HashMap<_, _>>();
        let mut chunk_indices = Vec::new();
        for chunk in idx.chunks(chunk_length) {
            chunk_indices.push(chunk_map[chunk]);
        }
        writeln!(
            &mut self.file,
            "static BITSET_LAST_CHUNK_MAP: (u16, u8) = ({}, {});",
            chunk_indices.len() - 1,
            chunk_indices.pop().unwrap(),
        )
        .unwrap();
        self.bytes_used += 3;
        // Strip out the empty pieces, presuming our above pop() made us now
        // have some trailing zeros.
        assert_eq!(unique_words[0], 0, "first word is all zeros");
        while let Some(0) = chunk_indices.last() {
            chunk_indices.pop();
        }
        writeln!(
            &mut self.file,
            "static BITSET_CHUNKS_MAP: [u8; {}] = [{}];",
            chunk_indices.len(),
            fmt_list(&chunk_indices),
        )
        .unwrap();
        self.bytes_used += chunk_indices.len();
        writeln!(
            &mut self.file,
            "static BITSET_INDEX_CHUNKS: [[u8; 16]; {}] = [{}];",
            chunks.len(),
            fmt_list(chunks.iter()),
        )
        .unwrap();
        self.bytes_used += 16 * chunks.len();
        writeln!(
            &mut self.file,
            "static BITSET: [u64; {}] = [{}];",
            unique_words.len(),
            fmt_list(&unique_words),
        )
        .unwrap();
        self.bytes_used += 8 * unique_words.len();
    }

    pub fn emit_lookup(&mut self) {
        writeln!(&mut self.file, "pub fn lookup(c: char) -> bool {{").unwrap();
        writeln!(&mut self.file, "    super::range_search(",).unwrap();
        writeln!(&mut self.file, "        c as u32,").unwrap();
        writeln!(&mut self.file, "        &BITSET_CHUNKS_MAP,").unwrap();
        writeln!(&mut self.file, "        BITSET_LAST_CHUNK_MAP,").unwrap();
        writeln!(&mut self.file, "        &BITSET_INDEX_CHUNKS,").unwrap();
        writeln!(&mut self.file, "        &BITSET,").unwrap();
        writeln!(&mut self.file, "    )").unwrap();
        writeln!(&mut self.file, "}}").unwrap();
    }
}

pub fn emit_codepoints(emitter: &mut RawEmitter, ranges: &[Range<u32>]) {
    emitter.blank_line();

    let last_code_point = ranges.last().unwrap().end;
    // bitset for every bit in the codepoint range
    //
    // + 2 to ensure an all zero word to use for padding
    let mut buckets = vec![0u64; (last_code_point as usize / 64) + 2];
    for range in ranges {
        for codepoint in range.clone() {
            let bucket = codepoint as usize / 64;
            let bit = codepoint as u64 % 64;
            buckets[bucket] |= 1 << bit;
        }
    }

    emitter.emit_bitset(&buckets);
    emitter.blank_line();
    emitter.emit_lookup();
}
