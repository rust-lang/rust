use crate::fmt_list;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::convert::TryFrom;
use std::fmt::{self, Write};
use std::ops::Range;

#[derive(Clone)]
pub struct RawEmitter {
    pub file: String,
    pub desc: String,
    pub bytes_used: usize,
}

impl RawEmitter {
    pub fn new() -> RawEmitter {
        RawEmitter { file: String::new(), bytes_used: 0, desc: String::new() }
    }

    fn blank_line(&mut self) {
        if self.file.is_empty() || self.file.ends_with("\n\n") {
            return;
        }
        writeln!(&mut self.file).unwrap();
    }

    fn emit_bitset(&mut self, ranges: &[Range<u32>]) -> Result<(), String> {
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

        let mut words = buckets;
        // Ensure that there's a zero word in the dataset, used for padding and
        // such.
        words.push(0);
        let unique_words =
            words.iter().cloned().collect::<BTreeSet<_>>().into_iter().collect::<Vec<_>>();
        if unique_words.len() > u8::MAX as usize {
            return Err(format!("cannot pack {} into 8 bits", unique_words.len()));
        }
        // needed for the chunk mapping to work
        assert_eq!(unique_words[0], 0, "has a zero word");
        let canonicalized = Canonicalized::canonicalize(&unique_words);

        let word_indices = canonicalized.unique_mapping.clone();
        let compressed_words = words.iter().map(|w| word_indices[w]).collect::<Vec<u8>>();

        let mut best = None;
        for length in 1..=64 {
            let mut temp = self.clone();
            temp.emit_chunk_map(word_indices[&0], &compressed_words, length);
            if let Some((_, size)) = best {
                if temp.bytes_used < size {
                    best = Some((length, temp.bytes_used));
                }
            } else {
                best = Some((length, temp.bytes_used));
            }
        }
        self.emit_chunk_map(word_indices[&0], &compressed_words, best.unwrap().0);

        struct Bits(u64);
        impl fmt::Debug for Bits {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "0b{:064b}", self.0)
            }
        }

        writeln!(
            &mut self.file,
            "static BITSET_CANONICAL: [u64; {}] = [{}];",
            canonicalized.canonical_words.len(),
            fmt_list(canonicalized.canonical_words.iter().map(|v| Bits(*v))),
        )
        .unwrap();
        self.bytes_used += 8 * canonicalized.canonical_words.len();
        writeln!(
            &mut self.file,
            "static BITSET_MAPPING: [(u8, u8); {}] = [{}];",
            canonicalized.canonicalized_words.len(),
            fmt_list(&canonicalized.canonicalized_words),
        )
        .unwrap();
        // 8 bit index into shifted words, 7 bits for shift + optional flip
        // We only need it for the words that we removed by applying a shift and
        // flip to them.
        self.bytes_used += 2 * canonicalized.canonicalized_words.len();

        self.blank_line();

        writeln!(&mut self.file, "pub fn lookup(c: char) -> bool {{").unwrap();
        writeln!(&mut self.file, "    super::bitset_search(",).unwrap();
        writeln!(&mut self.file, "        c as u32,").unwrap();
        writeln!(&mut self.file, "        &BITSET_CHUNKS_MAP,").unwrap();
        writeln!(&mut self.file, "        &BITSET_INDEX_CHUNKS,").unwrap();
        writeln!(&mut self.file, "        &BITSET_CANONICAL,").unwrap();
        writeln!(&mut self.file, "        &BITSET_MAPPING,").unwrap();
        writeln!(&mut self.file, "    )").unwrap();
        writeln!(&mut self.file, "}}").unwrap();

        Ok(())
    }

    fn emit_chunk_map(&mut self, zero_at: u8, compressed_words: &[u8], chunk_length: usize) {
        let mut compressed_words = compressed_words.to_vec();
        for _ in 0..(chunk_length - (compressed_words.len() % chunk_length)) {
            // pad out bitset index with zero words so we have all chunks of
            // chunkchunk_length
            compressed_words.push(zero_at);
        }

        let mut chunks = BTreeSet::new();
        for chunk in compressed_words.chunks(chunk_length) {
            chunks.insert(chunk);
        }
        let chunk_map =
            chunks.iter().enumerate().map(|(idx, &chunk)| (chunk, idx)).collect::<HashMap<_, _>>();
        let mut chunk_indices = Vec::new();
        for chunk in compressed_words.chunks(chunk_length) {
            chunk_indices.push(chunk_map[chunk]);
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
            "static BITSET_INDEX_CHUNKS: [[u8; {}]; {}] = [{}];",
            chunk_length,
            chunks.len(),
            fmt_list(chunks.iter()),
        )
        .unwrap();
        self.bytes_used += chunk_length * chunks.len();
    }
}

pub fn emit_codepoints(emitter: &mut RawEmitter, ranges: &[Range<u32>]) {
    emitter.blank_line();

    let mut bitset = emitter.clone();
    let bitset_ok = bitset.emit_bitset(&ranges).is_ok();

    let mut skiplist = emitter.clone();
    skiplist.emit_skiplist(&ranges);

    if bitset_ok && bitset.bytes_used <= skiplist.bytes_used {
        *emitter = bitset;
        emitter.desc = String::from("bitset");
    } else {
        *emitter = skiplist;
        emitter.desc = String::from("skiplist");
    }
}

pub fn emit_whitespace(emitter: &mut RawEmitter, ranges: &[Range<u32>]) {
    emitter.blank_line();

    let mut cascading = emitter.clone();
    cascading.emit_cascading_map(&ranges);
    *emitter = cascading;
    emitter.desc = String::from("cascading");
}

struct Canonicalized {
    canonical_words: Vec<u64>,
    canonicalized_words: Vec<(u8, u8)>,

    /// Maps an input unique word to the associated index (u8) which is into
    /// canonical_words or canonicalized_words (in order).
    unique_mapping: HashMap<u64, u8>,
}

impl Canonicalized {
    fn canonicalize(unique_words: &[u64]) -> Self {
        #[derive(Copy, Clone, Debug)]
        enum Mapping {
            Rotate(u32),
            Invert,
            RotateAndInvert(u32),
            ShiftRight(u32),
        }

        // key is the word being mapped to
        let mut mappings: BTreeMap<u64, Vec<(u64, Mapping)>> = BTreeMap::new();
        for &a in unique_words {
            'b: for &b in unique_words {
                // skip self
                if a == b {
                    continue;
                }

                // All possible distinct rotations
                for rotation in 1..64 {
                    if a.rotate_right(rotation) == b {
                        mappings.entry(b).or_default().push((a, Mapping::Rotate(rotation)));
                        // We're not interested in further mappings between a and b
                        continue 'b;
                    }
                }

                if (!a) == b {
                    mappings.entry(b).or_default().push((a, Mapping::Invert));
                    // We're not interested in further mappings between a and b
                    continue 'b;
                }

                // All possible distinct rotations, inverted
                for rotation in 1..64 {
                    if (!a.rotate_right(rotation)) == b {
                        mappings
                            .entry(b)
                            .or_default()
                            .push((a, Mapping::RotateAndInvert(rotation)));
                        // We're not interested in further mappings between a and b
                        continue 'b;
                    }
                }

                // All possible shifts
                for shift_by in 1..64 {
                    if a == (b >> shift_by) {
                        mappings
                            .entry(b)
                            .or_default()
                            .push((a, Mapping::ShiftRight(shift_by as u32)));
                        // We're not interested in further mappings between a and b
                        continue 'b;
                    }
                }
            }
        }
        // These are the bitset words which will be represented "raw" (as a u64)
        let mut canonical_words = Vec::new();
        // These are mapped words, which will be represented by an index into
        // the canonical_words and a Mapping; u16 when encoded.
        let mut canonicalized_words = Vec::new();
        let mut unique_mapping = HashMap::new();

        #[derive(Debug, PartialEq, Eq)]
        enum UniqueMapping {
            Canonical(usize),
            Canonicalized(usize),
        }

        // Map 0 first, so that it is the first canonical word.
        // This is realistically not inefficient because 0 is not mapped to by
        // anything else (a shift pattern could do it, but would be wasteful).
        //
        // However, 0s are quite common in the overall dataset, and it is quite
        // wasteful to have to go through a mapping function to determine that
        // we have a zero.
        //
        // FIXME: Experiment with choosing most common words in overall data set
        // for canonical when possible.
        while let Some((&to, _)) = mappings
            .iter()
            .find(|(&to, _)| to == 0)
            .or_else(|| mappings.iter().max_by_key(|m| m.1.len()))
        {
            // Get the mapping with the most entries. Currently, no mapping can
            // only exist transitively (i.e., there is no A, B, C such that A
            // does not map to C and but A maps to B maps to C), so this is
            // guaranteed to be acceptable.
            //
            // In the future, we may need a more sophisticated algorithm to
            // identify which keys to prefer as canonical.
            let mapped_from = mappings.remove(&to).unwrap();
            for (from, how) in &mapped_from {
                // Remove the entries which mapped to this one.
                // Noting that it should be associated with the Nth canonical word.
                //
                // We do not assert that this is present, because there may be
                // no mappings to the `from` word; that's fine.
                mappings.remove(from);
                assert_eq!(
                    unique_mapping
                        .insert(*from, UniqueMapping::Canonicalized(canonicalized_words.len())),
                    None
                );
                canonicalized_words.push((canonical_words.len(), *how));

                // Remove the now-canonicalized word from other mappings,
                // to ensure that we deprioritize them in the next iteration of
                // the while loop.
                for mapped in mappings.values_mut() {
                    let mut i = 0;
                    while i != mapped.len() {
                        if mapped[i].0 == *from {
                            mapped.remove(i);
                        } else {
                            i += 1;
                        }
                    }
                }
            }
            assert!(
                unique_mapping
                    .insert(to, UniqueMapping::Canonical(canonical_words.len()))
                    .is_none()
            );
            canonical_words.push(to);

            // Remove the now-canonical word from other mappings, to ensure that
            // we deprioritize them in the next iteration of the while loop.
            for mapped in mappings.values_mut() {
                let mut i = 0;
                while i != mapped.len() {
                    if mapped[i].0 == to {
                        mapped.remove(i);
                    } else {
                        i += 1;
                    }
                }
            }
        }

        // Any words which we couldn't shrink, just stick into the canonical
        // words.
        //
        // FIXME: work harder -- there are more possibilities for mapping
        // functions (e.g., multiplication, shifting instead of rotation, etc.)
        // We'll probably always have some slack though so this loop will still
        // be needed.
        for &w in unique_words {
            if !unique_mapping.contains_key(&w) {
                assert!(
                    unique_mapping
                        .insert(w, UniqueMapping::Canonical(canonical_words.len()))
                        .is_none()
                );
                canonical_words.push(w);
            }
        }
        assert_eq!(canonicalized_words.len() + canonical_words.len(), unique_words.len());
        assert_eq!(unique_mapping.len(), unique_words.len());

        let unique_mapping = unique_mapping
            .into_iter()
            .map(|(key, value)| {
                (
                    key,
                    match value {
                        UniqueMapping::Canonicalized(idx) => {
                            u8::try_from(canonical_words.len() + idx).unwrap()
                        }
                        UniqueMapping::Canonical(idx) => u8::try_from(idx).unwrap(),
                    },
                )
            })
            .collect::<HashMap<_, _>>();

        let mut distinct_indices = BTreeSet::new();
        for &w in unique_words {
            let idx = unique_mapping.get(&w).unwrap();
            assert!(distinct_indices.insert(idx));
        }

        const LOWER_6: u32 = (1 << 6) - 1;

        let canonicalized_words = canonicalized_words
            .into_iter()
            .map(|v| {
                (
                    u8::try_from(v.0).unwrap(),
                    match v.1 {
                        Mapping::RotateAndInvert(amount) => {
                            assert_eq!(amount, amount & LOWER_6);
                            1 << 6 | (amount as u8)
                        }
                        Mapping::Rotate(amount) => {
                            assert_eq!(amount, amount & LOWER_6);
                            amount as u8
                        }
                        Mapping::Invert => 1 << 6,
                        Mapping::ShiftRight(shift_by) => {
                            assert_eq!(shift_by, shift_by & LOWER_6);
                            1 << 7 | (shift_by as u8)
                        }
                    },
                )
            })
            .collect::<Vec<(u8, u8)>>();
        Canonicalized { unique_mapping, canonical_words, canonicalized_words }
    }
}
