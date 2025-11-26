use std::fmt::{self};
use std::ops::Range;

use crate::raw_emitter::RawEmitter;
use crate::writeln;

/// This will get packed into a single u32 before inserting into the data set.
#[derive(PartialEq)]
struct ShortOffsetRunHeader {
    /// Note, we actually only allow for 11 bits here. This should be enough --
    /// our largest sets are around ~1400 offsets long.
    start_index: u16,

    /// Note, we only allow for 21 bits here.
    prefix_sum: u32,
}

impl fmt::Debug for ShortOffsetRunHeader {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ShortOffsetRunHeader::new({}, {})", self.start_index, self.prefix_sum)
    }
}

impl RawEmitter {
    pub fn emit_skiplist(&mut self, ranges: &[Range<u32>]) {
        let first_code_point = ranges.first().unwrap().start;
        let mut offsets = Vec::<u32>::new();
        let points = ranges.iter().flat_map(|r| [r.start, r.end]).collect::<Vec<u32>>();
        let mut offset = 0;
        for pt in points {
            let delta = pt - offset;
            offsets.push(delta);
            offset = pt;
        }
        // Guaranteed to terminate, as it's impossible to subtract a value this
        // large from a valid char.
        offsets.push(std::char::MAX as u32 + 1);
        let mut coded_offsets: Vec<u8> = Vec::new();
        let mut short_offset_runs: Vec<ShortOffsetRunHeader> = vec![];
        let mut iter = offsets.iter().cloned();
        let mut prefix_sum = 0;
        loop {
            let mut any_elements = false;
            let mut inserted = false;
            let start = coded_offsets.len();
            for offset in iter.by_ref() {
                any_elements = true;
                prefix_sum += offset;
                if let Ok(offset) = offset.try_into() {
                    coded_offsets.push(offset);
                } else {
                    short_offset_runs.push(ShortOffsetRunHeader {
                        start_index: start.try_into().unwrap(),
                        prefix_sum,
                    });
                    // This is just needed to maintain indices even/odd
                    // correctly.
                    coded_offsets.push(0);
                    inserted = true;
                    break;
                }
            }
            if !any_elements {
                break;
            }
            // We always append the huge char::MAX offset to the end which
            // should never be able to fit into the u8 offsets.
            assert!(inserted);
        }

        self.bytes_used += 4 * short_offset_runs.len();
        self.bytes_used += coded_offsets.len();

        // The inlining in this code works like the following:
        //
        // The `skip_search` function is always inlined into the parent `lookup_slow` fn,
        // thus the compiler can generate optimal code based on the referenced `static`s.
        //
        // The lower-bounds check is inlined into the caller, and slower-path
        // `skip_search` is outlined into a separate `lookup_slow` fn.
        assert!(first_code_point > 0x7f);
        writeln!(self.file,
            "use super::ShortOffsetRunHeader;

            static SHORT_OFFSET_RUNS: [ShortOffsetRunHeader; {short_offset_runs_len}] = {short_offset_runs:?};
            static OFFSETS: [u8; {coded_offset_len}] = {coded_offsets:?};

            #[inline]
            pub fn lookup(c: char) -> bool {{
                debug_assert!(!c.is_ascii());
                (c as u32) >= {first_code_point:#04x} && lookup_slow(c)
            }}

            #[inline(never)]
            fn lookup_slow(c: char) -> bool {{
                const {{
                    assert!(SHORT_OFFSET_RUNS.last().unwrap().0 > char::MAX as u32);
                    let mut i = 0;
                    while i < SHORT_OFFSET_RUNS.len() {{
                        assert!(SHORT_OFFSET_RUNS[i].start_index() < OFFSETS.len());
                        i += 1;
                    }}
                }}
                // SAFETY: We just ensured the last element of `SHORT_OFFSET_RUNS` is greater than `std::char::MAX`
                // and the start indices of all elements in `SHORT_OFFSET_RUNS` are smaller than `OFFSETS.len()`.
                unsafe {{ super::skip_search(c, &SHORT_OFFSET_RUNS, &OFFSETS) }}
            }}",
            short_offset_runs_len = short_offset_runs.len(),
            coded_offset_len = coded_offsets.len(),
        );
    }
}
