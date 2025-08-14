use std::fmt::{self, Write as _};
use std::ops::Range;

use crate::fmt_list;
use crate::raw_emitter::RawEmitter;

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

        writeln!(&mut self.file, "use super::ShortOffsetRunHeader;\n").unwrap();
        writeln!(
            &mut self.file,
            "static SHORT_OFFSET_RUNS: [ShortOffsetRunHeader; {}] = [{}];",
            short_offset_runs.len(),
            fmt_list(short_offset_runs.iter())
        )
        .unwrap();
        self.bytes_used += 4 * short_offset_runs.len();
        writeln!(
            &mut self.file,
            "static OFFSETS: [u8; {}] = [{}];",
            coded_offsets.len(),
            fmt_list(&coded_offsets)
        )
        .unwrap();
        self.bytes_used += coded_offsets.len();

        // The inlining in this code works like the following:
        //
        // The `skip_search` function is always inlined into the parent `lookup` fn,
        // thus the compiler can generate optimal code based on the referenced `static`s.
        //
        // In the case of ASCII optimization, the lower-bounds check is inlined into
        // the caller, and slower-path `skip_search` is outlined into a separate `lookup_slow` fn.
        //
        // Thus, in both cases, the `skip_search` function is specialized for the `static`s,
        // and outlined into the prebuilt `std`.
        if first_code_point > 0x7f {
            writeln!(&mut self.file, "#[inline]").unwrap();
            writeln!(&mut self.file, "pub fn lookup(c: char) -> bool {{").unwrap();
            writeln!(&mut self.file, "    debug_assert!(!c.is_ascii());").unwrap();
            writeln!(&mut self.file, "    (c as u32) >= {first_code_point:#04x} && lookup_slow(c)")
                .unwrap();
            writeln!(&mut self.file, "}}").unwrap();
            writeln!(&mut self.file).unwrap();
            writeln!(&mut self.file, "#[inline(never)]").unwrap();
            writeln!(&mut self.file, "fn lookup_slow(c: char) -> bool {{").unwrap();
        } else {
            writeln!(&mut self.file, "pub fn lookup(c: char) -> bool {{").unwrap();
            writeln!(&mut self.file, "    debug_assert!(!c.is_ascii());").unwrap();
        }
        writeln!(&mut self.file, "    const {{").unwrap();
        writeln!(
            &mut self.file,
            "        assert!(SHORT_OFFSET_RUNS.last().unwrap().0 > char::MAX as u32);",
        )
        .unwrap();
        writeln!(&mut self.file, "        let mut i = 0;").unwrap();
        writeln!(&mut self.file, "        while i < SHORT_OFFSET_RUNS.len() {{").unwrap();
        writeln!(
            &mut self.file,
            "            assert!(SHORT_OFFSET_RUNS[i].start_index() < OFFSETS.len());",
        )
        .unwrap();
        writeln!(&mut self.file, "            i += 1;").unwrap();
        writeln!(&mut self.file, "        }}").unwrap();
        writeln!(&mut self.file, "    }}").unwrap();
        writeln!(
            &mut self.file,
            "    // SAFETY: We just ensured the last element of `SHORT_OFFSET_RUNS` is greater than `std::char::MAX`",
        )
        .unwrap();
        writeln!(
            &mut self.file,
            "    // and the start indices of all elements in `SHORT_OFFSET_RUNS` are smaller than `OFFSETS.len()`.",
        )
        .unwrap();
        writeln!(
            &mut self.file,
            "    unsafe {{ super::skip_search(c, &SHORT_OFFSET_RUNS, &OFFSETS) }}"
        )
        .unwrap();
        writeln!(&mut self.file, "}}").unwrap();
    }
}
