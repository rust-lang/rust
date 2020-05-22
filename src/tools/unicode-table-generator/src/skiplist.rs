use crate::fmt_list;
use crate::raw_emitter::RawEmitter;
use std::convert::TryInto;
use std::fmt::Write as _;
use std::ops::Range;

/// This will get packed into a single u32 before inserting into the data set.
#[derive(Debug, PartialEq)]
struct ShortOffsetRunHeader {
    /// Note, we only allow for 21 bits here.
    prefix_sum: u32,

    /// Note, we actually only allow for 11 bits here. This should be enough --
    /// our largest sets are around ~1400 offsets long.
    start_idx: u16,
}

impl ShortOffsetRunHeader {
    fn pack(&self) -> u32 {
        assert!(self.start_idx < (1 << 11));
        assert!(self.prefix_sum < (1 << 21));

        (self.start_idx as u32) << 21 | self.prefix_sum
    }
}

impl RawEmitter {
    pub fn emit_skiplist(&mut self, ranges: &[Range<u32>]) {
        let mut offsets = Vec::<u32>::new();
        let points = ranges.iter().flat_map(|r| vec![r.start, r.end]).collect::<Vec<u32>>();
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
                        start_idx: start.try_into().unwrap(),
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

        writeln!(
            &mut self.file,
            "static SHORT_OFFSET_RUNS: [u32; {}] = [{}];",
            short_offset_runs.len(),
            fmt_list(short_offset_runs.iter().map(|v| v.pack()))
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

        writeln!(&mut self.file, "pub fn lookup(c: char) -> bool {{").unwrap();
        writeln!(&mut self.file, "    super::skip_search(",).unwrap();
        writeln!(&mut self.file, "        c as u32,").unwrap();
        writeln!(&mut self.file, "        &SHORT_OFFSET_RUNS,").unwrap();
        writeln!(&mut self.file, "        &OFFSETS,").unwrap();
        writeln!(&mut self.file, "    )").unwrap();
        writeln!(&mut self.file, "}}").unwrap();
    }
}
