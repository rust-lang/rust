use crate::fmt_list;
use crate::raw_emitter::RawEmitter;
use std::collections::HashMap;
use std::fmt::Write as _;
use std::ops::Range;

impl RawEmitter {
    pub fn emit_cascading_map(&mut self, ranges: &[Range<u32>]) -> bool {
        let mut map: [u8; 256] = [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ];

        let points = ranges
            .iter()
            .flat_map(|r| (r.start..r.end).into_iter().collect::<Vec<u32>>())
            .collect::<Vec<u32>>();

        println!("there are {} points", points.len());

        // how many distinct ranges need to be counted?
        let mut codepoints_by_high_bytes = HashMap::<usize, Vec<u32>>::new();
        for point in points {
            // assert that there is no whitespace over the 0x3000 range.
            assert!(point <= 0x3000, "the highest unicode whitespace value has changed");
            let high_bytes = point as usize >> 8;
            let codepoints = codepoints_by_high_bytes.entry(high_bytes).or_insert_with(Vec::new);
            codepoints.push(point);
        }

        let mut bit_for_high_byte = 1u8;
        let mut arms = Vec::<String>::new();

        let mut high_bytes: Vec<usize> =
            codepoints_by_high_bytes.keys().map(|k| k.clone()).collect();
        high_bytes.sort();
        for high_byte in high_bytes {
            let codepoints = codepoints_by_high_bytes.get_mut(&high_byte).unwrap();
            if codepoints.len() == 1 {
                let ch = codepoints.pop().unwrap();
                arms.push(format!("{} => c as u32 == {:#04x}", high_byte, ch));
                continue;
            }
            // more than 1 codepoint in this arm
            for codepoint in codepoints {
                map[(*codepoint & 0xff) as usize] |= bit_for_high_byte;
            }
            arms.push(format!(
                "{} => WHITESPACE_MAP[c as usize & 0xff] & {} != 0",
                high_byte, bit_for_high_byte
            ));
            bit_for_high_byte <<= 1;
        }

        writeln!(&mut self.file, "static WHITESPACE_MAP: [u8; 256] = [{}];", fmt_list(map.iter()))
            .unwrap();
        self.bytes_used += 256;

        writeln!(&mut self.file, "#[inline]").unwrap();
        writeln!(&mut self.file, "pub fn lookup(c: char) -> bool {{").unwrap();
        writeln!(&mut self.file, "    match c as u32 >> 8 {{").unwrap();
        for arm in arms {
            writeln!(&mut self.file, "        {},", arm).unwrap();
        }
        writeln!(&mut self.file, "        _ => false,").unwrap();
        writeln!(&mut self.file, "    }}").unwrap();
        writeln!(&mut self.file, "}}").unwrap();

        true
    }
}
