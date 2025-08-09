use std::fmt::{self, Write as _};
use std::ops::Range;

use crate::raw_emitter::RawEmitter;

impl RawEmitter {
    pub fn emit_match(&mut self, ranges: &[Range<u32>]) -> Result<(), fmt::Error> {
        let arms: Vec<_> = ranges
            .iter()
            .map(|range| match range.len() {
                1 => format!("{:#x} => true,", range.start),

                // minus one because inclusive range pattern
                _ => format!("{:#x}..={:#x} => true,", range.start, range.end - 1),
            })
            .collect();

        writeln!(self.file, "#[inline]")?;
        writeln!(self.file, "pub const fn lookup(c: char) -> bool {{")?;
        writeln!(self.file, "    debug_assert!(!c.is_ascii());")?;
        writeln!(self.file, "    match c as u32 {{")?;
        for arm in arms {
            writeln!(self.file, "        {arm}")?;
        }
        writeln!(self.file, "        _ => false,")?;
        writeln!(self.file, "    }}")?;
        writeln!(self.file, "}}")?;
        Ok(())
    }
}
