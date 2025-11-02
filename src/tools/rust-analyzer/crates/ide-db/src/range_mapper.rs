//! Maps between ranges in documents.

use std::cmp::Ordering;

use stdx::equal_range_by;
use syntax::{TextRange, TextSize};

#[derive(Default)]
pub struct RangeMapper {
    buf: String,
    ranges: Vec<(TextRange, Option<TextRange>)>,
}

impl RangeMapper {
    pub fn add(&mut self, text: &str, source_range: TextRange) {
        let len = TextSize::of(text);
        assert_eq!(len, source_range.len());
        self.add_impl(text, Some(source_range.start()));
    }

    pub fn add_unmapped(&mut self, text: &str) {
        self.add_impl(text, None);
    }

    fn add_impl(&mut self, text: &str, source: Option<TextSize>) {
        let len = TextSize::of(text);
        let target_range = TextRange::at(TextSize::of(&self.buf), len);
        self.ranges.push((target_range, source.map(|it| TextRange::at(it, len))));
        self.buf.push_str(text);
    }

    pub fn take_text(&mut self) -> String {
        std::mem::take(&mut self.buf)
    }

    pub fn map_range_up(&self, range: TextRange) -> impl Iterator<Item = TextRange> + '_ {
        equal_range_by(&self.ranges, |&(r, _)| {
            if range.is_empty() && r.contains(range.start()) {
                Ordering::Equal
            } else {
                TextRange::ordering(r, range)
            }
        })
        .filter_map(move |i| {
            let (target_range, source_range) = self.ranges[i];
            let intersection = target_range.intersect(range).unwrap();
            let source_range = source_range?;
            Some(intersection - target_range.start() + source_range.start())
        })
    }

    pub fn map_offset_down(&self, offset: TextSize) -> Option<TextSize> {
        // Using a binary search here is a bit complicated because of the `None` entries.
        // But the number of lines in fixtures is usually low.
        let (target_range, source_range) =
            self.ranges.iter().find_map(|&(target_range, source_range)| {
                let source_range = source_range?;
                if !source_range.contains(offset) {
                    return None;
                }
                Some((target_range, source_range))
            })?;
        Some(offset - source_range.start() + target_range.start())
    }
}
