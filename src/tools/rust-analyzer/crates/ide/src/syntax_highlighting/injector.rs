//! Extracts a subsequence of a text document, remembering the mapping of ranges
//! between original and extracted texts.
use std::ops::{self, Sub};

use stdx::equal_range_by;
use syntax::{TextRange, TextSize};

#[derive(Default)]
pub(super) struct Injector {
    buf: String,
    ranges: Vec<(TextRange, Option<Delta<TextSize>>)>,
}

impl Injector {
    pub(super) fn add(&mut self, text: &str, source_range: TextRange) {
        let len = TextSize::of(text);
        assert_eq!(len, source_range.len());
        self.add_impl(text, Some(source_range.start()));
    }

    pub(super) fn add_unmapped(&mut self, text: &str) {
        self.add_impl(text, None);
    }

    fn add_impl(&mut self, text: &str, source: Option<TextSize>) {
        let len = TextSize::of(text);
        let target_range = TextRange::at(TextSize::of(&self.buf), len);
        self.ranges.push((target_range, source.map(|it| Delta::new(target_range.start(), it))));
        self.buf.push_str(text);
    }

    pub(super) fn take_text(&mut self) -> String {
        std::mem::take(&mut self.buf)
    }

    pub(super) fn map_range_up(&self, range: TextRange) -> impl Iterator<Item = TextRange> + '_ {
        equal_range_by(&self.ranges, |&(r, _)| TextRange::ordering(r, range)).filter_map(move |i| {
            let (target_range, delta) = self.ranges[i];
            let intersection = target_range.intersect(range).unwrap();
            Some(intersection + delta?)
        })
    }
}

#[derive(Clone, Copy)]
enum Delta<T> {
    Add(T),
    Sub(T),
}

impl<T> Delta<T> {
    fn new(from: T, to: T) -> Delta<T>
    where
        T: Ord + Sub<Output = T>,
    {
        if to >= from { Delta::Add(to - from) } else { Delta::Sub(from - to) }
    }
}

impl ops::Add<Delta<TextSize>> for TextSize {
    type Output = TextSize;

    fn add(self, rhs: Delta<TextSize>) -> TextSize {
        match rhs {
            Delta::Add(it) => self + it,
            Delta::Sub(it) => self - it,
        }
    }
}

impl ops::Add<Delta<TextSize>> for TextRange {
    type Output = TextRange;

    fn add(self, rhs: Delta<TextSize>) -> TextRange {
        TextRange::at(self.start() + rhs, self.len())
    }
}
