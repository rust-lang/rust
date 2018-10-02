use std::{
    fmt, ops,
};

use {
    SyntaxNodeRef, TextRange, TextUnit,
    text_utils::{intersect, contains_offset_nonstrict},
};

#[derive(Clone)]
pub struct SyntaxText<'a> {
    node: SyntaxNodeRef<'a>,
    range: TextRange,
}

impl<'a> SyntaxText<'a> {
    pub(crate) fn new(node: SyntaxNodeRef<'a>) -> SyntaxText<'a> {
        SyntaxText {
            node,
            range: node.range()
        }
    }
    pub fn chunks(&self) -> impl Iterator<Item=&'a str> {
        let range = self.range;
        self.node
            .descendants()
            .filter_map(move |node| {
                let text = node.leaf_text()?;
                let range = intersect(range, node.range())?;
                let range = range - node.range().start();
                Some(&text[range])
            })
    }
    pub fn push_to(&self, buf: &mut String) {
        self.chunks().for_each(|it| buf.push_str(it));
    }
    pub fn to_string(&self) -> String {
        self.chunks().collect()
    }
    pub fn contains(&self, c: char) -> bool {
        self.chunks().any(|it| it.contains(c))
    }
    pub fn find(&self, c: char) -> Option<TextUnit> {
        let mut acc: TextUnit = 0.into();
        for chunk in self.chunks() {
            if let Some(pos) = chunk.find(c) {
                let pos: TextUnit = (pos as u32).into();
                return Some(acc + pos);
            }
            acc += TextUnit::of_str(chunk);
        }
        None
    }
    pub fn len(&self) -> TextUnit {
        self.range.len()
    }
    pub fn slice(&self, range: impl SyntaxTextSlice) -> SyntaxText<'a> {
        let range = range.restrict(self.range)
            .unwrap_or_else(|| {
                panic!("invalid slice, range: {:?}, slice: {:?}", self.range, range)
            });
        SyntaxText { node: self.node, range }
    }
    pub fn char_at(&self, offset: TextUnit) -> Option<char> {
        let mut start: TextUnit = 0.into();
        for chunk in self.chunks() {
            let end = start + TextUnit::of_str(chunk);
            if start <= offset && offset < end {
                let off: usize = u32::from(offset - start) as usize;
                return Some(chunk[off..].chars().next().unwrap());
            }
            start = end;
        }
        None
    }
}

impl<'a> fmt::Debug for SyntaxText<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(&self.to_string(), f)
    }
}

impl<'a> fmt::Display for SyntaxText<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(&self.to_string(), f)
    }
}

pub trait SyntaxTextSlice: fmt::Debug {
    fn restrict(&self, range: TextRange) -> Option<TextRange>;
}

impl SyntaxTextSlice for TextRange {
    fn restrict(&self, range: TextRange) -> Option<TextRange> {
        intersect(*self, range)
    }
}

impl SyntaxTextSlice for ops::RangeTo<TextUnit> {
    fn restrict(&self, range: TextRange) -> Option<TextRange> {
        if !contains_offset_nonstrict(range, self.end) {
            return None;
        }
        Some(TextRange::from_to(range.start(), self.end))
    }
}

impl SyntaxTextSlice for ops::RangeFrom<TextUnit> {
    fn restrict(&self, range: TextRange) -> Option<TextRange> {
        if !contains_offset_nonstrict(range, self.start) {
            return None;
        }
        Some(TextRange::from_to(self.start, range.end()))
    }
}

impl SyntaxTextSlice for ops::Range<TextUnit> {
    fn restrict(&self, range: TextRange) -> Option<TextRange> {
        TextRange::from_to(self.start, self.end).restrict(range)
    }
}
