use std::{fmt, ops::{self, Bound}};

use crate::{SmolStr, SyntaxNode, TextRange, TextUnit, SyntaxElement};

#[derive(Clone)]
pub struct SyntaxText<'a> {
    node: &'a SyntaxNode,
    range: TextRange,
}

impl<'a> SyntaxText<'a> {
    pub(crate) fn new(node: &'a SyntaxNode) -> SyntaxText<'a> {
        SyntaxText { node, range: node.range() }
    }

    pub fn chunks(&self) -> impl Iterator<Item = &'a str> {
        let range = self.range;
        self.node.descendants_with_tokens().filter_map(move |el| match el {
            SyntaxElement::Token(t) => {
                let text = t.text();
                let range = range.intersection(&t.range())?;
                let range = range - t.range().start();
                Some(&text[range])
            }
            SyntaxElement::Node(_) => None,
        })
    }

    pub fn push_to(&self, buf: &mut String) {
        self.chunks().for_each(|it| buf.push_str(it));
    }

    pub fn to_string(&self) -> String {
        self.chunks().collect()
    }

    pub fn to_smol_string(&self) -> SmolStr {
        // FIXME: use `self.chunks().collect()` here too once
        // https://github.com/matklad/smol_str/pull/12 is merged and published
        self.to_string().into()
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

    /// NB, the offsets here are absolute, and this probably doesn't make sense!
    pub fn slice(&self, range: impl ops::RangeBounds<TextUnit>) -> SyntaxText<'a> {
        let start = match range.start_bound() {
            Bound::Included(b) => *b,
            Bound::Excluded(b) => *b + TextUnit::from(1u32),
            Bound::Unbounded => self.range.start(),
        };
        let end = match range.end_bound() {
            Bound::Included(b) => *b + TextUnit::from(1u32),
            Bound::Excluded(b) => *b,
            Bound::Unbounded => self.range.end(),
        };
        assert!(
            start <= end,
            "invalid slice, range: {:?}, slice: {:?}",
            self.range,
            (range.start_bound(), range.end_bound()),
        );
        let range = TextRange::from_to(start, end);
        assert!(
            range.is_subrange(&self.range),
            "invalid slice, range: {:?}, slice: {:?}",
            self.range,
            range,
        );
        SyntaxText { node: self.node, range }
    }

    pub fn char_at(&self, offset: impl Into<TextUnit>) -> Option<char> {
        let mut start: TextUnit = 0.into();
        let offset = offset.into();
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

impl From<SyntaxText<'_>> for String {
    fn from(text: SyntaxText) -> String {
        text.to_string()
    }
}

impl PartialEq<str> for SyntaxText<'_> {
    fn eq(&self, mut rhs: &str) -> bool {
        for chunk in self.chunks() {
            if !rhs.starts_with(chunk) {
                return false;
            }
            rhs = &rhs[chunk.len()..];
        }
        rhs.is_empty()
    }
}

impl PartialEq<&'_ str> for SyntaxText<'_> {
    fn eq(&self, rhs: &&str) -> bool {
        self == *rhs
    }
}
