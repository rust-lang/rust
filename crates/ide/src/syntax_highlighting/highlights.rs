//! Collects a tree of highlighted ranges and flattens it.
use std::{cmp::Ordering, iter};

use stdx::equal_range_by;
use syntax::TextRange;

use crate::{HighlightTag, HighlightedRange};

pub(super) struct Highlights {
    root: Node,
}

struct Node {
    highlighted_range: HighlightedRange,
    nested: Vec<Node>,
}

impl Highlights {
    pub(super) fn new(range: TextRange) -> Highlights {
        Highlights {
            root: Node::new(HighlightedRange {
                range,
                highlight: HighlightTag::Dummy.into(),
                binding_hash: None,
            }),
        }
    }

    pub(super) fn add(&mut self, highlighted_range: HighlightedRange) {
        self.root.add(highlighted_range);
    }

    pub(super) fn to_vec(self) -> Vec<HighlightedRange> {
        let mut res = Vec::new();
        self.root.flatten(&mut res);
        res
    }
}

impl Node {
    fn new(highlighted_range: HighlightedRange) -> Node {
        Node { highlighted_range, nested: Vec::new() }
    }

    fn add(&mut self, highlighted_range: HighlightedRange) {
        assert!(self.highlighted_range.range.contains_range(highlighted_range.range));

        // Fast path
        if let Some(last) = self.nested.last_mut() {
            if last.highlighted_range.range.contains_range(highlighted_range.range) {
                return last.add(highlighted_range);
            }
            if last.highlighted_range.range.end() <= highlighted_range.range.start() {
                return self.nested.push(Node::new(highlighted_range));
            }
        }

        let (start, len) = equal_range_by(&self.nested, |n| {
            ordering(n.highlighted_range.range, highlighted_range.range)
        });

        if len == 1
            && self.nested[start].highlighted_range.range.contains_range(highlighted_range.range)
        {
            return self.nested[start].add(highlighted_range);
        }

        let nested = self
            .nested
            .splice(start..start + len, iter::once(Node::new(highlighted_range)))
            .collect::<Vec<_>>();
        self.nested[start].nested = nested;
    }

    fn flatten(&self, acc: &mut Vec<HighlightedRange>) {
        let mut start = self.highlighted_range.range.start();
        let mut nested = self.nested.iter();
        loop {
            let next = nested.next();
            let end = next.map_or(self.highlighted_range.range.end(), |it| {
                it.highlighted_range.range.start()
            });
            if start < end {
                acc.push(HighlightedRange {
                    range: TextRange::new(start, end),
                    highlight: self.highlighted_range.highlight,
                    binding_hash: self.highlighted_range.binding_hash,
                });
            }
            start = match next {
                Some(child) => {
                    child.flatten(acc);
                    child.highlighted_range.range.end()
                }
                None => break,
            }
        }
    }
}

pub(super) fn ordering(r1: TextRange, r2: TextRange) -> Ordering {
    if r1.end() <= r2.start() {
        Ordering::Less
    } else if r2.end() <= r1.start() {
        Ordering::Greater
    } else {
        Ordering::Equal
    }
}
