// build-pass (FIXME(62277): could be check-pass?)

#![deny(warnings)]

use std::collections::BTreeMap;

pub struct RangeMap {
    map: BTreeMap<Range, u8>,
}

#[derive(Eq, PartialEq, Ord, PartialOrd)]
struct Range;

impl RangeMap {
    fn iter_with_range<'a>(&'a self) -> impl Iterator<Item = (&'a Range, &'a u8)> + 'a {
        self.map.range(Range..Range)
    }

    pub fn iter<'a>(&'a self) -> impl Iterator<Item = &'a u8> + 'a {
        self.iter_with_range().map(|(_, data)| data)
    }

}

fn main() {}
