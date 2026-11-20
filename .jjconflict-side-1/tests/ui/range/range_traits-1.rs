use std::ops::*;

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
struct AllTheRanges {
    a: Range<usize>,
    //~^ ERROR Ord
    b: RangeTo<usize>,
    //~^ ERROR Ord
    c: RangeFrom<usize>,
    //~^ ERROR Ord
    d: RangeFull,
    //~^ ERROR Ord
    e: RangeInclusive<usize>,
    //~^ ERROR Ord
    f: RangeToInclusive<usize>,
    //~^ ERROR Ord
}

fn main() {}
