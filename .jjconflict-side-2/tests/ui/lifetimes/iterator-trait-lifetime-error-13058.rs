//! Regression test for https://github.com/rust-lang/rust/issues/13058

use std::ops::Range;

trait Itble<'r, T, I: Iterator<Item=T>> { fn iter(&'r self) -> I; }

impl<'r> Itble<'r, usize, Range<usize>> for (usize, usize) {
    fn iter(&'r self) -> Range<usize> {
        let &(min, max) = self;
        min..max
    }
}

fn check<'r, I: Iterator<Item=usize>, T: Itble<'r, usize, I>>(cont: &T) -> bool
{
    let cont_iter = cont.iter();
//~^ ERROR explicit lifetime required in the type of `cont` [E0621]
    let result = cont_iter.fold(Some(0), |state, val| {
        state.map_or(None, |mask| {
            let bit = 1 << val;
            if mask & bit == 0 {Some(mask|bit)} else {None}
        })
    });
    result.is_some()
}

fn main() {
    check(&(3, 5));
}
