//@ run-rustfix

fn main() {
    let mut vec = vec![1, 2, 3, 4, 5];

    vec.drain([..3]);
    //~^ ERROR: the trait bound `[RangeTo<{integer}>; 1]: RangeBounds<usize>` is not satisfied [E0277]
    //~| SUGGESTION: ..3

    vec.drain([..=3]);
    //~^ ERROR: the trait bound `[std::ops::RangeToInclusive<{integer}>; 1]: RangeBounds<usize>` is not satisfied [E0277]
    //~| SUGGESTION: ..=3

    vec.drain([1..3]);
    //~^ ERROR: the trait bound `[std::ops::Range<{integer}>; 1]: RangeBounds<usize>` is not satisfied [E0277]
    //~| SUGGESTION: 1..3

    vec.drain([1..=3]);
    //~^ ERROR: the trait bound `[std::ops::RangeInclusive<{integer}>; 1]: RangeBounds<usize>` is not satisfied [E0277]
    //~| SUGGESTION: 1..=3

    vec.drain([1..]);
    //~^ ERROR: the trait bound `[std::ops::RangeFrom<{integer}>; 1]: RangeBounds<usize>` is not satisfied [E0277]
    //~| SUGGESTION: 1..

    vec.drain([..]);
    //~^ ERROR: the trait bound `[RangeFull; 1]: RangeBounds<usize>` is not satisfied [E0277]
    //~| SUGGESTION: ..
}
