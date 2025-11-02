fn main() {
    let mut vec = vec![1, 2, 3, 4, 5];
    vec.drain([..3]);
    //~^ ERROR: the trait bound `[RangeTo<{integer}>; 1]: RangeBounds<usize>` is not satisfied [E0277]
    //~| SUGGESTION: ..3

    // Should not produce this suggestion:

    let range = [1..5];
    let mut vec = vec![1, 2, 3, 4, 5];
    vec.drain(range);
    //~^ ERROR: the trait bound `[std::ops::Range<{integer}>; 1]: RangeBounds<usize>` is not satisfied [E0277]

    take_array_of_one_range_bounds([..3]);

    take_array_of_one_different_trait([5..10]);
    //~^ ERROR: no implementation for `std::ops::Range<{integer}> << usize` [E0277]

    take_different_trait([5..10]);
    //~^ ERROR: no implementation for `[std::ops::Range<{integer}>; 1] << usize` [E0277]
}

fn take_array_of_one_range_bounds<T: std::ops::RangeBounds<usize>>(_arr: [T; 1]) {}

fn take_array_of_one_different_trait<T: std::ops::Shl<usize>>(_arr: [T; 1]) {}

fn take_different_trait<T: std::ops::Shl<usize>>(_ty: T) {}
