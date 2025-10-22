fn main() {
    let mut vec = vec![1, 2, 3, 4, 5];
    vec.drain([..3]);
    //~^ ERROR: the trait bound `[RangeTo<{integer}>; 1]: RangeBounds<usize>` is not satisfied [E0277]
    //~| NOTE: the trait `RangeBounds<usize>` is not implemented for `[RangeTo<{integer}>; 1]`
    //~| NOTE: required by a bound introduced by this call
    //~| NOTE: required by a bound in `Vec::<T, A>::drain`
    //~| HELP: consider removing `[]`

    // Should not produce this  suggestion:

    take_array_of_one_range_bounds([..3]);

    take_array_of_one_different_trait([5..10]);
    //~^ ERROR: no implementation for `std::ops::Range<{integer}> << usize` [E0277]
    //~| NOTE: no implementation for `std::ops::Range<{integer}> << usize`
    //~| NOTE: required by a bound introduced by this call
    //~| HELP: the trait `Shl<usize>` is not implemented for `std::ops::Range<{integer}>`

    take_different_trait([5..10]);
    //~^ ERROR: no implementation for `[std::ops::Range<{integer}>; 1] << usize` [E0277]
    //~| NOTE: no implementation for `[std::ops::Range<{integer}>; 1] << usize`
    //~| NOTE: required by a bound introduced by this call
    //~| HELP: the trait `Shl<usize>` is not implemented for `[std::ops::Range<{integer}>; 1]`
}

fn take_array_of_one_range_bounds<T: std::ops::RangeBounds<usize>>(_arr: [T; 1]) {}

fn take_array_of_one_different_trait<T: std::ops::Shl<usize>>(_arr: [T; 1]) {}
//~^ NOTE: required by a bound in `take_array_of_one_different_trait`
//~| NOTE: required by this bound in `take_array_of_one_different_trait`

fn take_different_trait<T: std::ops::Shl<usize>>(_ty: T) {}
//~^ NOTE: required by a bound in `take_different_trait`
//~| NOTE: required by this bound in `take_different_trait`
