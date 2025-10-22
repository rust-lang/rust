fn main() {
    let mut vec = vec![1, 2, 3, 4, 5];
    vec.drain([..3]);
    //~^ ERROR: the trait bound `[RangeTo<{integer}>; 1]: RangeBounds<usize>` is not satisfied [E0277]
    //~| NOTE: the trait `RangeBounds<usize>` is not implemented for `[RangeTo<{integer}>; 1]`
    //~| NOTE: required by a bound introduced by this call
    //~| NOTE: required by a bound in `Vec::<T, A>::drain`
    //~| HELP: consider removing `[]`
}
