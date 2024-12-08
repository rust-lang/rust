#![warn(clippy::reversed_empty_ranges)]

const ANSWER: i32 = 42;
const SOME_NUM: usize = 3;

fn main() {
    let arr = [1, 2, 3, 4, 5];
    let _ = &arr[3usize..=1usize];
    //~^ ERROR: this range is reversed and using it to index a slice will panic at run-tim
    //~| NOTE: `-D clippy::reversed-empty-ranges` implied by `-D warnings`
    let _ = &arr[SOME_NUM..1];
    //~^ ERROR: this range is reversed and using it to index a slice will panic at run-tim

    for _ in ANSWER..ANSWER {}
    //~^ ERROR: this range is empty so it will yield no values

    // Should not be linted, see issue #5689
    let _ = (42 + 10..42 + 10).map(|x| x / 2).find(|&x| x == 21);
}
