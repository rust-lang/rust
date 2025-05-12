#![warn(clippy::reversed_empty_ranges)]

const ANSWER: i32 = 42;
const SOME_NUM: usize = 3;

fn main() {
    let arr = [1, 2, 3, 4, 5];
    let _ = &arr[3usize..=1usize];
    //~^ reversed_empty_ranges

    let _ = &arr[SOME_NUM..1];
    //~^ reversed_empty_ranges

    for _ in ANSWER..ANSWER {}
    //~^ reversed_empty_ranges

    // Should not be linted, see issue #5689
    let _ = (42 + 10..42 + 10).map(|x| x / 2).find(|&x| x == 21);
}
