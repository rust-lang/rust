#![warn(clippy::reversed_empty_ranges)]

const ANSWER: i32 = 42;
const SOME_NUM: usize = 3;

fn main() {
    let _ = (42 + 10..42 + 10).map(|x| x / 2).find(|&x| x == 21);

    let arr = [1, 2, 3, 4, 5];
    let _ = &arr[3usize..=1usize];
    let _ = &arr[SOME_NUM..1];
    let _ = &arr[3..3];

    for _ in ANSWER..ANSWER {}
}
