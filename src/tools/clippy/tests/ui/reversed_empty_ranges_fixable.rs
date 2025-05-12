#![warn(clippy::reversed_empty_ranges)]
#![allow(clippy::uninlined_format_args)]

const ANSWER: i32 = 42;

fn main() {
    // These should be linted:

    (42..=21).for_each(|x| println!("{}", x));
    //~^ reversed_empty_ranges
    let _ = (ANSWER..21).filter(|x| x % 2 == 0).take(10).collect::<Vec<_>>();
    //~^ reversed_empty_ranges

    for _ in -21..=-42 {}
    //~^ reversed_empty_ranges
    for _ in 42u32..21u32 {}
    //~^ reversed_empty_ranges

    // These should be ignored as they are not empty ranges:

    (21..=42).for_each(|x| println!("{}", x));
    (21..42).for_each(|x| println!("{}", x));

    let arr = [1, 2, 3, 4, 5];
    let _ = &arr[1..=3];
    let _ = &arr[1..3];

    for _ in 21..=42 {}
    for _ in 21..42 {}

    // This range is empty but should be ignored, see issue #5689
    let _ = &arr[0..0];
}
