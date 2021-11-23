// run-pass
#![allow(incomplete_features)]
#![feature(exclusive_range_pattern)]
#![feature(half_open_range_patterns)]
#![feature(inline_const_pat)]

fn main() {
    let mut if_lettable = vec![];
    let mut first_or = vec![];
    let mut or_two = vec![];
    let mut range_from = vec![];
    let mut bottom = vec![];

    for x in -9 + 1..=(9 - 2) {
        if let -1..=0 | 2..3 | 4 = x {
            if_lettable.push(x)
        }
        match x {
            1 | -3..0 => first_or.push(x),
            y @ (0..5 | 6) => or_two.push(y),
            y @ 0..const { 5 + 1 } => assert_eq!(y, 5),
            y @ -5.. => range_from.push(y),
            y @ ..-7 => assert_eq!(y, -8),
            y => bottom.push(y),
        }
    }
    assert_eq!(if_lettable, [-1, 0, 2, 4]);
    assert_eq!(first_or, [-3, -2, -1, 1]);
    assert_eq!(or_two, [0, 2, 3, 4, 6]);
    assert_eq!(range_from, [-5, -4, 7]);
    assert_eq!(bottom, [-7, -6]);
}
