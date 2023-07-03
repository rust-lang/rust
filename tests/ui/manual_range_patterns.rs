//@run-rustfix

#![allow(unused)]
#![warn(clippy::manual_range_patterns)]
#![feature(exclusive_range_pattern)]

fn main() {
    let f = 6;

    let _ = matches!(f, 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10);
    let _ = matches!(f, 4 | 2 | 3 | 1 | 5 | 6 | 9 | 7 | 8 | 10);
    let _ = matches!(f, 4 | 2 | 3 | 1 | 5 | 6 | 9 | 8 | 10); // 7 is missing
    let _ = matches!(f, | 4);
    let _ = matches!(f, 4 | 5);
    let _ = matches!(f, 1 | 2147483647);
    let _ = matches!(f, 0 | 2147483647);
    let _ = matches!(f, -2147483647 | 2147483647);
    let _ = matches!(f, 1 | (2..=4));
    let _ = matches!(f, 1 | (2..4));
    let _ = matches!(f, (1..=10) | (2..=13) | (14..=48324728) | 48324729);
    let _ = matches!(f, 0 | (1..=10) | 48324730 | (2..=13) | (14..=48324728) | 48324729);
    let _ = matches!(f, 0..=1 | 0..=2 | 0..=3);
    #[allow(clippy::match_like_matches_macro)]
    let _ = match f {
        1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 => true,
        _ => false,
    };
    let _ = matches!(f, -1 | -5 | 3 | -2 | -4 | -3 | 0 | 1 | 2);
    let _ = matches!(f, -1 | -5 | 3 | -2 | -4 | -3 | 0 | 1); // 2 is missing
    let _ = matches!(f, -1_000_000..=1_000_000 | -1_000_001 | 1_000_001);
    let _ = matches!(f, -1_000_000..=1_000_000 | -1_000_001 | 1_000_002);

    macro_rules! mac {
        ($e:expr) => {
            matches!($e, 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10)
        };
    }
    mac!(f);
}
