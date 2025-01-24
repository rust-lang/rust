#![allow(unused)]
#![allow(non_contiguous_range_endpoints)]
#![warn(clippy::manual_range_patterns)]

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

    matches!(f, 0x00 | 0x01 | 0x02 | 0x03);
    matches!(f, 0x00..=0x05 | 0x06 | 0x07);
    matches!(f, -0x09 | -0x08 | -0x07..=0x00);

    matches!(f, 0..5 | 5);
    matches!(f, 0 | 1..5);

    matches!(f, 0..=5 | 6..10);
    matches!(f, 0..5 | 5..=10);
    matches!(f, 5..=10 | 0..5);

    macro_rules! mac {
        ($e:expr) => {
            matches!($e, 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10)
        };
    }
    mac!(f);

    #[rustfmt::skip]
    let _ = match f {
        | 2..=15 => 4,
        | 241..=254 => 5,
        | 255 => 6,
        | _ => 7,
    };
}
