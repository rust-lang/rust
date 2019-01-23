#![warn(clippy::single_match_else)]

fn main() {
    let n = match (42, 43) {
        (42, n) => n,
        _ => panic!("typeck error"),
    };
    assert_eq!(n, 43);
}
