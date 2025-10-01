//@ compile-flags: -C opt-level=3
//@ check-pass

fn main() {
    let one = Thing;
    let two = Thing;

    debug_assert_eq!(one, two);
}

#[derive(Debug)]
#[cfg_attr(debug_assertions, derive(PartialEq))]
struct Thing;
