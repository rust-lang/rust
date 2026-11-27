// Ensure if let guards can be used in constant expressions.

//@ build-pass
//@revisions: edition2021 edition2024
//@[edition2021] edition:2021
//@[edition2024] edition:2024

const fn match_if_let(x: Option<i32>, y: Option<i32>) -> i32 {
    match x {
        None if let Some(a @ 5) = y => a,
        Some(z) if let (Some(_), 12) = (y, z) => 2,
        _ => 3,
    }
}

const ASSERTS: usize = {
    assert!(match_if_let(None, Some(5)) == 5);
    assert!(match_if_let(Some(12), Some(3)) == 2);
    assert!(match_if_let(None, Some(4)) == 3);
    assert!(match_if_let(Some(11), Some(3)) == 3);
    assert!(match_if_let(Some(12), None) == 3);
    assert!(match_if_let(None, None) == 3);
    0
};

fn main() {
    let _: [(); ASSERTS];
}
