//@ check-pass

// In PR 71930, it was discovered that the code to retrieve the inferred type of a match scrutinee
// was incorrect.

fn f() -> ! {
    panic!()
}

fn g() -> usize {
    match f() { // Should infer type `bool`
        false => 0,
        true => 1,
    }
}

fn h() -> usize {
    match f() { // Should infer type `!`
    }
}

fn main() {}
