// Regression test for issue #130142

// Checks that we emit no warnings when a lint's level
// is overridden by an expect or allow attr on a Stmt node

//@ check-pass

#[must_use]
pub fn must_use_result() -> i32 {
    42
}

fn main() {
    #[expect(unused_must_use)]
    must_use_result();

    #[allow(unused_must_use)]
    must_use_result();
}
