// run-pass

// Test that failing macro matchers will not cause pre-expansion errors
// even though they use a feature that is pre-expansion gated.

#[allow(unused_macro_rules)]
macro_rules! m {
    ($e:expr) => {
        0
    }; // This fails on the input below due to `, foo`.
    ($e:expr,) => {
        1
    }; // This also fails to match due to `foo`.
    (do yeet $e:expr, foo) => {
        2
    }; // Successful matcher, we should get `2`.
}

fn main() {
    assert_eq!(2, m!(do yeet 42, foo));
}
