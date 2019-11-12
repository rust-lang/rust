// run-pass

// Test that failing macro matchers will not cause pre-expansion errors
// even though they use a feature that is pre-expansion gated.

macro_rules! m {
    ($e:expr) => { 0 }; // This fails on the input below due to `, foo`.
    ($e:expr,) => { 1 }; // This also fails to match due to `foo`.
    (box $e:expr, foo) => { 2 }; // Successful matcher, we should get `2`.
}

fn main() {
    assert_eq!(2, m!(box 42, foo));
}
