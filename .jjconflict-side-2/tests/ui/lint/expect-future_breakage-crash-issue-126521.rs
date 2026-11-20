//@ check-pass

// This test covers similar crashes from both #126521 and #126751.

macro_rules! foo {
    ($val:ident) => {
        true;
    };
}

macro_rules! bar {
    ($val:ident) => {
        (5_i32.overflowing_sub(3));
    };
}

fn allow() {
    #[allow(semicolon_in_expressions_from_macros)]
    let _ = foo!(x);

    #[allow(semicolon_in_expressions_from_macros)]
    let _ = bar!(x);
}

// The `semicolon_in_expressions_from_macros` lint seems to be emitted even if the
// lint level is `allow` as shown in the function above. The behavior of `expect`
// should mirror this behavior. However, no `unfulfilled_lint_expectation` lint
// is emitted, since the expectation is theoretically fulfilled.
fn expect() {
    #[expect(semicolon_in_expressions_from_macros)]
    let _ = foo!(x);

    #[expect(semicolon_in_expressions_from_macros)]
    let _ = bar!(x);
}

fn main() {
}
