//! Check that braces has the expected precedence in relation to index op and some arithmetic
//! bin-ops involving nested braces.
//!
//! This is a regression test for [Wrapping expr in curly braces changes the operator precedence
//! #28777](https://github.com/rust-lang/rust/issues/28777), which was fixed by
//! <https://github.com/rust-lang/rust/pull/30375>.

//@ run-pass

fn that_odd_parse(c: bool, n: usize) -> u32 {
    let x = 2;
    let a = [1, 2, 3, 4];
    let b = [5, 6, 7, 7];
    x + if c { a } else { b }[n]
}

/// See [Wrapping expr in curly braces changes the operator precedence
/// #28777](https://github.com/rust-lang/rust/issues/28777). This was fixed by
/// <https://github.com/rust-lang/rust/pull/30375>. #30375 added the `that_odd_parse` example above,
/// but that is not *quite* the same original example as reported in #28777, so we also include the
/// original example here.
fn check_issue_28777() {
    // Before #30375 fixed the precedence...

    // ... `v1` evaluated to 9, indicating a parse of `(1 + 2) * 3`, while
    let v1 = { 1 + { 2 } * { 3 } };

    // `v2` evaluated to 7, indicating a parse of `1 + (2 * 3)`.
    let v2 = 1 + { 2 } * { 3 };

    // Check that both now evaluate to 7, as was fixed by #30375.
    assert_eq!(v1, 7);
    assert_eq!(v2, 7);
}

fn main() {
    assert_eq!(4, that_odd_parse(true, 1));
    assert_eq!(8, that_odd_parse(false, 1));

    check_issue_28777();
}
