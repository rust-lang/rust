// run-pass
// Issue #33540
// We previously used to generate a 3-armed boolean `SwitchInt` in the
// MIR of the function `foo` below. #33583 changed rustc to
// generate an `If` terminator instead. This test is to just ensure
// sanity in that we generate an if-else chain giving the correct
// results.

fn foo(x: bool, y: bool) -> u32 {
    match (x, y) {
        (false, _) => 0,
        (_, false) => 1,
        (true, true) => 2
    }
}

fn main() {
    assert_eq!(foo(false, true), 0);
    assert_eq!(foo(false, false), 0);
    assert_eq!(foo(true, false), 1);
    assert_eq!(foo(true, true), 2);
}
