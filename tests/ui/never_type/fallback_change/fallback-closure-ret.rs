// Tests the pattern of returning `!` from a closure and then checking if the
// return type iumplements a trait (not implemented for `!`).
//
// This test used to test that this pattern is not broken by context dependant
// never type fallback. However, it got removed, so now this is an example of
// expected breakage from the never type fallback change.
//
//@ edition: 2018..2024

trait Bar {}
impl Bar for () {}
impl Bar for u32 {}

fn foo<R: Bar>(_: impl Fn() -> R) {}

fn main() {
    foo(|| panic!()); //~ error: the trait bound `!: Bar` is not satisfied
}
