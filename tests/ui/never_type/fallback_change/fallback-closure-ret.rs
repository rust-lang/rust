// Tests the pattern of returning `!` from a closure and then checking if the
// return type iumplements a trait (not implemented for `!`).
//
// This test used to test that this pattern is not broken by context dependant
// never type fallback. However, it got removed, so now this is an example of
// expected breakage from the never type fallback change.
//
//@ revisions: e2021 e2024
//@[e2021] edition: 2021
//@[e2024] edition: 2024
//
//@[e2021] check-pass

trait Bar {}
impl Bar for () {}
impl Bar for u32 {}

fn foo<R: Bar>(_: impl Fn() -> R) {}

#[cfg_attr(e2021, expect(dependency_on_unit_never_type_fallback))]
fn main() {
    foo(|| panic!()); //[e2024]~ error: the trait bound `!: Bar` is not satisfied
}
