// This test used to test that this pattern is not broken by context dependant
// never type fallback. However, it got removed, so now this is an example of
// expected breakage from the never type fallback change.
//
//@ revisions: nofallback fallback
//@[nofallback] check-pass
//@[fallback] edition: 2024

#![cfg_attr(nofallback, expect(dependency_on_unit_never_type_fallback))]

trait Bar {}
impl Bar for () {}
impl Bar for u32 {}

fn foo<R: Bar>(_: impl Fn() -> R) {}

fn main() {
    foo(|| panic!()); //[fallback]~ error: the trait bound `!: Bar` is not satisfied
}
