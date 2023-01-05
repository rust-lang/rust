// This test verifies that never type fallback preserves the following code in a
// compiling state. This pattern is fairly common in the wild, notably seen in
// wasmtime v0.16. Typically this is some closure wrapper that expects a
// collection of 'known' signatures, and -> ! is not included in that set.
//
// This test is specifically targeted by the unit type fallback when
// encountering a set of obligations like `?T: Foo` and `Trait::Projection =
// ?T`. In the code below, these are `R: Bar` and `Fn::Output = R`.
//
// revisions: nofallback fallback
// check-pass

#![cfg_attr(fallback, feature(never_type_fallback))]

trait Bar { }
impl Bar for () {  }
impl Bar for u32 {  }

fn foo<R: Bar>(_: impl Fn() -> R) {}

fn main() {
    foo(|| panic!());
}
