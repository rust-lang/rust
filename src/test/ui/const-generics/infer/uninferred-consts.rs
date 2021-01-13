// Test that we emit an error if we cannot properly infer a constant.
// revisions: full min

#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]

// taken from https://github.com/rust-lang/rust/issues/70507#issuecomment-615268893
struct Foo;
impl Foo {
    fn foo<const A: usize, const B: usize>(self) {}
}
fn main() {
    Foo.foo();
    //~^ ERROR type annotations needed
}
