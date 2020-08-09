// Test if emits error if cannot properly infer constant.
// revisions: full min

#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]
#![cfg_attr(min, feature(min_const_generics))]

// taken from https://github.com/rust-lang/rust/issues/70507#issuecomment-615268893
struct Foo;
impl Foo {
    fn foo<const N: usize>(self) {}
}
fn main() {
    Foo.foo();
    //~^ ERROR type annotations needed
}
