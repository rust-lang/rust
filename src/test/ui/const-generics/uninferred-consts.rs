#![feature(const_generics)]
//~^ WARN the feature `const_generics` is incomplete

// taken from https://github.com/rust-lang/rust/issues/70507#issuecomment-615268893
struct Foo;
impl Foo {
    fn foo<const N: usize>(self) {}
}
fn main() {
    Foo.foo();
    //~^ ERROR type annotations needed
}
