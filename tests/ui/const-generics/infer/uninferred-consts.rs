// Test that we emit an error if we cannot properly infer a constant.

// taken from https://github.com/rust-lang/rust/issues/70507#issuecomment-615268893
struct Foo;
impl Foo {
    fn foo<const A: usize, const B: usize>(self) {}
}
fn main() {
    Foo.foo();
    //~^ ERROR type annotations needed
    //~| ERROR type annotations needed
}
