//@ check-fail

struct Foo {}
impl Foo {
    fn bar(foo: Foo<Target = usize>) {}
    //~^ associated item constraints are not allowed here
}
fn main() {}
