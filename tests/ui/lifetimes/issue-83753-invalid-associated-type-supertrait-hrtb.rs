//@ check-fail

struct Foo {}
impl Foo {
    fn bar(foo: Foo<Target = usize>) {}
    //~^ associated type bindings are not allowed here
}
fn main() {}
