struct Foo;
struct Bar;
impl From<Bar> for Foo {
    fn from(_: Bar) -> Self { Foo }
}
fn qux(_: impl From<Bar>) {}
fn main() {
    qux(Bar.into()); //~ ERROR type annotations needed
    //~| HELP try using a fully qualified path to specify the expected types
    //~| HELP consider removing this method call, as the receiver has type `Bar` and `Bar: From<Bar>` trivially holds
}
