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

// regression test for https://github.com/rust-lang/rust/issues/149487.
fn quux() {
    let mut tx_heights: std::collections::BTreeMap<(), Option<()>> = <_>::default();
    tx_heights.get(&()).unwrap_or_default();
    //~^ ERROR the trait bound `&Option<()>: Default` is not satisfied
    //~| HELP: the trait `Default` is implemented for `Option<T>`
}
