// This test case checks the behavior of typeck::check::method::suggest::is_fn on Ty::Error.

struct Foo;

trait Bar {
    //~^ NOTE `Bar` defines an item `bar`, perhaps you need to implement it
    fn bar(&self) {}
}

impl Bar for Foo {}

fn main() {
    let arc = std::sync::Arc::new(oops);
    //~^ ERROR cannot find value `oops` in this scope
    //~| NOTE not found
    arc.bar();

    let arc2 = std::sync::Arc::new(|| Foo);
    arc2.bar();
    //~^ ERROR no method named `bar`
    //~| NOTE method not found
    //~| HELP items from traits can only be used if the trait is implemented and in scope
    //~| HELP use parentheses to call this closure
}
