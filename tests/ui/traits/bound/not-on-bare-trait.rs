trait Foo {
    fn dummy(&self) {}
}

// This should emit the less confusing error, not the more confusing one.

fn foo(_x: Foo + Send) {
    //~^ ERROR the size for values of type
    //~| WARN trait objects without an explicit `dyn` are deprecated
    //~| WARN this is accepted in the current edition
}
fn bar(_x: (dyn Foo + Send)) {
    //~^ ERROR the size for values of type
}

fn main() {}
