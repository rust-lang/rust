trait Trait {
    fn foo(&self) -> Self;
}

fn call_foo(x: Box<dyn Trait>) {
    //~^ ERROR E0038
    let y = x.foo();
    //~^ ERROR E0038
    //~| ERROR E0277
}

fn main() {
}
