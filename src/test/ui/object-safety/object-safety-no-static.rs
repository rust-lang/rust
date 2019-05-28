// Check that we correctly prevent users from making trait objects
// from traits with static methods.

trait Foo {
    fn foo();
}

fn foo_implicit<T:Foo+'static>(b: Box<T>) -> Box<dyn Foo + 'static> {
    //~^ ERROR E0038
    loop { }
}

fn main() {
}
