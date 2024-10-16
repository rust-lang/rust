//@ compile-flags: -Znext-solver

trait Mirror {
    type Assoc;
}
impl<T> Mirror for T {
    type Assoc = T;
}

fn arg() -> &'static [i32; 1] { todo!() }

fn arg_error(x: <fn() as Mirror>::Assoc, y: ()) { todo!() }

fn main() {
    // Should suggest to reverse the args...
    // but if we don't normalize the expected, then we don't.
    arg_error((), || ());
    //~^ ERROR arguments to this function are incorrect
}
