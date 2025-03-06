// Regression test for the ICE described in #87495.

trait T {
    const CONST: (bool, dyn T);
    //~^ ERROR: the trait `T` is not dyn compatible [E0038]
    //~| ERROR: the size for values of type `(dyn T + 'static)` cannot be known at compilation time
}

fn main() {}
