// Regression test for the ICE described in #87495.

trait T {
    const CONST: (bool, dyn T);
    //~^ ERROR: the trait `T` is not dyn compatible [E0038]
}

fn main() {}
