// Test that the `Fn` traits require `()` form without a feature gate.

fn bar1(x: &dyn Fn<(), Output=()>) {
    //~^ ERROR of `Fn`-family traits' type parameters is subject to change
}

fn bar2<T>(x: &T) where T: Fn<()> {
    //~^ ERROR of `Fn`-family traits' type parameters is subject to change
}

fn main() { }
