//@ run-rustfix
fn e0658<F, G, H>(f: F, g: G, h: H) -> i32
where
    F: Fn<i32, Output = i32>, //~ ERROR E0658
    //~^ ERROR E0059
    G: Fn<(i32, i32, ), Output = (i32, i32)>, //~ ERROR E0658
    H: Fn<(i32,), Output = i32>, //~ ERROR E0658
{
    f(3);
    //~^ ERROR: cannot use call notation
    //~| ERROR: `i32` is not a tuple
    g(3, 4);
    h(3)
}

fn main() {
    e0658( //~ ERROR: mismatched types
        |a| a,
        |a, b| (b, a),
        |a| a,
    );
}
