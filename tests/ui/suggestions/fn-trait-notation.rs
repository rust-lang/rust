// run-rustfix
fn e0658<F, G, H>(f: F, g: G, h: H) -> i32
where
    F: Fn<i32, Output = i32>, //~ ERROR E0658
    G: Fn<(i32, i32, ), Output = (i32, i32)>, //~ ERROR E0658
    H: Fn<(i32,), Output = i32>, //~ ERROR E0658
{
    f(3);
    g(3, 4);
    h(3)
}

fn main() {
    e0658(
        |a| a,
        |a, b| (b, a),
        |a| a,
    );
}
