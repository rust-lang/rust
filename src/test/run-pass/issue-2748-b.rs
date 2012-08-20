fn thing(x: &r/[int]) -> &r/[int] { x }
fn main() {
    let x = &[1,2,3];
    let y = x;
    let z = thing(x);
    assert(z[2] == x[2]);
    assert(z[1] == y[1]);
}
