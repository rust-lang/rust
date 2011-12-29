// Allow block arguments with ternary... why not, no chance of ambig.
fn main() {
    let v = [-1f, 1f];
    let foo = vec::any(v) { |e| float::negative(e) } ? true : false;
    assert foo;
}
