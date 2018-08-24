// Check that bogus field access is non-fatal
fn main() {
    let x = 0;
    let _ = x.foo; //~ `{integer}` is a primitive type and therefore doesn't have fields [E0610]
    let _ = x.bar; //~ `{integer}` is a primitive type and therefore doesn't have fields [E0610]
}
