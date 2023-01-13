// Check that bogus field access is non-fatal
fn main() {
    let x = 0;
    let _ = x.foo; //~ `{integer}` is a primitive type and therefore doesn't have fields [E0610]
    let _ = x.bar; //~ `{integer}` is a primitive type and therefore doesn't have fields [E0610]
    let _ = 0.f; //~ `{integer}` is a primitive type and therefore doesn't have fields [E0610]
    let _ = 2.l; //~ `{integer}` is a primitive type and therefore doesn't have fields [E0610]
    let _ = 12.F; //~ `{integer}` is a primitive type and therefore doesn't have fields [E0610]
    let _ = 34.L; //~ `{integer}` is a primitive type and therefore doesn't have fields [E0610]
}
