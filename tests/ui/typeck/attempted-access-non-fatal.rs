// Check that bogus field access is non-fatal
fn main() {
    let x = 0;
    let _ = x.foo; //~ ERROR `{integer}` is a primitive type and therefore doesn't have fields [E0610]
    let _ = x.bar; //~ ERROR `{integer}` is a primitive type and therefore doesn't have fields [E0610]
    let _ = 0.f; //~ ERROR `{integer}` is a primitive type and therefore doesn't have fields [E0610]
    let _ = 2.l; //~ ERROR `{integer}` is a primitive type and therefore doesn't have fields [E0610]
    let _ = 12.F; //~ ERROR `{integer}` is a primitive type and therefore doesn't have fields [E0610]
    let _ = 34.L; //~ ERROR `{integer}` is a primitive type and therefore doesn't have fields [E0610]
}
