// Regression test for #81899.
// The `panic!()` below is important to trigger the fixed ICE.

//@ dont-require-annotations: NOTE

const _CONST: &[u8] = &f(&[], |_| {}); //~ ERROR explicit panic
//~^ NOTE constant

const fn f<F>(_: &[u8], _: F) -> &[u8]
where
    F: FnMut(&u8),
{
    panic!() //~ NOTE inside `f
}

fn main() {}
