// Regression test for issue 88434

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
