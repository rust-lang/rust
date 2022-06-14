// Regression test for #81899.
// The `panic!()` below is important to trigger the fixed ICE.

const _CONST: &[u8] = &f(&[], |_| {});
//~^ ERROR any use of this value
//~| WARNING this was previously

const fn f<F>(_: &[u8], _: F) -> &[u8]
where
    F: FnMut(&u8),
{
    panic!() //~ ERROR: evaluation of constant value failed
}

fn main() {}
