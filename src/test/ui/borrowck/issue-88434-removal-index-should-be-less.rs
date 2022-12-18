// Regression test for issue 88434

const _CONST: &[u8] = &f(&[], |_| {});
//~^ constant

const fn f<F>(_: &[u8], _: F) -> &[u8]
where
    F: FnMut(&u8),
{
    panic!() //~ ERROR evaluation of constant value failed
    //~^ panic
}

fn main() { }
