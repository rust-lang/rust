// Regression test related to issue 88434

const _CONST: &() = &f(&|_| {});
//~^ constant

const fn f<F>(_: &F)
where
    F: FnMut(&u8),
{
    panic!() //~ ERROR evaluation of `_CONST::{promoted#0}` failed
    //~^ panic
}

fn main() { }
