// Regression test related to issue 88434

const _CONST: &() = &f(&|_| {}); //~ ERROR evaluation of constant value failed
//~^ constant

const fn f<F>(_: &F)
where
    F: FnMut(&u8),
{
    panic!() //~ inside `f
}

fn main() { }
