// Regression test related to issue 88434

const _CONST: &() = &f(&|_| {}); //~ ERROR evaluation of constant value failed
//~^ NOTE_NONVIRAL constant

const fn f<F>(_: &F)
where
    F: FnMut(&u8),
{
    panic!() //~ NOTE_NONVIRAL inside `f
}

fn main() { }
