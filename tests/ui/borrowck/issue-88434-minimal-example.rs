// Regression test related to issue 88434

//@ dont-require-annotations: NOTE

const _CONST: &() = &f(&|_| {}); //~ ERROR explicit panic
//~^ NOTE constant

const fn f<F>(_: &F)
where
    F: FnMut(&u8),
{
    panic!() //~ NOTE inside `f
}

fn main() {}
