// Regression test related to issue 88434

const _CONST: &() = &f(&|_| {});
//~^ ERROR any use of this value
//~| WARNING this was previously

const fn f<F>(_: &F)
where
    F: FnMut(&u8),
{
    panic!() //~ ERROR evaluation of constant value failed
}

fn main() { }
