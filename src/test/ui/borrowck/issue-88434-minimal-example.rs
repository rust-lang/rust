// Regression test related to issue 88434

const _CONST: &() = &f(&|_| {});
//~^ ERROR any use of this value will cause an error
//~| WARNING this was previously accepted by the compiler but is being phased out

const fn f<F>(_: &F)
where
    F: FnMut(&u8),
{
    panic!() //~ ERROR evaluation of constant value failed
}

fn main() { }
