#![feature(const_fn_trait_bound)]
// Regression test related to issue 88434

const _CONST: &() = &f(&|_| {});

const fn f<F>(_: &F)
where
    F: FnMut(&u8),
{
    panic!() //~ ERROR evaluation of constant value failed
}

fn main() { }
