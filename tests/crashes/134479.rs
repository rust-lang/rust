//@ known-bug: #134479
//@ compile-flags: -Csymbol-mangling-version=v0 -Cdebuginfo=1

#![feature(generic_const_exprs)]

fn main() {
    test::<2>();
}

struct Test<const N: usize>;

fn new<const N: usize>() -> Test<N>
where
    [(); N * 1]: Sized,
{
    Test
}

fn test<const N: usize>() -> Test<{ N - 1 }>
where
    [(); (N - 1) * 1]: Sized,
{
    new()
}
