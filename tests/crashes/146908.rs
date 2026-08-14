//@ known-bug: rust-lang/rust#146908
#![feature(generic_const_exprs)]

fn can_double<const N: usize>(x: [(); N])
where
    [(); N * 2]:,
{
    x[0];
}

fn foo<const N: usize>()
where
    [(); (N + 1) * 2]:,
{
    can_double([(); { N + 1 }]);
}

fn main() {
    foo::<1>();
}
