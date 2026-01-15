//! regression test for <https://github.com/rust-lang/rust/issues/151024>
#![feature(min_generic_const_args)]
#![feature(adt_const_params)]
#![expect(incomplete_features)]

trait Trait1<const N: usize> {}
trait Trait2<const N: [u8; 3]> {}

fn foo<T>()
where
    T: Trait1<{ [] }>, //~ ERROR: expected `usize`, found const array
{
}

fn bar<T>()
where
    T: Trait2<3>, //~ ERROR: mismatched types
{
}

fn main() {}
