//! regression test for <https://github.com/rust-lang/rust/issues/151024>
#![feature(adt_const_params, gca_adts, gca_min_const_items, gca_macroless_args)]

trait Trait1<const N: usize> {}
trait Trait2<const N: [u8; 3]> {}

fn foo<T>()
where
    T: Trait1<{ [] }>, //~ ERROR: expected `usize`, found const array
{
}

fn bar<T>()
where
    T: Trait2<3>, //~ ERROR: type annotations needed for the literal
{
}

fn main() {}
