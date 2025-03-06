#![feature(generic_const_items, min_generic_const_args)]
#![expect(incomplete_features)]

// Tests behaviour of attempting to evaluate const arguments whose wellformedness
// depends on impossible to satisfy predicates, that are satisfied from another
// trivially false clause in the environment.
//
// Additionally tests how (or if) this behaviour differs depending on whether the
// constants where trivially false where clause is global or not.

trait Unimplemented<'a> {}

trait Trait {
    const NON_GLOBAL<T>: usize
    where
        for<'a> T: Unimplemented<'a>;

    const GLOBAL: usize
    where
        for<'a> (): Unimplemented<'a>;
}

impl Trait for u8 {
    const NON_GLOBAL<T>: usize = 1
    where
        for<'a> T: Unimplemented<'a>;

    const GLOBAL: usize = 1
    //~^ ERROR: evaluation of constant value failed
    where
        for<'a> (): Unimplemented<'a>;
}

struct Foo<const N: usize>;

fn non_global()
where
    for<'a> (): Unimplemented<'a>,
{
    let _: Foo<1> = Foo::<{ <u8 as Trait>::NON_GLOBAL::<()> }>;
}

fn global()
where
    for<'a> (): Unimplemented<'a>,
{
    let _: Foo<1> = Foo::<{ <u8 as Trait>::GLOBAL }>;
}

fn main() {}
