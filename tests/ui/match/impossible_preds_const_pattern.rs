#![feature(generic_const_items)]
#![expect(incomplete_features)]

trait Unimplemented<'a> {}

trait Trait<T>
where
    for<'a> T: Unimplemented<'a>,
{
    const ASSOC: usize;
}

impl<T> Trait<T> for ()
where
    for<'a> T: Unimplemented<'a>,
{
    const ASSOC: usize = 0;
}

trait Global {
    const ASSOC: usize
    where
        for<'a> (): Unimplemented<'a>;
}

impl Global for () {
    const ASSOC: usize = 0
    where
        for<'a> (): Unimplemented<'a>;
}

fn also_errors(x: usize)
where
    for<'a> (): Unimplemented<'a>,
{
    // In order for match exhaustiveness to determine this match is OK, CTFE must
    // be able to evalaute `<() as Trait>::ASSOC` which depends on a trivially
    // false bound for well formedness.
    //
    // This used to succeed but now errors
    match x {
        <() as Trait<()>>::ASSOC => todo!(),
        //~^ ERROR: constant pattern cannot depend on generic parameters
        1.. => todo!(),
    }
}

fn errors(x: usize)
where
    for<'a> (): Unimplemented<'a>,
{
    // This previously would error due to the MIR for `ASSOC` being empty
    // due to the globally false bound.
    match x {
        <() as Global>::ASSOC => todo!(),
        //~^ ERROR: constant pattern cannot depend on generic parameters
        1.. => todo!(),
    }
}

fn main() {}
