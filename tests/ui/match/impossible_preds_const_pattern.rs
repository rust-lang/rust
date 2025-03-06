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
    //~^ ERROR: evaluation of constant value failed
    where
        for<'a> (): Unimplemented<'a>;
}

fn works(x: usize)
where
    for<'a> (): Unimplemented<'a>,
{
    // In order for match exhaustiveness to determine this match is OK, CTFE must
    // be able to evalaute `<() as Trait>::ASSOC` which depends on a trivially
    // false bound for well formedness.
    match x {
        <() as Trait<()>>::ASSOC => todo!(),
        1.. => todo!(),
    }
}

fn errors(x: usize)
where
    for<'a> (): Unimplemented<'a>,
{
    // This errors due to the MIR for `ASSOC` being empty due to the
    // globally false bound.
    match x {
        <() as Global>::ASSOC => todo!(),
        1.. => todo!(),
    }
}

fn main() {}
