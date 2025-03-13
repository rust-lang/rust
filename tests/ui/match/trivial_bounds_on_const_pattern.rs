// Tests how we handle const patterns with trivial bounds. Previously this
// would ICE as CTFE could not handle evaluating bodies with trivial bounds.

trait Trait
where
    for<'a> [u8]: Sized,
{
    const ASSOC: [u8];
}

impl Trait for u8
where
    for<'a> [u8]: Sized,
{
    const ASSOC: [u8] = loop {};
}

fn foo()
where
    for<'a> [u8]: Sized,
{
    match &[10_u8; 2] as &[u8] {
        &<u8 as Trait>::ASSOC => todo!(),
        //~^ ERROR: constant pattern cannot depend on generic parameters
        _ => todo!(),
    }
}

fn main() {}
