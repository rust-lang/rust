#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

struct Outer<const A: i64, const B: usize>();
impl<const A: usize, const B: usize> Outer<A, B>
//~^ ERROR: `A` is not of type `i64`
//~| ERROR: mismatched types
where
    [(); A + (B * 2)]:,
{
    fn o() {}
}

fn main() {
    Outer::<1, 1>::o();
    //~^ ERROR: no function or associated item named `o` found
}
