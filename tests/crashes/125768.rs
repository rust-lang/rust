//@ known-bug: rust-lang/rust#125768

#![feature(generic_const_exprs)]

struct Outer<const A: i64, const B: usize>();
impl<const A: usize, const B: usize> Outer<A, B>
where
    [(); A + (B * 2)]:,
{
    fn o() -> Union {}
}

fn main() {
    Outer::<1, 1>::o();
}
