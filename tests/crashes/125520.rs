//@ known-bug: #125520
#![feature(generic_const_exprs)]

struct Outer<const A: i64, const B: i64>();
impl<const A: usize, const B: usize> Outer<A, B>
where
    [(); A + (B * 2)]:,
{
    fn i() -> Self {
        Self
    }
}

fn main() {
    Outer::<1, 1>::o();
}
