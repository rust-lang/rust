#![feature(fn_delegation)]

struct S<'a, A, B, const C: usize> {
    xd: &'a [(A, B); C],
}

impl<'a, 'b, 'c, A, const C: usize> S<'static, A, usize, C> {
    fn foo_self<'d: 'd, 'e, T, const B: bool>(self) {}
}

trait Trait<'a, AA, BB> where Self: Sized {
    reuse S::foo_self;
    //~^ ERROR: cannot find function `foo_self` in `S`
}

fn main() {}
