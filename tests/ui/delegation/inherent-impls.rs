#![feature(fn_delegation)]

struct S<A, const C: usize> {
    xd: [A; C]
}

impl<'a, 'b, 'c, A, const C: usize> S<A, C> {
    fn foo<'d, T, const B: bool>() {}
}

reuse S::foo;
reuse S::<(), 1>::foo::<'static, (), true> as bar;

trait Trait {
    reuse S::foo;
    reuse S::<(), 1>::foo::<'static, (), true> as bar;
}

fn main() {}
