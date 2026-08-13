#![feature(fn_delegation)]

struct S<A, const C: usize> {
    xd: [A; C],
}

impl<'a, 'b, 'c, A, const C: usize> S<A, C> {
    fn foo_static<'d: 'd, 'e, T, const B: bool>() {}
    fn foo_self<'d: 'd, 'e, T, const B: bool>(self) {}
}

reuse S::<(), 1>::foo_static::<'static, (), true> as foo_static_1;
reuse S::<(), 1>::foo_static as foo_static_3;
reuse S::<_, 1>::foo_static::<'static, _, _> as foo_static_4;

reuse S::<(), 1>::foo_self::<'static, (), true> as foo_self_1;
reuse S::<(), 1>::foo_self as foo_self_3;
reuse S::<_, 1>::foo_self::<'static, _, _> as foo_self_4;

trait Trait<'a, AA, BB> where Self: Sized {
    reuse S::<(), 1>::foo_static::<'static, (), true> as foo_static_1;
    reuse S::<(), 1>::foo_static as foo_static_3;
    reuse S::<_, 1>::foo_static::<'static, _, _> as foo_static_4;

    fn get_s(self) -> S<(), 1> {
        panic!();
    }

    reuse S::<(), 1>::foo_self::<'static, (), true> as foo_self_1 { self.get_s() }
    reuse S::<(), 1>::foo_self as foo_self_3 { self.get_s() }
    reuse S::<_, 1>::foo_self::<'static, _, _> as foo_self_4;
    //~^ ERROR: mismatched types [E0308]
}

struct X;

impl<'a, A, B> Trait<'a, A, B> for X {
    reuse S::<(), 1>::foo_static::<'static, (), true> as foo_static_1;
    reuse S::<(), 1>::foo_static as foo_static_3;
    reuse S::<_, 1>::foo_static::<'static, _, _> as foo_static_4;
    //~^ ERROR: type annotations needed [E0284]

    reuse S::<(), 1>::foo_self::<'static, (), true> as foo_self_1 { self.get_s() }
    reuse S::<(), 1>::foo_self as foo_self_3 { self.get_s() }
    reuse S::<_, 1>::foo_self::<'static, _, _> as foo_self_4;
    //~^ ERROR: mismatched types [E0308]
}

impl X {
    reuse S::<(), 1>::foo_static::<'static, (), true> as foo_static_1;
    reuse S::<(), 1>::foo_static as foo_static_3;
    reuse S::<_, 1>::foo_static::<'static, _, _> as foo_static_4;

    fn get_s(self) -> S<(), 1> {
        panic!();
    }

    reuse S::<(), 1>::foo_self::<'static, (), true> as foo_self_1 { self.get_s() }
    reuse S::<(), 1>::foo_self as foo_self_3 { self.get_s() }
    reuse S::<_, 1>::foo_self::<'static, _, _> as foo_self_4;
    //~^ ERROR: mismatched types [E0308]
}

fn main() {}
