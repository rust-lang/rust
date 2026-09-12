#![feature(fn_delegation)]

enum S<'a: 'a, A: 'a, const C: usize> {
    A(A),
    B(&'a [A; C]),
}

impl<'a, 'b, 'c, A: 'a, const C: usize> S<'a, A, C> {
    fn foo_static<'d: 'd, 'e, T, const B: bool>() {}
    fn foo_self<'d: 'd, 'e, T, const B: bool>(self) {}
}

reuse S::<'static, (), 1>::foo_static::<'static, (), true> as foo_static_1;
reuse S::<'static, (), 1>::foo_static as foo_static_3;
reuse S::<'static, (), 1>::foo_static::<'static, _, _> as foo_static_4;

reuse S::<'static, (), 1>::foo_self::<'static, (), true> as foo_self_1;
reuse S::<'static, (), 1>::foo_self as foo_self_3;
reuse S::<'static, (), 1>::foo_self::<'static, _, _> as foo_self_4;

trait Trait<'a, AA, BB>
where
    Self: Sized,
{
    reuse S::<'static, (), 1>::foo_static::<'static, (), true> as foo_static_1;
    reuse S::<'static, (), 1>::foo_static as foo_static_3;
    reuse S::<'static, (), 1>::foo_static::<'static, _, _> as foo_static_4;

    fn get_s(self) -> S<'static, (), 1> {
        panic!();
    }

    reuse S::<'static, (), 1>::foo_self::<'static, (), true> as foo_self_1 { self.get_s() }
    reuse S::<'static, (), 1>::foo_self as foo_self_3 { self.get_s() }
    reuse S::<'static, (), 1>::foo_self::<'static, _, _> as foo_self_4;
    //~^ ERROR: mismatched types [E0308]
}

struct X;

impl<'a, A, B> Trait<'a, A, B> for X {
    reuse S::<'static, (), 1>::foo_static::<'static, (), true> as foo_static_1;
    reuse S::<'static, (), 1>::foo_static as foo_static_3;
    reuse S::<'static, (), 1>::foo_static::<'static, _, _> as foo_static_4;
    //~^ ERROR: type annotations needed [E0284]

    reuse S::<'static, (), 1>::foo_self::<'static, (), true> as foo_self_1 { self.get_s() }
    reuse S::<'static, (), 1>::foo_self as foo_self_3 { self.get_s() }
    reuse S::<'static, (), 1>::foo_self::<'static, _, _> as foo_self_4;
    //~^ ERROR: mismatched types [E0308]
}

impl X {
    reuse S::<'static, (), 1>::foo_static::<'static, (), true> as foo_static_1;
    reuse S::<'static, (), 1>::foo_static as foo_static_3;
    reuse S::<'static, (), 1>::foo_static::<'static, _, _> as foo_static_4;

    fn get_s(self) -> S<'static, (), 1> {
        panic!();
    }

    reuse S::<'static, (), 1>::foo_self::<'static, (), true> as foo_self_1 { self.get_s() }
    reuse S::<'static, (), 1>::foo_self as foo_self_3 { self.get_s() }
    reuse S::<'static, (), 1>::foo_self::<'static, _, _> as foo_self_4;
    //~^ ERROR: mismatched types [E0308]
}

fn main() {}
