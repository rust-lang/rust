#![feature(fn_delegation)]

enum S<'a: 'a, A: 'a, const C: usize> {
    A(A),
    B(&'a [A; C]),
}

impl<'a, 'b, 'c, A: 'a, const C: usize> S<'a, A, C> {
    fn foo_static<'d: 'd, 'e, T, const B: bool>() {}
    fn foo_self<'d: 'd, 'e, T, const B: bool>(self) {}
}

reuse S::foo_static;
reuse S::<'static, (), 1>::foo_static::<'static, (), true> as foo_static_1;
reuse S::foo_static::<'static, (), true> as foo_static_2;
reuse S::<'static, (), 1>::foo_static as foo_static_3;
reuse S::<'_, _, 1>::foo_static::<'static, _, _> as foo_static_4;

reuse S::foo_self;
reuse S::<'static, (), 1>::foo_self::<'static, (), true> as foo_self_1;
reuse S::foo_self::<'static, (), true> as foo_self_2;
reuse S::<'static, (), 1>::foo_self as foo_self_3;
reuse S::<'_, _, 1>::foo_self::<'static, _, _> as foo_self_4;

trait Trait<'a, AA, BB>
where
    Self: Sized,
{
    reuse S::foo_static;
    reuse S::<'static, (), 1>::foo_static::<'static, (), true> as foo_static_1;
    reuse S::foo_static::<'static, (), true> as foo_static_2;
    reuse S::<'static, (), 1>::foo_static as foo_static_3;
    reuse S::<'_, _, 1>::foo_static::<'static, _, _> as foo_static_4;

    fn get_s(self) -> S<'static, (), 1> {
        panic!();
    }

    reuse S::foo_self;
    //~^ ERROR: mismatched types [E0308]
    reuse S::<'static, (), 1>::foo_self::<'static, (), true> as foo_self_1 { self.get_s() }
    reuse S::foo_self::<'static, (), true> as foo_self_2;
    //~^ ERROR: mismatched types [E0308]
    reuse S::<'static, (), 1>::foo_self as foo_self_3 { self.get_s() }
    reuse S::<'_, _, 1>::foo_self::<'static, _, _> as foo_self_4;
    //~^ ERROR: mismatched types [E0308]
}

struct X;

impl<'a, A, B> Trait<'a, A, B> for X {
    reuse S::foo_static;
    //~^ ERROR: associated function takes at most 2 generic arguments but 4 generic arguments were supplied [E0107]
    //~| ERROR: associated function takes 1 lifetime argument but 2 lifetime arguments were supplied
    reuse S::<'static, (), 1>::foo_static::<'static, (), true> as foo_static_1;
    reuse S::foo_static::<'static, (), true> as foo_static_2;
    //~^ ERROR: type annotations needed [E0284]
    reuse S::<'static, (), 1>::foo_static as foo_static_3;
    reuse S::<'_, _, 1>::foo_static::<'static, _, _> as foo_static_4;
    //~^ ERROR: type annotations needed [E0284]

    reuse S::foo_self;
    //~^ ERROR: mismatched types [E0308]
    //~| ERROR: method takes at most 2 generic arguments but 4 generic arguments were supplied
    //~| ERROR: method takes 1 lifetime argument but 2 lifetime arguments were supplied
    reuse S::<'static, (), 1>::foo_self::<'static, (), true> as foo_self_1 { self.get_s() }
    reuse S::foo_self::<'static, (), true> as foo_self_2;
    //~^ ERROR: mismatched types [E0308]
    reuse S::<'static, (), 1>::foo_self as foo_self_3 { self.get_s() }
    reuse S::<'_, _, 1>::foo_self::<'static, _, _> as foo_self_4;
    //~^ ERROR: mismatched types [E0308]
}

impl X {
    reuse S::foo_static;
    reuse S::<'static, (), 1>::foo_static::<'static, (), true> as foo_static_1;
    reuse S::foo_static::<'static, (), true> as foo_static_2;
    reuse S::<'static, (), 1>::foo_static as foo_static_3;
    reuse S::<'_, _, 1>::foo_static::<'static, _, _> as foo_static_4;

    fn get_s(self) -> S<'static, (), 1> {
        panic!();
    }

    reuse S::foo_self;
    //~^ ERROR: mismatched types [E0308]
    reuse S::<'static, (), 1>::foo_self::<'static, (), true> as foo_self_1 { self.get_s() }
    reuse S::foo_self::<'static, (), true> as foo_self_2;
    //~^ ERROR: mismatched types [E0308]
    reuse S::<'static, (), 1>::foo_self as foo_self_3 { self.get_s() }
    reuse S::<'_, _, 1>::foo_self::<'static, _, _> as foo_self_4;
    //~^ ERROR: mismatched types [E0308]
}

fn main() {}
