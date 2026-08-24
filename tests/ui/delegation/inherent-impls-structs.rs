#![feature(fn_delegation)]

struct S<A, const C: usize> {
    xd: [A; C],
}

impl<'a, 'b, 'c, A, const C: usize> S<A, C> {
    fn foo_static<'d: 'd, 'e, T, const B: bool>() {}
    fn foo_self<'d: 'd, 'e, T, const B: bool>(self) {}
}

reuse S::<(), 1>::foo_static::<'static, (), true> as foo_static_1;
//~^ ERROR: cannot find function `foo_static` in `S`
reuse S::<(), 1>::foo_static as foo_static_3;
//~^ ERROR: cannot find function `foo_static` in `S`
reuse S::<usize, 1>::foo_static::<'static, _, _> as foo_static_4;
//~^ ERROR: cannot find function `foo_static` in `S`

reuse S::<(), 1>::foo_self::<'static, (), true> as foo_self_1;
//~^ ERROR: cannot find function `foo_self` in `S`
reuse S::<(), 1>::foo_self as foo_self_3;
//~^ ERROR: cannot find function `foo_self` in `S`
reuse S::<String, 1>::foo_self::<'static, _, _> as foo_self_4;
//~^ ERROR: cannot find function `foo_self` in `S`

trait Trait<'a, AA, BB> where Self: Sized {
    reuse S::<(), 1>::foo_static::<'static, (), true> as foo_static_1;
    //~^ ERROR: cannot find function `foo_static` in `S`
    reuse S::<(), 1>::foo_static as foo_static_3;
    //~^ ERROR: cannot find function `foo_static` in `S`
    reuse S::<(), 1>::foo_static::<'static, _, _> as foo_static_4;
    //~^ ERROR: cannot find function `foo_static` in `S`

    fn get_s(self) -> S<(), 1> {
        panic!();
    }

    reuse S::<(), 1>::foo_self::<'static, (), true> as foo_self_1 { self.get_s() }
    //~^ ERROR: cannot find function `foo_self` in `S`
    reuse S::<(), 1>::foo_self as foo_self_3 { self.get_s() }
    //~^ ERROR: cannot find function `foo_self` in `S`
    reuse S::<(), 1>::foo_self::<'static, _, _> as foo_self_4;
    //~^ ERROR: cannot find function `foo_self` in `S`
}

struct X;

impl<'a, A, B> Trait<'a, A, B> for X {
    reuse S::<(), 1>::foo_static::<'static, (), true> as foo_static_1;
    //~^ ERROR: cannot find function `foo_static` in `S`
    reuse S::<(), 1>::foo_static as foo_static_3;
    //~^ ERROR: cannot find function `foo_static` in `S`
    reuse S::<(), 1>::foo_static::<'static, _, _> as foo_static_4;
    //~^ ERROR: cannot find function `foo_static` in `S`

    reuse S::<(), 1>::foo_self::<'static, (), true> as foo_self_1 { self.get_s() }
    //~^ ERROR: cannot find function `foo_self` in `S`
    //~| ERROR: delegation's target expression is specified for function with no params
    reuse S::<(), 1>::foo_self as foo_self_3 { self.get_s() }
    //~^ ERROR: cannot find function `foo_self` in `S`
    //~| ERROR: delegation's target expression is specified for function with no params
    reuse S::<(), 1>::foo_self::<'static, _, _> as foo_self_4;
    //~^ ERROR: cannot find function `foo_self` in `S`
}

impl X {
    reuse S::<(), 1>::foo_static::<'static, (), true> as foo_static_1;
    //~^ ERROR: cannot find function `foo_static` in `S`
    reuse S::<(), 1>::foo_static as foo_static_3;
    //~^ ERROR: cannot find function `foo_static` in `S`
    reuse S::<(), 1>::foo_static::<'static, _, _> as foo_static_4;
    //~^ ERROR: cannot find function `foo_static` in `S`

    fn get_s(self) -> S<(), 1> {
        panic!();
    }

    reuse S::<(), 1>::foo_self::<'static, (), true> as foo_self_1 { self.get_s() }
    //~^ ERROR: cannot find function `foo_self` in `S`
    reuse S::<(), 1>::foo_self as foo_self_3 { self.get_s() }
    //~^ ERROR: cannot find function `foo_self` in `S`
    reuse S::<(), 1>::foo_self::<'static, _, _> as foo_self_4;
    //~^ ERROR: cannot find function `foo_self` in `S`
}

fn main() {}
