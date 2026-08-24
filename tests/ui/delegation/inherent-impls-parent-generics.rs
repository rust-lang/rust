#![feature(fn_delegation)]
#![allow(late_bound_lifetime_arguments)]

enum E<'a: 'a, A: 'a, const C: usize> {
    A(A),
    B(&'a [A; C]),
}

impl<'a, 'b, 'c, A: 'a, const C: usize> E<'a, A, C> {
    fn foo_static<'d: 'd, 'e, T, const B: bool>() {}
    fn foo_self<'d: 'd, 'e, T, const B: bool>(self) {}
}

reuse E::foo_static as e;
//~^ ERROR: cannot find function `foo_static` in enum `E`

reuse E::foo_self as e1;
//~^ ERROR: cannot find function `foo_self` in enum `E`

reuse E::foo_static::<'static, (), true> as e2;
//~^ ERROR: cannot find function `foo_static` in enum `E`

reuse E::foo_self::<'static, (), true> as e3;
//~^ ERROR: cannot find function `foo_self` in enum `E`

reuse E::<'static, (), 123>::foo_static as e4;
//~^ ERROR: cannot find function `foo_static` in enum `E`
reuse E::<'static, (), 123>::foo_self as e5;
//~^ ERROR: cannot find function `foo_self` in enum `E`

reuse E::<'static, (), 123>::foo_static::<'static, (), true> as e6;
//~^ ERROR: cannot find function `foo_static` in enum `E`
reuse E::<'static, (), 123>::foo_self::<'static, (), true> as e7;
//~^ ERROR: cannot find function `foo_self` in enum `E`

reuse E::<'_, (), _>::foo_static as e8;
//~^ ERROR: cannot find function `foo_static` in enum `E`

reuse E::<'_, _, _>::foo_self as e9;
//~^ ERROR: cannot find function `foo_self` in enum `E`

struct S<A, const C: usize> {
    xd: [A; C],
}

impl<'a, 'b, 'c, A, const C: usize> S<A, C> {
    fn foo_static<'d: 'd, 'e, T, const B: bool>() {}
    fn foo_self<'d: 'd, 'e, T, const B: bool>(self) {}
}

reuse S::foo_static as s;
//~^ ERROR: cannot find function `foo_static` in `S`

reuse S::foo_self as s1;
//~^ ERROR: cannot find function `foo_self` in `S`

reuse S::foo_static::<'static, (), true> as s2;
//~^ ERROR: cannot find function `foo_static` in `S`

reuse S::foo_self::<'static, (), true> as s3;
//~^ ERROR: cannot find function `foo_self` in `S`

reuse S::<(), 123>::foo_static as s4;
//~^ ERROR: cannot find function `foo_static` in `S`
reuse S::<(), 123>::foo_self as s5;
//~^ ERROR: cannot find function `foo_self` in `S`

reuse S::<(), 123>::foo_static::<'static, (), true> as s6;
//~^ ERROR: cannot find function `foo_static` in `S`
reuse S::<(), 123>::foo_self::<'static, (), true> as s7;
//~^ ERROR: cannot find function `foo_self` in `S`

reuse S::<(), _>::foo_static as s8;
//~^ ERROR: cannot find function `foo_static` in `S`

reuse S::<_, 123>::foo_self as s9;
//~^ ERROR: cannot find function `foo_self` in `S`


fn main() {}
