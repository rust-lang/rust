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
//~^ ERROR: delegation to inherent impl must contain parent generics
//~| ERROR: type annotations needed
//~| ERROR: type annotations needed
//~| ERROR: type annotations needed
reuse E::foo_self as e1;
//~^ ERROR: delegation to inherent impl must contain parent generics
//~| ERROR: this function takes 1 argument but 0 arguments were supplied

reuse E::foo_static::<'static, (), true> as e2;
//~^ ERROR: delegation to inherent impl must contain parent generics
//~| ERROR: type annotations needed
//~| ERROR: type annotations needed
reuse E::foo_self::<'static, (), true> as e3;
//~^ ERROR: delegation to inherent impl must contain parent generics
//~| ERROR: this function takes 1 argument but 0 arguments were supplied

reuse E::<'static, (), 123>::foo_static as e4;
reuse E::<'static, (), 123>::foo_self as e5;

reuse E::<'static, (), 123>::foo_static::<'static, (), true> as e6;
reuse E::<'static, (), 123>::foo_self::<'static, (), true> as e7;

struct S<A, const C: usize> {
    xd: [A; C],
}

impl<'a, 'b, 'c, A, const C: usize> S<A, C> {
    fn foo_static<'d: 'd, 'e, T, const B: bool>() {}
    fn foo_self<'d: 'd, 'e, T, const B: bool>(self) {}
}

reuse S::foo_static as s;
//~^ ERROR: delegation to inherent impl must contain parent generics
//~| ERROR: type annotations needed
//~| ERROR: type annotations needed
//~| ERROR: type annotations needed
reuse S::foo_self as s1;
//~^ ERROR: delegation to inherent impl must contain parent generics
//~| ERROR: this function takes 1 argument but 0 arguments were supplied

reuse S::foo_static::<'static, (), true> as s2;
//~^ ERROR: delegation to inherent impl must contain parent generics
//~| ERROR: type annotations needed
//~| ERROR: type annotations needed
reuse S::foo_self::<'static, (), true> as s3;
//~^ ERROR: delegation to inherent impl must contain parent generics
//~| ERROR: this function takes 1 argument but 0 arguments were supplied

reuse S::<(), 123>::foo_static as s4;
reuse S::<(), 123>::foo_self as s5;

reuse S::<(), 123>::foo_static::<'static, (), true> as s6;
reuse S::<(), 123>::foo_self::<'static, (), true> as s7;

fn main() {}
