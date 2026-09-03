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
//~^ ERROR: inferred lifetimes are not allowed in delegations as we need to inherit signature
//~| ERROR: the placeholder `_` is not allowed within types on item signatures for functions
//~| ERROR: the placeholder `_` is not allowed within types on item signatures for functions
reuse E::foo_self as e1;
//~^ ERROR: inferred lifetimes are not allowed in delegations as we need to inherit signature
//~| ERROR: the placeholder `_` is not allowed within types on item signatures for functions
//~| ERROR: the placeholder `_` is not allowed within types on item signatures for functions
//~| ERROR: this function takes 1 argument but 0 arguments were supplied

reuse E::foo_static::<'static, (), true> as e2;
//~^ ERROR: inferred lifetimes are not allowed in delegations as we need to inherit signature
//~| ERROR: the placeholder `_` is not allowed within types on item signatures for functions
//~| ERROR: the placeholder `_` is not allowed within types on item signatures for functions
reuse E::foo_self::<'static, (), true> as e3;
//~^ ERROR: inferred lifetimes are not allowed in delegations as we need to inherit signature
//~| ERROR: the placeholder `_` is not allowed within types on item signatures for functions
//~| ERROR: the placeholder `_` is not allowed within types on item signatures for functions
//~| ERROR: this function takes 1 argument but 0 arguments were supplied

reuse E::<'static, (), 123>::foo_static as e4;
reuse E::<'static, (), 123>::foo_self as e5;

reuse E::<'static, (), 123>::foo_static::<'static, (), true> as e6;
reuse E::<'static, (), 123>::foo_self::<'static, (), true> as e7;

reuse E::<'_, (), _>::foo_static as e8;
//~^ ERROR: the placeholder `_` is not allowed within types on item signatures for functions
//~| ERROR: inferred lifetimes are not allowed in delegations as we need to inherit signature

reuse E::<'_, _, _>::foo_self as e9;
//~^ ERROR: the placeholder `_` is not allowed within types on item signatures for functions
//~| ERROR: the placeholder `_` is not allowed within types on item signatures for functions
//~| ERROR: inferred lifetimes are not allowed in delegations as we need to inherit signature
//~| ERROR: this function takes 1 argument but 0 arguments were supplied

struct S<A, const C: usize> {
    xd: [A; C],
}

impl<'a, 'b, 'c, A, const C: usize> S<A, C> {
    fn foo_static<'d: 'd, 'e, T, const B: bool>() {}
    fn foo_self<'d: 'd, 'e, T, const B: bool>(self) {}
}

reuse S::foo_static as s;
//~^ ERROR: the placeholder `_` is not allowed within types on item signatures for functions
//~| ERROR: the placeholder `_` is not allowed within types on item signatures for functions

reuse S::foo_self as s1;
//~^ ERROR: the placeholder `_` is not allowed within types on item signatures for functions
//~| ERROR: the placeholder `_` is not allowed within types on item signatures for functions
//~| ERROR: this function takes 1 argument but 0 arguments were supplied

reuse S::foo_static::<'static, (), true> as s2;
//~^ ERROR: the placeholder `_` is not allowed within types on item signatures for functions
//~| ERROR: the placeholder `_` is not allowed within types on item signatures for functions

reuse S::foo_self::<'static, (), true> as s3;
//~^ ERROR: the placeholder `_` is not allowed within types on item signatures for functions
//~| ERROR: the placeholder `_` is not allowed within types on item signatures for functions
//~| ERROR: this function takes 1 argument but 0 arguments were supplied

reuse S::<(), 123>::foo_static as s4;
reuse S::<(), 123>::foo_self as s5;

reuse S::<(), 123>::foo_static::<'static, (), true> as s6;
reuse S::<(), 123>::foo_self::<'static, (), true> as s7;

reuse S::<(), _>::foo_static as s8;
//~^ ERROR: the placeholder `_` is not allowed within types on item signatures for functions

reuse S::<_, 123>::foo_self as s9;
//~^ ERROR: the placeholder `_` is not allowed within types on item signatures for functions
//~| ERROR: this function takes 1 argument but 0 arguments were supplied

fn main() {}
