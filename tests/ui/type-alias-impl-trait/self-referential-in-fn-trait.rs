//! This test checks that we do not
//! end up with an infinite recursion,
//! a cycle error or an overflow when
//! encountering an opaque type that has
//! an associated type that is just itself
//! again.

#![feature(type_alias_impl_trait)]
// revisions: next old working
//[next] compile-flags: -Ztrait-solver=next
//[working] check-pass

type Foo<'a> = impl Fn() -> Foo<'a>;
//[old,next]~^ ERROR: unconstrained opaque type

fn crash<'a>(_: &'a (), x: Foo<'a>) -> Foo<'a> {
    x
}

#[cfg(working)]
fn foo<'a>() -> Foo<'a> {
    foo
}

fn main() {}
