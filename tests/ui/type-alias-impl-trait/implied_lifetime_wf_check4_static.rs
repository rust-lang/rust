//@ revisions: old next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@[next] failure-status: 101
//@[next] dont-check-compiler-stderr
//@[next] known-bug: trait-system-refactor-initiative#293
//@[next] needs-rustc-debug-assertions
#![feature(type_alias_impl_trait)]

pub type Ty<A> = impl Sized + 'static;
#[define_opaque(Ty)]
fn defining<A: 'static>(s: A) -> Ty<A> {
    s
    //[old]~^ ERROR: the parameter type `A` may not live long enough
}
pub fn assert_static<A: 'static>() {}

fn test<A>()
where
    Ty<A>: 'static,
{
    assert_static::<A>()
    //[old]~^ ERROR: the parameter type `A` may not live long enough
}

fn main() {}
