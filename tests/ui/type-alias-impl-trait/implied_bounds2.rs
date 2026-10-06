//@ revisions: old next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@[next] failure-status: 101
//@[next] dont-check-compiler-stderr
//@[next] known-bug: trait-system-refactor-initiative#293
//@[old] check-pass

#![feature(type_alias_impl_trait)]

pub type Ty<'a, A> = impl Sized + 'a;
#[define_opaque(Ty)]
fn defining<'a, A>() -> Ty<'a, A> {}
pub fn assert_static<T: 'static>() {}
fn test<'a, A>()
where
    Ty<'a, A>: 'static,
{
    assert_static::<Ty<'a, A>>()
}

fn main() {}
