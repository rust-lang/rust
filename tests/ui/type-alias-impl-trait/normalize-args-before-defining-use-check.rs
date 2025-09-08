#![feature(type_alias_impl_trait)]

//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@ [next] compile-flags: -Znext-solver
//@ check-pass

// Regression test for trait-system-refactor-initiative#49.

trait Mirror<'a> {
    type Assoc;
}
impl<'a, T> Mirror<'a> for T {
    type Assoc = T;
}

type HrAmbigAlias<T> = impl Sized;
fn ret_tait<T>() -> for<'a> fn(HrAmbigAlias<<T as Mirror<'a>>::Assoc>) {
    |_| ()
}

#[define_opaque(HrAmbigAlias)]
fn define_hr_ambig_alias<T>() {
    let _: fn(T) = ret_tait::<T>();
}

type InUserType<T> = impl Sized;
#[define_opaque(InUserType)]
fn in_user_type<T>() {
    let x: InUserType<<T as Mirror<'static>>::Assoc> = ();
}

fn main() {}
