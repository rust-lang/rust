#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

pub struct Foo<const N: usize, const M: usize = { N + 1 }>;
pub fn needs_evaluatable_bound<const N1: usize>() -> Foo<N1> {
    //~^ error: unconstrained generic constant
    loop {}
}
pub fn has_evaluatable_bound<const N1: usize>() -> Foo<N1> where [(); N1 + 1]: {
    loop {}
}

type FooAlias<const N: usize, const NP: usize = { N + 1 }> = [(); NP];
fn needs_evaluatable_bound_alias<T, const N: usize>() -> FooAlias<N>
{
    //~^^ error: unconstrained generic constant
    todo!()
}
fn has_evaluatable_bound_alias<const N: usize>() -> FooAlias<N>
where [(); N + 1]: {
    todo!()
}

fn main() {}
