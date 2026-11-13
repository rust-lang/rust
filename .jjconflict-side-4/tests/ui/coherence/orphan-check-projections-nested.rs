// This used to ICE in an earlier iteration of #117164. Minimized from crate `proqnt`.

//@ check-pass
//@ revisions: classic next
//@[next] compile-flags: -Znext-solver
//@ aux-crate:dep=trait-with-assoc-ty.rs
//@ edition: 2021

pub(crate) trait Trait<T> {
    type Assoc;
}

pub(crate) struct Type<T, U, V>(T, U, V);

impl<T, U> dep::Trait for Type<T, <<T as dep::Trait>::Assoc as Trait<U>>::Assoc, U>
where
    T: dep::Trait,
    <T as dep::Trait>::Assoc: Trait<U>,
{
    type Assoc = U;
}

fn main() {}
