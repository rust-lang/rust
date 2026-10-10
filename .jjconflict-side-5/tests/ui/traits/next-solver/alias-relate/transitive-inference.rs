//@ revisions: old next
//@[next] compile-flags: -Znext-solver=globally
//@ ignore-compare-mode-next-solver (explicit revisions)
//@ check-pass
// Regression test for trait-system-refactor-initiative#7. This test
// would error if we were to rely on lazy normalization here.
//
// We eagerly normalize the associated types, here, causing this to
// compile.

use std::marker::PhantomData;

#[derive(Default)]
struct Foo<T, U>(PhantomData<(T, U)>);

trait Trait {
    type Assoc;

    fn to_assoc(self) -> Self::Assoc;
}

impl Trait for Foo<u32, i32> {
    type Assoc = Foo<u32, i32>;
    fn to_assoc(self) -> Self::Assoc {
        Foo(PhantomData)
    }
}
impl Trait for Foo<i32, u32> {
    type Assoc = Foo<i32, u32>;
    fn to_assoc(self) -> Self::Assoc {
        Foo(PhantomData)
    }
}

#[allow(unused_assignments)]
fn main() {
    let mut x: Foo<_, _> = Default::default();
    let mut assoc = x.to_assoc();
    assoc = Foo::<u32, _>(PhantomData);
    assoc = Foo::<_, i32>(PhantomData);
    x = assoc;
}
