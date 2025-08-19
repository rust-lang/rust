//@ revisions: current next
//@[next] compile-flags: -Znext-solver
//@ check-pass

#![feature(const_trait_impl)]

#[const_trait]
trait Trait {
    type Assoc: [const] Trait;
    fn func() -> i32;
}

const fn unqualified<T: [const] Trait>() -> i32 {
    T::Assoc::func()
}

const fn qualified<T: [const] Trait>() -> i32 {
    <T as Trait>::Assoc::func()
}

fn main() {}
