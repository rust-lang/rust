//@ compile-flags: -Znext-solver
//@ check-pass

// Regression test for #158784
//
// We should assert that only opaque types that is to be replaced
// are non-rigid. We can have rigid opaque types elsewhere.

#![feature(type_alias_impl_trait)]
type Rigid = impl Sized;
#[define_opaque(Rigid)]
fn define_rigid() -> Rigid {}

type MyIter<T> = impl Iterator<Item = T>;

#[define_opaque(MyIter)]
fn define_my_iter<T>(a: T) -> MyIter<T> {
    if false {
        // `Rigid` being rigid is totally fine here.
        let _: MyIter<Rigid> = std::iter::once(define_rigid());
    }
    std::iter::once(a)
}

fn main() {}
