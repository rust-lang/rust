//@ revisions: current next
//@[next] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)
//@ known-bug: #117606

#![feature(associated_type_defaults)]

trait Foo {
    type Bar<T>: Baz<Self> = i32;
    // We should be able to prove that `i32: Baz<Self>` because of
    // the impl below, which requires that `Self::Bar<()>: Eq<i32>`
    // which is true, because we assume `for<T> Self::Bar<T> = i32`.
}

trait Baz<T: ?Sized> {}
impl<T: Foo + ?Sized> Baz<T> for i32 where T::Bar<()>: Eq<i32> {}

trait Eq<T> {}
impl<T> Eq<T> for T {}

fn main() {}
