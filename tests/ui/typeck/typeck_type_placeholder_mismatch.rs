// This test checks that genuine type errors with partial
// type hints are understandable.

use std::marker::PhantomData;

struct Foo<T>(PhantomData<T>);
struct Bar<U>(PhantomData<U>);

pub fn main() {
}

fn test1() {
    let x: Foo<_> = Bar::<usize>(PhantomData);
    //~^ ERROR mismatched types
    //~| expected struct `Foo<_>`
    //~| found struct `Bar<usize>`
    //~| expected `Foo<_>`, found `Bar<usize>`
    let y: Foo<usize> = x;
}

fn test2() {
    let x: Foo<_> = Bar::<usize>(PhantomData);
    //~^ ERROR mismatched types
    //~| expected struct `Foo<_>`
    //~| found struct `Bar<usize>`
    //~| expected `Foo<_>`, found `Bar<usize>`
}
