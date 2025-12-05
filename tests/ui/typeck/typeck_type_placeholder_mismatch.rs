// This test checks that genuine type errors with partial
// type hints are understandable.

//@ dont-require-annotations: NOTE

use std::marker::PhantomData;

struct Foo<T>(PhantomData<T>);
struct Bar<U>(PhantomData<U>);

pub fn main() {
}

fn test1() {
    let x: Foo<_> = Bar::<usize>(PhantomData);
    //~^ ERROR mismatched types
    //~| NOTE expected struct `Foo<_>`
    //~| NOTE found struct `Bar<usize>`
    //~| NOTE expected `Foo<_>`, found `Bar<usize>`
    let y: Foo<usize> = x;
}

fn test2() {
    let x: Foo<_> = Bar::<usize>(PhantomData);
    //~^ ERROR mismatched types
    //~| NOTE expected struct `Foo<_>`
    //~| NOTE found struct `Bar<usize>`
    //~| NOTE expected `Foo<_>`, found `Bar<usize>`
}
