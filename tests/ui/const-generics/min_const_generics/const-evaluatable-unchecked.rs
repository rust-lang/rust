//@ check-pass
//@ compile-flags: -Zdeduplicate-diagnostics=yes
#![allow(dead_code)]

fn foo<T>() {
    [0; std::mem::size_of::<*mut T>()];
    //~^ WARN cannot use constants which depend on generic parameters in types
    //~| WARN this was previously accepted by the compiler but is being phased out
}

struct Foo<T>(T);

impl<T> Foo<T> {
    const ASSOC: usize = 4;

    fn test() {
        let _ = [0; Self::ASSOC];
        //~^ WARN cannot use constants which depend on generic parameters in types
        //~| WARN this was previously accepted by the compiler but is being phased out
    }
}

struct Bar<const N: usize>;

impl<const N: usize> Bar<N> {
    const ASSOC: usize = 4;

    fn test() {
        let _ = [0; Self::ASSOC];
        //~^ WARN cannot use constants which depend on generic parameters in types
        //~| WARN this was previously accepted by the compiler but is being phased out
    }
}

fn main() {}
