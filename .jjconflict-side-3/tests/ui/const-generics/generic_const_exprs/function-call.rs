//@ check-pass
//@ compile-flags: -Zdeduplicate-diagnostics=yes

const fn foo<T>() -> usize {
    // We might instead branch on `std::mem::size_of::<*mut T>() < 8` here,
    // which would cause this function to fail on 32 bit systems.
    if false {
        std::mem::size_of::<T>()
    } else {
        8
    }
}

fn test<T>() {
    let _ = [0; foo::<T>()];
    //~^ WARN cannot use constants which depend on generic parameters in types
    //~| WARN this was previously accepted by the compiler but is being phased out
}

fn main() {}
