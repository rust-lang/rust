// check-pass

const fn foo<T>() -> usize {
    if std::mem::size_of::<*mut T>() < 8 { // size of *mut T does not depend on T
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
