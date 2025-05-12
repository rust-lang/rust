//@ run-rustfix

#![allow(unused)]
#![deny(dependency_on_unit_never_type_fallback)]

fn foo<T: Default>() -> Result<T, ()> {
    Err(())
}

fn test() -> Result<(), ()> {
    //~^ ERROR this function depends on never type fallback being `()`
    //~| WARN this was previously accepted by the compiler but is being phased out
    _ = foo()?;
    Ok(())
}

fn main() {}
