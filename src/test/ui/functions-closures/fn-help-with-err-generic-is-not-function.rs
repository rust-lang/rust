struct Struct<T>(T);
impl Struct<T>
//~^ ERROR cannot find type `T` in this scope
//~| NOTE not found in this scope
//~| HELP you might be missing a type parameter
where
    T: Copy,
    //~^ ERROR cannot find type `T` in this scope
    //~| NOTE not found in this scope
{
    // The part where it claims that there is no method named `len` is a bug. Feel free to fix it.
    // This test is intended to ensure that a different bug, where it claimed
    // that `v` was a function, does not regress.
    fn method(v: Vec<u8>) { v.len(); }
    //~^ ERROR type annotations needed
    //~| NOTE cannot infer type
    //~| NOTE type must be known at this point
    //~| ERROR no method named `len`
    //~| NOTE private field, not a method
}

fn main() {}
