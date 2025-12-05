//@ check-fail
//@ run-rustfix

#![deny(dropping_references)]

struct SomeStruct;

fn main() {
    drop(&SomeStruct); //~ ERROR calls to `std::mem::drop`

    let mut owned1 = SomeStruct;
    drop(&owned1); //~ ERROR calls to `std::mem::drop`
    drop(&&owned1); //~ ERROR calls to `std::mem::drop`
    drop(&mut owned1); //~ ERROR calls to `std::mem::drop`
    drop(owned1);

    let reference1 = &SomeStruct;
    drop(reference1); //~ ERROR calls to `std::mem::drop`

    let reference2 = &mut SomeStruct;
    drop(reference2); //~ ERROR calls to `std::mem::drop`

    let ref reference3 = SomeStruct;
    drop(reference3); //~ ERROR calls to `std::mem::drop`
}

#[allow(dead_code)]
fn test_generic_fn_drop<T>(val: T) {
    drop(&val); //~ ERROR calls to `std::mem::drop`
    drop(val);
}
