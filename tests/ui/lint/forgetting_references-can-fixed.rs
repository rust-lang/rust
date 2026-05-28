//@ check-fail
//@ run-rustfix

#![deny(forgetting_references)]

use std::mem::forget;

struct SomeStruct;

fn main() {
    forget(&SomeStruct); //~ ERROR calls to `std::mem::forget`

    let mut owned = SomeStruct;
    forget(&owned); //~ ERROR calls to `std::mem::forget`
    forget(&&owned); //~ ERROR calls to `std::mem::forget`
    forget(&mut owned); //~ ERROR calls to `std::mem::forget`
    forget(owned);

    let reference1 = &SomeStruct;
    forget(&*reference1); //~ ERROR calls to `std::mem::forget`

    let reference2 = &mut SomeStruct;
    forget(reference2); //~ ERROR calls to `std::mem::forget`

    let ref reference3 = SomeStruct;
    forget(reference3); //~ ERROR calls to `std::mem::forget`
}

#[allow(dead_code)]
fn test_generic_fn_forget<T>(val: T) {
    forget(&val); //~ ERROR calls to `std::mem::forget`
    forget(val);
}

#[allow(dead_code)]
fn test_similarly_named_function() {
    fn forget<T>(_val: T) {}
    forget(&SomeStruct); //OK; call to unrelated function which happens to have the same name
    std::mem::forget(&SomeStruct); //~ ERROR calls to `std::mem::forget`
}
