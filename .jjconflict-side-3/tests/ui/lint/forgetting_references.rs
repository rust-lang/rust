//@ check-pass

#![warn(forgetting_references)]

use std::mem::forget;

struct SomeStruct;

fn main() {
    forget(&SomeStruct); //~ WARN calls to `std::mem::forget`

    let mut owned = SomeStruct;
    forget(&owned); //~ WARN calls to `std::mem::forget`
    forget(&&owned); //~ WARN calls to `std::mem::forget`
    forget(&mut owned); //~ WARN calls to `std::mem::forget`
    forget(owned);

    let reference1 = &SomeStruct;
    forget(&*reference1); //~ WARN calls to `std::mem::forget`

    let reference2 = &mut SomeStruct;
    forget(reference2); //~ WARN calls to `std::mem::forget`

    let ref reference3 = SomeStruct;
    forget(reference3); //~ WARN calls to `std::mem::forget`

    let ref reference4 = SomeStruct;

    let a = 1;
    match a {
        1 => forget(&*reference1), //~ WARN calls to `std::mem::forget`
        2 => forget(reference3), //~ WARN calls to `std::mem::forget`
        3 => forget(reference4), //~ WARN calls to `std::mem::forget`
        _ => {}
    }
}

#[allow(dead_code)]
fn test_generic_fn_forget<T>(val: T) {
    forget(&val); //~ WARN calls to `std::mem::forget`
    forget(val);
}

#[allow(dead_code)]
fn test_similarly_named_function() {
    fn forget<T>(_val: T) {}
    forget(&SomeStruct); //OK; call to unrelated function which happens to have the same name
    std::mem::forget(&SomeStruct); //~ WARN calls to `std::mem::forget`
}
