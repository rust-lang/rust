#![warn(clippy::forget_ref)]
#![allow(clippy::toplevel_ref_arg)]
#![allow(clippy::unnecessary_wraps)]

use std::mem::forget;

struct SomeStruct;

fn main() {
    forget(&SomeStruct);

    let mut owned = SomeStruct;
    forget(&owned);
    forget(&&owned);
    forget(&mut owned);
    forget(owned); //OK

    let reference1 = &SomeStruct;
    forget(&*reference1);

    let reference2 = &mut SomeStruct;
    forget(reference2);

    let ref reference3 = SomeStruct;
    forget(reference3);
}

#[allow(dead_code)]
fn test_generic_fn_forget<T>(val: T) {
    forget(&val);
    forget(val); //OK
}

#[allow(dead_code)]
fn test_similarly_named_function() {
    fn forget<T>(_val: T) {}
    forget(&SomeStruct); //OK; call to unrelated function which happens to have the same name
    std::mem::forget(&SomeStruct);
}

#[derive(Copy, Clone)]
pub struct Error;
fn produce_half_owl_error() -> Result<(), Error> {
    Ok(())
}

fn produce_half_owl_ok() -> Result<bool, ()> {
    Ok(true)
}
