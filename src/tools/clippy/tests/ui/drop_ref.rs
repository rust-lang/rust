#![warn(clippy::drop_ref)]
#![allow(clippy::toplevel_ref_arg)]
#![allow(clippy::map_err_ignore)]
#![allow(clippy::unnecessary_wraps)]

use std::mem::drop;

struct SomeStruct;

fn main() {
    drop(&SomeStruct);

    let mut owned1 = SomeStruct;
    drop(&owned1);
    drop(&&owned1);
    drop(&mut owned1);
    drop(owned1); //OK

    let reference1 = &SomeStruct;
    drop(reference1);

    let reference2 = &mut SomeStruct;
    drop(reference2);

    let ref reference3 = SomeStruct;
    drop(reference3);
}

#[allow(dead_code)]
fn test_generic_fn_drop<T>(val: T) {
    drop(&val);
    drop(val); //OK
}

#[allow(dead_code)]
fn test_similarly_named_function() {
    fn drop<T>(_val: T) {}
    drop(&SomeStruct); //OK; call to unrelated function which happens to have the same name
    std::mem::drop(&SomeStruct);
}

#[derive(Copy, Clone)]
pub struct Error;
fn produce_half_owl_error() -> Result<(), Error> {
    Ok(())
}

fn produce_half_owl_ok() -> Result<bool, ()> {
    Ok(true)
}

#[allow(dead_code)]
fn test_owl_result() -> Result<(), ()> {
    produce_half_owl_error().map_err(|_| ())?;
    produce_half_owl_ok().map(|_| ())?;
    // the following should not be linted,
    // we should not force users to use toilet closures
    // to produce owl results when drop is more convenient
    produce_half_owl_error().map_err(drop)?;
    produce_half_owl_ok().map_err(drop)?;
    Ok(())
}

#[allow(dead_code)]
fn test_owl_result_2() -> Result<u8, ()> {
    produce_half_owl_error().map_err(|_| ())?;
    produce_half_owl_ok().map(|_| ())?;
    // the following should not be linted,
    // we should not force users to use toilet closures
    // to produce owl results when drop is more convenient
    produce_half_owl_error().map_err(drop)?;
    produce_half_owl_ok().map(drop)?;
    Ok(1)
}
