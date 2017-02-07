#![feature(plugin)]
#![plugin(clippy)]

#![deny(drop_ref, forget_ref)]
#![allow(toplevel_ref_arg, similar_names)]

use std::mem::{drop, forget};

struct SomeStruct;

fn main() {
    drop(&SomeStruct); //~ERROR call to `std::mem::drop` with a reference argument
    forget(&SomeStruct); //~ERROR call to `std::mem::forget` with a reference argument

    let mut owned1 = SomeStruct;
    drop(&owned1); //~ERROR call to `std::mem::drop` with a reference argument
    drop(&&owned1); //~ERROR call to `std::mem::drop` with a reference argument
    drop(&mut owned1); //~ERROR call to `std::mem::drop` with a reference argument
    drop(owned1); //OK
    let mut owned2 = SomeStruct;
    forget(&owned2); //~ERROR call to `std::mem::forget` with a reference argument
    forget(&&owned2); //~ERROR call to `std::mem::forget` with a reference argument
    forget(&mut owned2); //~ERROR call to `std::mem::forget` with a reference argument
    forget(owned2); //OK

    let reference1 = &SomeStruct;
    drop(reference1); //~ERROR call to `std::mem::drop` with a reference argument
    forget(&*reference1); //~ERROR call to `std::mem::forget` with a reference argument

    let reference2 = &mut SomeStruct;
    drop(reference2); //~ERROR call to `std::mem::drop` with a reference argument
    let reference3 = &mut SomeStruct;
    forget(reference3); //~ERROR call to `std::mem::forget` with a reference argument

    let ref reference4 = SomeStruct;
    drop(reference4); //~ERROR call to `std::mem::drop` with a reference argument
    forget(reference4); //~ERROR call to `std::mem::forget` with a reference argument
}

#[allow(dead_code)]
fn test_generic_fn_drop<T>(val: T) {
    drop(&val); //~ERROR call to `std::mem::drop` with a reference argument
    drop(val); //OK
}

#[allow(dead_code)]
fn test_generic_fn_forget<T>(val: T) {
    forget(&val); //~ERROR call to `std::mem::forget` with a reference argument
    forget(val); //OK
}

#[allow(dead_code)]
fn test_similarly_named_function() {
    fn drop<T>(_val: T) {}
    drop(&SomeStruct); //OK; call to unrelated function which happens to have the same name
    std::mem::drop(&SomeStruct); //~ERROR call to `std::mem::drop` with a reference argument
    fn forget<T>(_val: T) {}
    forget(&SomeStruct); //OK; call to unrelated function which happens to have the same name
    std::mem::forget(&SomeStruct); //~ERROR call to `std::mem::forget` with a reference argument
}
