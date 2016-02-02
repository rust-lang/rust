#![feature(plugin)]
#![plugin(clippy)]

#![deny(drop_ref)]
#![allow(toplevel_ref_arg)]

use std::mem::drop;

struct DroppableStruct;
impl Drop for DroppableStruct { fn drop(&mut self) {} }

fn main() {
    drop(&DroppableStruct); //~ERROR call to `std::mem::drop` with a reference argument

    let mut owned = DroppableStruct;
    drop(&owned); //~ERROR call to `std::mem::drop` with a reference argument
    drop(&&owned); //~ERROR call to `std::mem::drop` with a reference argument
    drop(&mut owned); //~ERROR call to `std::mem::drop` with a reference argument
    drop(owned); //OK

    let reference1 = &DroppableStruct;
    drop(reference1); //~ERROR call to `std::mem::drop` with a reference argument
    drop(&*reference1); //~ERROR call to `std::mem::drop` with a reference argument

    let reference2 = &mut DroppableStruct;
    drop(reference2); //~ERROR call to `std::mem::drop` with a reference argument

    let ref reference3 = DroppableStruct;
    drop(reference3); //~ERROR call to `std::mem::drop` with a reference argument
}

#[allow(dead_code)]
fn test_generic_fn<T>(val: T) {
    drop(&val); //~ERROR call to `std::mem::drop` with a reference argument
    drop(val); //OK
}

#[allow(dead_code)]
fn test_similarly_named_function() {
    fn drop<T>(_val: T) {}
    drop(&DroppableStruct); //OK; call to unrelated function which happens to have the same name
    std::mem::drop(&DroppableStruct); //~ERROR call to `std::mem::drop` with a reference argument
}
