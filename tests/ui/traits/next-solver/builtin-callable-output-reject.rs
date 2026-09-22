//@ revisions: output lifetime unsafe_fn
//@ compile-flags: -Znext-solver=globally

#![allow(dead_code)]

fn invoke<A, R, F: FnOnce(A) -> R>(f: F, arg: A) -> R {
    f(arg)
}

fn borrow<'a>(value: &'a u32) -> &'a u32 {
    value
}

#[cfg(output)]
fn wrong_output(value: &u32) {
    let _: bool = invoke(borrow, value);
    //[output]~^ ERROR mismatched types
    let pointer: for<'a> fn(&'a u32) -> &'a u32 = borrow;
    let _: bool = invoke(pointer, value);
    //[output]~^ ERROR mismatched types
}

#[cfg(lifetime)]
fn escaping_item(value: &u32) -> &'static u32 {
    invoke(borrow, value)
    //[lifetime]~^ ERROR lifetime may not live long enough
}

#[cfg(lifetime)]
fn escaping_pointer(value: &u32) -> &'static u32 {
    let pointer: for<'a> fn(&'a u32) -> &'a u32 = borrow;
    invoke(pointer, value)
    //[lifetime]~^ ERROR lifetime may not live long enough
}

#[cfg(unsafe_fn)]
fn incompatible_signature() {
    unsafe fn unsafe_identity(value: u32) -> u32 {
        value
    }
    let _ = invoke(unsafe_identity, 1);
    //[unsafe_fn]~^ ERROR E0277
    let pointer: unsafe fn(u32) -> u32 = unsafe_identity;
    let _ = invoke(pointer, 1);
    //[unsafe_fn]~^ ERROR E0277
}

fn main() {}
