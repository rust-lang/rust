#![allow(unused_unsafe)]
#![allow(dead_code)]
#![deny(unsafe_code)]

struct Bar;
struct Bar2;
struct Bar3;

#[allow(unsafe_code)]
mod allowed_unsafe {
    fn allowed() { unsafe {} }
    unsafe fn also_allowed() {}
    unsafe trait AllowedUnsafe { }
    unsafe impl AllowedUnsafe for super::Bar {}
}

macro_rules! unsafe_in_macro {
    () => {
        unsafe {} //~ ERROR: usage of an `unsafe` block
    }
}

unsafe fn baz() {} //~ ERROR: declaration of an `unsafe` function
unsafe trait Foo {} //~ ERROR: declaration of an `unsafe` trait
unsafe impl Foo for Bar {} //~ ERROR: implementation of an `unsafe` trait

trait Baz {
    unsafe fn baz(&self); //~ ERROR: declaration of an `unsafe` method
    unsafe fn provided(&self) {} //~ ERROR: implementation of an `unsafe` method
    unsafe fn provided_override(&self) {} //~ ERROR: implementation of an `unsafe` method
}

impl Baz for Bar {
    unsafe fn baz(&self) {} //~ ERROR: implementation of an `unsafe` method
    unsafe fn provided_override(&self) {} //~ ERROR: implementation of an `unsafe` method
}


#[allow(unsafe_code)]
trait A {
    unsafe fn allowed_unsafe(&self);
    unsafe fn allowed_unsafe_provided(&self) {}
}

#[allow(unsafe_code)]
impl Baz for Bar2 {
    unsafe fn baz(&self) {}
    unsafe fn provided_override(&self) {}
}

impl Baz for Bar3 {
    #[allow(unsafe_code)]
    unsafe fn baz(&self) {}
    unsafe fn provided_override(&self) {} //~ ERROR: implementation of an `unsafe` method
}

#[allow(unsafe_code)]
unsafe trait B {
    fn dummy(&self) {}
}

trait C {
    #[allow(unsafe_code)]
    unsafe fn baz(&self);
    unsafe fn provided(&self) {} //~ ERROR: implementation of an `unsafe` method
}

impl C for Bar {
    #[allow(unsafe_code)]
    unsafe fn baz(&self) {}
    unsafe fn provided(&self) {} //~ ERROR: implementation of an `unsafe` method
}

impl C for Bar2 {
    unsafe fn baz(&self) {} //~ ERROR: implementation of an `unsafe` method
}

trait D {
    #[allow(unsafe_code)]
    unsafe fn unsafe_provided(&self) {}
}

impl D for Bar {}

fn main() {
    unsafe {} //~ ERROR: usage of an `unsafe` block

    unsafe_in_macro!()
}
