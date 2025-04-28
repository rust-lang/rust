#![allow(
    dead_code,
    unused_must_use
)]

struct Named {
    foo: usize,
}

struct Unnamed(usize);

unsafe fn named_struct_field_access(named: *mut Named) {
    named->foo += 1; //~ ERROR `->` is not valid syntax for field accesses and method calls
    //~^ ERROR no field `foo` on type `*mut Named`
}

unsafe fn unnamed_struct_field_access(unnamed: *mut Unnamed) {
    unnamed->0 += 1; //~ ERROR `->` is not valid syntax for field accesses and method calls
    //~^ ERROR no field `0` on type `*mut Unnamed`
}

fn main() {}
