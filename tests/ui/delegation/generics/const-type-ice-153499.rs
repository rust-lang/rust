#![feature(fn_delegation)]
#![allow(incomplete_features)]

trait Trait<'a, T, const F: fn(&CStr) -> usize> {
    //~^ ERROR: cannot find type `CStr` in this scope
    //~| ERROR: using function pointers as const generic parameters is forbidden
    fn foo<'x: 'x, A, B>(&self) {}
}

reuse Trait::foo;
//~^ ERROR: using function pointers as const generic parameters is forbidden

fn main() {}
