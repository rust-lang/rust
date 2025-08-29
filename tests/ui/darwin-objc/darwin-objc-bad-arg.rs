// Test that `objc::class!` and `objc::selector!` only take string literals.

//@ edition: 2024
//@ only-apple

#![feature(darwin_objc)]

use std::os::darwin::objc;

pub fn main() {
    let s = "NSObject";
    objc::class!(s);
    //~^ ERROR attribute value must be a literal

    objc::class!(NSObject);
    //~^ ERROR attribute value must be a literal

    objc::class!(123);
    //~^ ERROR malformed `rustc_objc_class` attribute input [E0539]

    let s = "alloc";
    objc::selector!(s);
    //~^ ERROR attribute value must be a literal

    objc::selector!(alloc);
    //~^ ERROR attribute value must be a literal

    objc::selector!(123);
    //~^ ERROR malformed `rustc_objc_selector` attribute input [E0539]
}
