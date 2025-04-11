// Test that the `#[loop_match]` and `#[const_continue]` attributes can only be
// placed on expressions.

#![allow(incomplete_features)]
#![feature(loop_match)]
#![loop_match] //~ ERROR should be applied to a loop
#![const_continue] //~ ERROR should be applied to a break expression

extern "C" {
    #[loop_match] //~ ERROR should be applied to a loop
    #[const_continue] //~ ERROR should be applied to a break expression
    fn f();
}

#[loop_match] //~ ERROR should be applied to a loop
#[const_continue] //~ ERROR should be applied to a break expression
#[repr(C)]
struct S {
    a: u32,
    b: u32,
}

trait Invoke {
    #[loop_match] //~ ERROR should be applied to a loop
    #[const_continue] //~ ERROR should be applied to a break expression
    extern "C" fn invoke(&self);
}

#[loop_match] //~ ERROR should be applied to a loop
#[const_continue] //~ ERROR should be applied to a break expression
extern "C" fn ok() {}

fn main() {
    #[loop_match] //~ ERROR should be applied to a loop
    #[const_continue] //~ ERROR should be applied to a break expression
    || {};

    {
        #[loop_match] //~ ERROR should be applied to a loop
        #[const_continue] //~ ERROR should be applied to a break expression
        5
    };
}
