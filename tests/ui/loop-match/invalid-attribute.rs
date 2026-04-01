// Test that the `#[loop_match]` and `#[const_continue]` attributes can only be
// placed on expressions.

#![allow(incomplete_features)]
#![feature(loop_match)]
#![loop_match] //~ ERROR attribute cannot be used on
#![const_continue] //~ ERROR attribute cannot be used on

extern "C" {
    #[loop_match] //~ ERROR attribute cannot be used on
    #[const_continue] //~ ERROR attribute cannot be used on
    fn f();
}

#[loop_match] //~ ERROR attribute cannot be used on
#[const_continue] //~ ERROR attribute cannot be used on
#[repr(C)]
struct S {
    a: u32,
    b: u32,
}

trait Invoke {
    #[loop_match] //~ ERROR attribute cannot be used on
    #[const_continue] //~ ERROR attribute cannot be used on
    extern "C" fn invoke(&self);
}

#[loop_match] //~ ERROR attribute cannot be used on
#[const_continue] //~ ERROR attribute cannot be used on
extern "C" fn ok() {}

fn main() {
    #[loop_match] //~ ERROR attribute cannot be used on
    #[const_continue] //~ ERROR attribute cannot be used on
    || {};

    {
        #[loop_match] //~ ERROR should be applied to a loop
        #[const_continue] //~ ERROR should be applied to a break expression
        5
    };
}
