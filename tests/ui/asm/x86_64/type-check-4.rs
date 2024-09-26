//@ only-x86_64
//@ build-fail

use std::arch::global_asm;

fn main() {}

// Constants must be... constant

static mut S: i32 = 1;
const fn const_foo(x: i32) -> i32 {
    x
}
const fn const_bar<T>(x: T) -> T {
    x
}
global_asm!("{}", const unsafe { S });
//~^ ERROR evaluation of constant value failed
//~| mutable global memory
global_asm!("{}", const const_foo(0));
global_asm!("{}", const const_foo(unsafe { S }));
//~^ ERROR evaluation of constant value failed
//~| mutable global memory
global_asm!("{}", const const_bar(0));
global_asm!("{}", const const_bar(unsafe { S }));
//~^ ERROR evaluation of constant value failed
//~| mutable global memory
