#![warn(const_err)]

#![feature(const_fn)]

const fn foo(x: u32) -> u32 {
    x
}

fn main() {
    const X: u32 = 0-1;
    //~^ WARN this constant cannot be used
    const Y: u32 = foo(0-1);
    //~^ WARN this constant cannot be used
    println!("{} {}", X, Y);
    //~^ ERROR erroneous constant used
    //~| ERROR erroneous constant used
    //~| ERROR E0080
    //~| ERROR E0080
}
