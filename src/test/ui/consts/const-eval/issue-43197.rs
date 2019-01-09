#![warn(const_err)]

const fn foo(x: u32) -> u32 {
    x
}

fn main() {
    const X: u32 = 0-1;
    //~^ WARN any use of this value will cause
    const Y: u32 = foo(0-1);
    //~^ WARN any use of this value will cause
    println!("{} {}", X, Y);
    //~^ ERROR evaluation of constant expression failed
    //~| ERROR evaluation of constant expression failed
}
