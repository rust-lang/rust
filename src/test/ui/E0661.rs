#![feature(asm)]

fn main() {
    let a;
    asm!("nop" : "r"(a));
    //~^ ERROR E0661
}
