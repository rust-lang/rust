#![feature(asm)]

fn main() {
    let a;
    asm!("nop" "nop");
    //~^ ERROR E0660
    asm!("nop" "nop" : "=r"(a));
    //~^ ERROR E0660
}
