#![feature(llvm_asm)]

fn main() {
    let a;
    llvm_asm!("nop" "nop");
    //~^ ERROR E0660
    llvm_asm!("nop" "nop" : "=r"(a));
    //~^ ERROR E0660
}
