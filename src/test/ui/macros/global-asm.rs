#![feature(global_asm)]

fn main() {
    global_asm!();  //~ ERROR requires a string literal as an argument
    global_asm!(struct); //~ ERROR expected expression
    global_asm!(123); //~ ERROR inline assembly must be a string literal
}
