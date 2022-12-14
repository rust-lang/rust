use std::arch::global_asm;

fn main() {
    global_asm!(); //~ ERROR requires at least a template string argument
    global_asm!(struct); //~ ERROR expected expression
    global_asm!(123); //~ ERROR asm template must be a string literal
}
