#![feature(asm)]

fn main() {
    asm!(); //~ ERROR requires a string literal as an argument
    asm!("nop" : struct); //~ ERROR expected string literal
    asm!("mov %eax, $$0x2" : struct); //~ ERROR expected string literal
    asm!("mov %eax, $$0x2" : "={eax}" struct); //~ ERROR expected `(`
    asm!("mov %eax, $$0x2" : "={eax}"(struct)); //~ ERROR expected expression
    asm!("in %dx, %al" : "={al}"(result) : struct); //~ ERROR expected string literal
    asm!("in %dx, %al" : "={al}"(result) : "{dx}" struct); //~ ERROR expected `(`
    asm!("in %dx, %al" : "={al}"(result) : "{dx}"(struct)); //~ ERROR expected expression
    asm!("mov $$0x200, %eax" : : : struct); //~ ERROR expected string literal
    asm!("mov eax, 2" : "={eax}"(foo) : : : struct); //~ ERROR expected string literal
    asm!(123); //~ ERROR inline assembly must be a string literal
}
