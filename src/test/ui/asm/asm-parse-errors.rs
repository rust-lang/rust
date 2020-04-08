#![feature(llvm_asm)]

fn main() {
    llvm_asm!(); //~ ERROR requires a string literal as an argument
    llvm_asm!("nop" : struct); //~ ERROR expected string literal
    llvm_asm!("mov %eax, $$0x2" : struct); //~ ERROR expected string literal
    llvm_asm!("mov %eax, $$0x2" : "={eax}" struct); //~ ERROR expected `(`
    llvm_asm!("mov %eax, $$0x2" : "={eax}"(struct)); //~ ERROR expected expression
    llvm_asm!("in %dx, %al" : "={al}"(result) : struct); //~ ERROR expected string literal
    llvm_asm!("in %dx, %al" : "={al}"(result) : "{dx}" struct); //~ ERROR expected `(`
    llvm_asm!("in %dx, %al" : "={al}"(result) : "{dx}"(struct)); //~ ERROR expected expression
    llvm_asm!("mov $$0x200, %eax" : : : struct); //~ ERROR expected string literal
    llvm_asm!("mov eax, 2" : "={eax}"(foo) : : : struct); //~ ERROR expected string literal
    llvm_asm!(123); //~ ERROR inline assembly must be a string literal
}
