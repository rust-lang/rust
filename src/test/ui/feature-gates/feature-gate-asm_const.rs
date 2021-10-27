// only-x86_64

#![feature(asm)]

fn main() {
    unsafe {
        asm!("mov eax, {}", const 123);
        //~^ ERROR const operands for inline assembly are unstable
    }
}
