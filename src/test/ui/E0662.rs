#![feature(asm)]

fn main() {
    asm!("xor %eax, %eax"
         :
         : "=test"("a") //~ ERROR E0662
        );
}
