#![feature(asm)]

fn main() {
    asm!("mov $$0x200, %eax"
         :
         :
         : "{eax}" //~ ERROR E0664
        );
}
