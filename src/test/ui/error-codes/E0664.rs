// ignore-emscripten

#![feature(llvm_asm)]

fn main() {
    llvm_asm!("mov $$0x200, %eax"
              :
              :
              : "{eax}" //~ ERROR E0664
             );
}
