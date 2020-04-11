// ignore-emscripten

fn main() {
    unsafe {
        println!("{:?}", asm!("")); //~ ERROR inline assembly is not stable
        //~^ WARN use of deprecated item 'asm'
        println!("{:?}", llvm_asm!("")); //~ ERROR inline assembly is not stable
    }
}
