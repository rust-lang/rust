// ignore-emscripten

fn main() {
    unsafe {
        asm!(""); //~ ERROR inline assembly is not stable enough
        //~^ WARN use of deprecated item 'asm'
        llvm_asm!(""); //~ ERROR inline assembly is not stable enough
    }
}
