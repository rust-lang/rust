// ignore-emscripten

fn main() {
    unsafe {
        println!("{:?}", asm!(""));
        //~^ ERROR inline assembly is not stable enough
        println!("{:?}", llvm_asm!(""));
        //~^ ERROR LLVM-style inline assembly will never be stabilized
    }
}
