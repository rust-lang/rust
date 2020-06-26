// ignore-emscripten

fn main() {
    unsafe {
        println!("{:?}", asm!(""));
        //~^ ERROR inline assembly is not stable enough
        println!("{:?}", llvm_asm!(""));
        //~^ ERROR prefer using the new asm! syntax instead
    }
}
