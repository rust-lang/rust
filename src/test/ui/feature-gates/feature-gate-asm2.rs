// ignore-emscripten

fn main() {
    unsafe {
        println!("{:?}", asm!("")); //~ ERROR inline assembly is not stable
    }
}
