fn main() {
    unsafe {
        asm!(""); //~ ERROR inline assembly is not stable enough
    }
}
