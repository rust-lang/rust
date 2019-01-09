// error-pattern:thread 'main' panicked at 'attempt to multiply with overflow'
// compile-flags: -C debug-assertions

fn main() {
    let _x = 2u32.pow(1024);
}
