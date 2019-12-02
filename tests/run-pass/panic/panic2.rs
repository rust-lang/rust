// ignore-windows: Unwind panicking does not currently work on Windows
fn main() {
    std::panic!("{}-panicking from libstd", 42);
}
