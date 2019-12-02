// ignore-windows: Unwind panicking does not currently work on Windows
fn main() {
    core::panic!("panicking from libcore");
}
