// ignore-windows: Unwind panicking does not currently work on Windows
fn main() {
    let val = "Value".to_string();
    panic!("Miri panic with value: {}", val);
}
