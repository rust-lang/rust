//@ known-bug: #150128
fn main() {
    match 0 {
        _ => || (),
        _ => || (),
        _ => (|| ()) as unsafe fn,
    };
}
