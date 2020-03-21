// error-pattern: the evaluated program aborted
// ignore-windows (panics dont work on Windows)

struct Foo;
impl Drop for Foo {
    fn drop(&mut self) {
        panic!("second");
    }
}
fn main() {
    let _foo = Foo;
    panic!("first"); 
}
