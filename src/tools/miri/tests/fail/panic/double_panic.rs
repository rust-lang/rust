//@error-pattern: the program aborted
//@normalize-stderr-test: "\| +\^+" -> "| ^"
//@normalize-stderr-test: "unsafe \{ libc::abort\(\) \}|crate::intrinsics::abort\(\);" -> "ABORT();"

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
