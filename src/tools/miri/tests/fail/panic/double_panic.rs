//@error-pattern: the program aborted
//@normalize-stderr-test: "\| +\^+" -> "| ^"
//@normalize-stderr-test: "unsafe \{ libc::abort\(\) \}|crate::intrinsics::abort\(\);" -> "ABORT();"
//@normalize-stderr-test: "\n  +[0-9]+:[^\n]+" -> "$1"
//@normalize-stderr-test: "\n at [^\n]+" -> "$1"

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
