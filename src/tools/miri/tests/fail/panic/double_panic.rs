//@normalize-stderr-test: "\| +\^+" -> "| ^"
//@normalize-stderr-test: "\n  +[0-9]+:[^\n]+" -> "$1"
//@normalize-stderr-test: "\n at [^\n]+" -> "$1"

struct Foo;
impl Drop for Foo {
    fn drop(&mut self) {
        panic!("second");
    }
}
fn main() {
    //~^ERROR: panic in a function that cannot unwind
    let _foo = Foo;
    panic!("first");
}
