//@normalize-stderr-test: "\n +[0-9]+:[^\n]+" -> ""
//@normalize-stderr-test: "\n +at [^\n]+" -> ""
//@error-in-other-file: aborted execution

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
