//@run
//@compile-flags:-C debuginfo=2
//@ignore-target-asmjs wasm2js does not support source maps yet

fn foo() -> impl Copy {
    foo
}
fn main() {
    foo();
}
