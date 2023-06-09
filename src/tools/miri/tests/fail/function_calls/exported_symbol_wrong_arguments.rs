#[no_mangle]
fn foo() {}

fn main() {
    extern "Rust" {
        fn foo(_: i32);
    }
    unsafe { foo(1) } //~ ERROR: calling a function with more arguments than it expected
}
