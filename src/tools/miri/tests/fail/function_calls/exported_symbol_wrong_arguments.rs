#[no_mangle]
fn foo() {} //~ ERROR: calling a function with more arguments than it expected

fn main() {
    extern "Rust" {
        fn foo(_: i32);
    }
    unsafe { foo(1) }
}
