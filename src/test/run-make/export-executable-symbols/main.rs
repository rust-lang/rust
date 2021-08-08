// edition:2018

fn main() {
    foo();
}

#[no_mangle]
pub extern "C" fn foo() -> i32 {
    1 + 1
}
