#![crate_type = "cdylib"]

#[no_mangle]
pub extern "C" fn foo() {
    println!("foo");
    panic!("test");
}
