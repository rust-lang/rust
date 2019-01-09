#![crate_type = "cdylib"]

#[no_mangle]
pub extern fn foo() {
    println!("foo");
    panic!("test");
}
