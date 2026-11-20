#![crate_type = "staticlib"]

#[no_mangle]
pub extern "C" fn foo(x: u32) {
    // using the println! makes it so that enough code from the standard
    // library is included (see issue #68794)
    println!("foo: {}", x);
}
