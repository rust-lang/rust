#![crate_type = "cdylib"]

#[no_mangle]
#[cfg(a)]
pub fn foo() {
    panic!("test");
}

#[no_mangle]
#[cfg(b)]
pub fn foo() {
    panic!("{}", 1);
}

#[no_mangle]
#[cfg(c)]
pub fn foo() {
    panic!("{}", "a");
}
