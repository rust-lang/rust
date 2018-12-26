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

#[no_mangle]
#[cfg(d)]
pub fn foo() -> usize {
    use std::cell::Cell;
    thread_local!(static A: Cell<Vec<u32>> = Cell::new(Vec::new()));
    A.try_with(|x| x.replace(Vec::new()).len()).unwrap_or(0)
}
