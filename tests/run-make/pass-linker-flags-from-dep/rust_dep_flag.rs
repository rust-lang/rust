extern "C" {
    pub fn foo();
}

pub fn f() {
    unsafe {
        foo();
    }
}
