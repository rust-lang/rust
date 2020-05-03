#![deny(unused_unsafe)]

unsafe fn unsf() {}

unsafe fn foo() {
    unsafe { //~ ERROR unnecessary `unsafe` block
        unsf()
    }
}

fn main() {}
