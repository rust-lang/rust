// edition:2018
// revisions: mir thir
// [thir]compile-flags: -Z thir-unsafeck

#![deny(unused_unsafe)]

fn main() {
    let _ = async {
        unsafe { async {}.await; } //~ ERROR unnecessary `unsafe`
    };
}
