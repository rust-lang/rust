// issue #12418

// revisions: mir thir
// [thir]compile-flags: -Z thir-unsafeck

#![deny(unused_unsafe)]

fn main() {
    unsafe { println!("foo"); } //~ ERROR unnecessary `unsafe`
}
