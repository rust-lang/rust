// revisions: mir thir
// [thir]compile-flags: -Z thir-unsafeck

fn f(p: *mut u8) {
    *p = 0; //~ ERROR dereference of raw pointer is unsafe
    return;
}

fn main() {
}
