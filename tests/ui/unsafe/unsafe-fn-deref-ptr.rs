// revisions: mir thir
// [thir]compile-flags: -Z thir-unsafeck

fn f(p: *const u8) -> u8 {
    return *p; //~ ERROR dereference of raw pointer is unsafe
}

fn main() {
}
