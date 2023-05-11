// revisions: mir thir
// [thir]compile-flags: -Z thir-unsafeck

fn f(p: *const u8) -> u8 {
    let _ = *p; //~ ERROR dereference of raw pointer is unsafe
    let _: u8 = *p; //~ ERROR dereference of raw pointer is unsafe
    _ = *p; //~ ERROR dereference of raw pointer is unsafe
    return *p; //~ ERROR dereference of raw pointer is unsafe
}

fn main() {
}
