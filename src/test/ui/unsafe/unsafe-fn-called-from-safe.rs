// revisions: mir thir
// [thir]compile-flags: -Z thir-unsafeck

unsafe fn f() { return; }

fn main() {
    f(); //~ ERROR call to unsafe function is unsafe
}
