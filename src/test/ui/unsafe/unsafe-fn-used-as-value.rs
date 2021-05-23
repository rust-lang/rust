// revisions: mir thir
// [thir]compile-flags: -Z thir-unsafeck

unsafe fn f() { return; }

fn main() {
    let x = f;
    x();    //~ ERROR call to unsafe function is unsafe
}
