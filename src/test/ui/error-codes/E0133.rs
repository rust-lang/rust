// revisions: mir thir
// [thir]compile-flags: -Z thir-unsafeck

unsafe fn f() { return; }

fn main() {
    f();
    //~^ ERROR E0133
}
