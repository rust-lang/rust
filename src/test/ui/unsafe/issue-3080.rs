// revisions: mir thir
// [thir]compile-flags: -Z thir-unsafeck

struct X(());
impl X {
    pub unsafe fn with(&self) { }
}

fn main() {
    X(()).with(); //~ ERROR requires unsafe function or block
}
