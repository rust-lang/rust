// revisions: mir thir
// [thir]compile-flags: -Z thir-unsafeck

static mut A: isize = 3;

fn main() {
    A += 3; //~ ERROR: requires unsafe
    A = 4; //~ ERROR: requires unsafe
    let _b = A; //~ ERROR: requires unsafe
}
