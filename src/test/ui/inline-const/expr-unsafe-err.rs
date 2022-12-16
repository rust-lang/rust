// revisions: mir thir
// [thir]compile-flags: -Z thir-unsafeck
#![feature(inline_const)]
const unsafe fn require_unsafe() -> usize { 1 }

fn main() {
    const {
        require_unsafe();
        //~^ ERROR [E0133]
    }
}
