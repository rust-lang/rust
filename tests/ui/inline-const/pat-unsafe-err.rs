// ignore-test This is currently broken
// revisions: mir thir
// [thir]compile-flags: -Z thir-unsafeck

#![allow(incomplete_features)]
#![feature(inline_const_pat)]

const unsafe fn require_unsafe() -> usize { 1 }

fn main() {
    match () {
        const {
            require_unsafe();
            //~^ ERROR [E0133]
        } => (),
    }
}
