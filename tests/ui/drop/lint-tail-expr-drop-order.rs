//@compile-flags: -Z unstable-options
//@edition:2024

#![deny(tail_expr_drop_order)]
#![feature(shorter_tail_lifetimes)]

struct LoudDropper;
impl Drop for LoudDropper {
    fn drop(&mut self) {
        println!("loud drop")
    }
}
impl LoudDropper {
    fn get(&self) -> i32 {
        0
    }
}

fn should_lint() -> i32 {
    let x = LoudDropper;
    // Should lint
    x.get() + LoudDropper.get()
    //~^ ERROR: these values and local bindings have significant drop implementation that will have a different drop order from that of Edition 2021
    //~| WARN: this changes meaning in Rust 2024
}

fn should_not_lint() -> i32 {
    let x = LoudDropper;
    // Should not lint
    x.get()
}

fn main() {}
