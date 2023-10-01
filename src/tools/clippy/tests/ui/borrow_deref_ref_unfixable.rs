//@no-rustfix: overlapping suggestions
#![allow(dead_code, unused_variables)]

fn main() {}

mod should_lint {
    fn two_helps() {
        let s = &String::new();
        let x: &str = &*s;
        //~^ ERROR: deref on an immutable reference
        //~| NOTE: `-D clippy::borrow-deref-ref` implied by `-D warnings`
    }
}
