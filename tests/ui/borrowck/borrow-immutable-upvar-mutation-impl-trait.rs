#![feature(unboxed_closures)]

// Tests that we can't assign to or mutably borrow upvars from `Fn`
// closures (issue #17780)

fn main() {}

fn bar() -> impl Fn() -> usize {
    let mut x = 0;
    move || {
        x += 1; //~ ERROR cannot assign
        x
    }
}
