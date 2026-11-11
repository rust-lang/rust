//! Test that we can return a captured `&mut` from a closure
//! (because we aren't evil, unlike `mut-autoreborrow-closure-cursed-fail`).
//@ run-pass

fn main() {
    let x = &mut Box::new(0);
    let f = || {
        let y = x;
        y
    };
    f();
}
