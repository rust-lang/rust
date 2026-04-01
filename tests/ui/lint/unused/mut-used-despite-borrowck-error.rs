//! Do not fire unused_mut lint when mutation of the bound variable fails due to a borrow-checking
//! error.
//!
//! Regression test for https://github.com/rust-lang/rust/issues/152024
//@ compile-flags: -W unused_mut

struct Thing;
impl Drop for Thing {
    fn drop(&mut self) {}
}

fn main() {
    let mut t;
    let mut b = None;
    loop {
        t = Thing; //~ ERROR cannot assign to `t` because it is borrowed
        b.insert(&t);
    }
}
