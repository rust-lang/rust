//! Regression test for <https://github.com/rust-lang/rust/issues/25515>.
//! Test we don't drop twice when we take a reference through other
//! object like `Rc<dyn Trait>`.
//!
//! This used to drop Foo twice while coerced to a trait object.
//! `&T`, `Rc` and `&mut` dropped any unsized item each time
//! reference was taken.
//!
//! Value was being dropped when taking the address of an unsized field.
//@ run-pass

use std::rc::Rc;

struct Foo<'r>(&'r mut i32);

impl<'r> Drop for Foo<'r> {
    fn drop(&mut self) {
        *self.0 += 1;
    }
}

fn main() {
    let mut drops = 0;

    {
        let _: Rc<dyn Send> = Rc::new(Foo(&mut drops));
    }

    assert_eq!(1, drops);
}
