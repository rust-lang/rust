//! Test that argument names starting with `_` are usable.

//@ run-pass

fn good(_a: &isize) {}

fn called<F>(_f: F)
where
    F: FnOnce(&isize),
{
}

pub fn main() {
    called(good);
}
