//! Test that functions with unnamed arguments are correct handled by compiler

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
