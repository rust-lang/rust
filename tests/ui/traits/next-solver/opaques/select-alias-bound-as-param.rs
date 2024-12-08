//@ check-pass
//@ compile-flags: -Znext-solver

pub(crate) fn y() -> impl FnMut() {
    || {}
}

pub(crate) fn x(a: (), b: ()) {
    let x = ();
    y()()
}

fn main() {}
