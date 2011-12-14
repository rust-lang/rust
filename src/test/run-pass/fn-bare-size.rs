// xfail-test

use std;

fn main() {
    // Bare functions should just be a pointer
    assert sys::rustrt::size_of::<fn()>() ==
        sys::rustrt::size_of::<int>();
}