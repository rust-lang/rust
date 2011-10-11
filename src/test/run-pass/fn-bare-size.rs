// xfail-test

use std;

fn main() {
    // Bare functions should just be a pointer
    assert std::sys::rustrt::size_of::<fn#()>() ==
        std::sys::rustrt::size_of::<int>();
}