// run-fail
//@error-in-other-file:index out of bounds
//@ignore-target-emscripten no processes

use std::mem::size_of;

fn main() {
    let xs = [1, 2, 3];
    xs[usize::MAX / size_of::<isize>() + 1];
}
