use std::mem;

fn main() {
    let xs = vec![1, 2, 3, 4];
    mem::forget(xs);
}
