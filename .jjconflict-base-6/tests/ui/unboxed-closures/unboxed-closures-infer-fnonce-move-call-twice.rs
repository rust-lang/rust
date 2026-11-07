// Test that we are able to infer a suitable kind for this closure
// that is just called (`FnMut`).

use std::mem;

fn main() {
    let mut counter: Vec<i32> = Vec::new();
    let tick = move || mem::drop(counter);
    tick();
    tick(); //~ ERROR use of moved value: `tick`
}
