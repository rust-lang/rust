// Test that we are able to infer a suitable kind for this closure
// that is just called (`FnMut`).

fn main() {
    let mut counter = 0;
    let tick = || counter += 1;
    tick(); //~ ERROR cannot borrow `tick` as mutable, as it is not declared as mutable
}
