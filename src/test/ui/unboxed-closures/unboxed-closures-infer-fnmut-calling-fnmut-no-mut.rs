// Test that we are able to infer a suitable kind for this closure
// that is just called (`FnMut`).

fn main() {
    let mut counter = 0;

    // Here this must be inferred to FnMut so that it can mutate counter,
    // but we forgot the mut.
    let tick1 = || {
        counter += 1;
    };

    // In turn, tick2 must be inferred to FnMut so that it can call
    // tick1, but we forgot the mut. The error message we currently
    // get seems... suboptimal.
    let tick2 = || { //~ ERROR closure cannot assign to immutable local variable `tick1`
        tick1();
    };

    tick2(); //~ ERROR cannot borrow
}
