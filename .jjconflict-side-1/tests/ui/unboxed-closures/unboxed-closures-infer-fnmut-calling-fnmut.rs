//@ run-pass
// Test that we are able to infer a suitable kind for this closure
// that is just called (`FnMut`).

fn main() {
    let mut counter = 0;

    {
        // Here this must be inferred to FnMut so that it can mutate counter:
        let mut tick1 = || counter += 1;

        // In turn, tick2 must be inferred to FnMut so that it can call tick1:
        let mut tick2 = || { tick1(); tick1(); };

        tick2();
    }

    assert_eq!(counter, 2);
}
