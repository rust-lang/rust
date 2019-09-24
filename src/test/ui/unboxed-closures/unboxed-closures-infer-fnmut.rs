// run-pass
// Test that we are able to infer a suitable kind for this closure
// that is just called (`FnMut`).

fn main() {
    let mut counter = 0;

    {
        let mut tick = || counter += 1;
        tick();
        tick();
    }

    assert_eq!(counter, 2);
}
