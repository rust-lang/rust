//@ run-pass
// Test that we are able to infer a suitable kind for this `move`
// closure that is just called (`FnMut`).

fn main() {
    let mut counter = 0;

    let v = {
        let mut tick = move || { counter += 1; counter };
        tick();
        tick()
    };

    assert_eq!(counter, 0);
    assert_eq!(v, 2);
}
