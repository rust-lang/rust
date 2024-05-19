//@ run-pass
// Test that we mutate a counter on the stack only when we expect to.

fn call<F>(f: F) where F : FnOnce() {
    f();
}

fn main() {
    let y = vec![format!("Hello"), format!("World")];
    let mut counter = 22_u32;

    call(|| {
        // Move `y`, but do not move `counter`, even though it is read
        // by value (note that it is also mutated).
        for item in y { //~ WARN unused variable: `item`
            let v = counter;
            counter += v;
        }
    });
    assert_eq!(counter, 88);

    call(move || {
        // this mutates a moved copy, and hence doesn't affect original
        counter += 1; //~  WARN value assigned to `counter` is never read
                      //~| WARN unused variable: `counter`
    });
    assert_eq!(counter, 88);
}
