// run-pass
fn main() {
    // Check that the tail statement in the body unifies with something
    for _ in 0..3 {
        #[allow(deprecated)]
        unsafe { std::mem::uninitialized() }
    }

    // Check that the tail statement in the body can be unit
    for _ in 0..3 {
        ()
    }
}
