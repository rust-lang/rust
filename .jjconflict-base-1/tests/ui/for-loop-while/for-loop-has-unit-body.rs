//@ run-pass
fn main() {
    // Check that the tail statement in the body unifies with something
    for _ in 0..3 {
        // `()` is fine to zero-initialize as it is zero sized and inhabited.
        unsafe { std::mem::zeroed() }
    }

    // Check that the tail statement in the body can be unit
    for _ in 0..3 {
        ()
    }
}
