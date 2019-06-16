fn main() {
    // Check that the tail statement in the body unifies with something
    for _ in 0..3 {
        unsafe { std::mem::uninitialized() } //~ ERROR type annotations needed
    }

    // Check that the tail statement in the body can be unit
    for _ in 0..3 {
        ()
    }
}
