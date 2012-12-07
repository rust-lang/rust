fn main() {

    // Testing that method lookup does not automatically borrow
    // vectors to slices then automatically create a &mut self
    // reference.  That would allow creating a mutable pointer to a
    // temporary, which would be a source of confusion

    let mut a = @[0];
    a.test_mut(); //~ ERROR type `@[int]` does not implement any method in scope named `test_mut`
}

trait MyIter {
    pure fn test_mut(&mut self);
}

impl &[int]: MyIter {
    pure fn test_mut(&mut self) { }
}
