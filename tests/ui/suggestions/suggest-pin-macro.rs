use std::pin::Pin;
use std::marker::PhantomPinned;

#[derive(Debug)]
struct Test {
    _marker: PhantomPinned,
}
impl Test {
    fn new() -> Self {
        Test {
            _marker: PhantomPinned, // This makes our type `!Unpin`
        }
    }
}

fn dummy(_: &mut Test) {}

pub fn main() {
    let mut test1 = Test::new();
    let mut test1 = unsafe { Pin::new_unchecked(&mut test1) };

    dummy(test1.get_mut()); //~ ERROR E0277
}
