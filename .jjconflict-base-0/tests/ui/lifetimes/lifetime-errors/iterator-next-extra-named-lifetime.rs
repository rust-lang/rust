//! Regression test for <https://github.com/rust-lang/rust/issues/37884>.
//! This used to leak compiler data structures in error message.
//@ dont-require-annotations: NOTE

struct RepeatMut<'a, T>(T, &'a ());

impl<'a, T: 'a> Iterator for RepeatMut<'a, T> {

    type Item = &'a mut T;
    fn next(&'a mut self) -> Option<Self::Item>
    //~^ ERROR method not compatible with trait
    //~| NOTE lifetime mismatch
    {
        Some(&mut self.0)
    }
}

fn main() {}
