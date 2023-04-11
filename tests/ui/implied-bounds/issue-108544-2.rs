// check-pass

// Similar to issue-108544.rs except that we have a generic `T` which
// previously caused an overeager fast-path to trigger.
use std::borrow::Cow;

pub trait Trait<T: Clone> {
    fn method(self) -> Option<Cow<'static, T>>
    where
        Self: Sized;
}

impl<'a, T: Clone> Trait<T> for Cow<'a, T> {
    fn method(self) -> Option<Cow<'static, T>> {
        None
    }
}

fn main() {}
