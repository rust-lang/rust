
// check-pass

use std::borrow::Cow;

pub trait Trait {
    fn method(self) -> Option<Cow<'static, str>>
    where
        Self: Sized;
}

impl<'a> Trait for Cow<'a, str> {
    // We have to check `WF(return-type)` which requires `Cow<'static, str>: Sized`.
    // If we use the `Self: Sized` bound from the trait method we end up equating
    // `Cow<'a, str>` with `Cow<'static, str>`, causing an error.
    fn method(self) -> Option<Cow<'static, str>> {
        None
    }
}

fn main() {}
