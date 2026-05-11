//@ run-pass

#![feature(reborrow)]
use std::marker::{Reborrow, PhantomData};

struct CustomMarker<'a>(PhantomData<&'a ()>);
impl<'a> Reborrow for CustomMarker<'a> {}

impl<'a> std::ops::Deref for CustomMarker<'a> {
    type Target = CustomMarker<'a>;
    fn deref(&self) -> &Self::Target {
        self
    }
}

impl<'a> std::ops::DerefMut for CustomMarker<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self
    }
}

fn main() {
    let mut a = CustomMarker(PhantomData);

    *a = CustomMarker(PhantomData);
}
