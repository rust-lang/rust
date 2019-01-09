// run-pass
// Test that we are able to type-check this example. In particular,
// knowing that `T: 'a` allows us to deduce that `[U]: 'a` (because
// when `T=[U]` it implies that `U: 'a`).
//
// Regr. test for live code we found in the wild when fixing #18937.

pub trait Leak<T : ?Sized> {
    fn leak<'a>(self) -> &'a T where T: 'a;
}

impl<U> Leak<[U]> for Vec<U> {
    fn leak<'a>(mut self) -> &'a [U] where [U]: 'a {
        let r: *mut [U] = &mut self[..];
        std::mem::forget(self);
        unsafe { &mut *r }
    }
}
fn main() {
    println!("Hello, world!");
}
