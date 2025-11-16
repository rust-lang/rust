//@ check-pass

use std::num::NonZero;

trait Foo<T>: Sized {
    fn bar(self, other: T) {}
}

impl Foo<u8> for u8 {}
impl Foo<u16> for u16 {}

trait Bar {}
impl Bar for u8 {}
impl Bar for u16 {}
fn foo(_: impl Bar) {}

fn main() {
    // Check that we can coerce
    let x = NonZero::new(5_u8).unwrap();
    let y: u8 = x;

    // Can coerce by looking at the trait
    let x = NonZero::new(5_u8).unwrap();
    5_u8.bar(x);

    // Check that we can infer the nonzero wrapped type through the coercion
    let a = NonZero::new(5).unwrap();
    let b: u8 = a;

    let a = NonZero::new(5).unwrap();
    5_u8.bar(a);
}
