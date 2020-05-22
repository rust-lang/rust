// check-pass
// compile-flags: -Z chalk

#![allow(dead_code)]

trait Foo { }

trait Bar where Self: Foo { }

fn bar<T: Bar + ?Sized>() {
}

fn main() {
}
