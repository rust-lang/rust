// check-pass
// compile-flags: -Z chalk

trait Bar { }

trait Foo<S, T: ?Sized> {
    type Assoc: Bar + ?Sized;
}

fn main() {
}
