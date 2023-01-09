// check-pass
// compile-flags: -Z trait-solver=chalk

trait Bar { }

trait Foo<S, T: ?Sized> {
    type Assoc: Bar + ?Sized;
}

fn main() {
}
