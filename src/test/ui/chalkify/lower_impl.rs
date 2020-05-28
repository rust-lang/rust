// check-pass
// compile-flags: -Z chalk

trait Foo { }

impl<T: 'static> Foo for T where T: Iterator<Item = i32> { }

trait Bar {
    type Assoc;
}

impl<T> Bar for T where T: Iterator<Item = i32> {
    type Assoc = Vec<T>;
}

fn main() {
}
