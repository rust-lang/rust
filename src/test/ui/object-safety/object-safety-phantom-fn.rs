// Check that `Self` appearing in a phantom fn does not make a trait not object safe.

// compile-pass
#![allow(dead_code)]

trait Baz {
}

trait Bar<T> {
}

fn make_bar<T:Bar<u32>>(t: &T) -> &dyn Bar<u32> {
    t
}

fn make_baz<T:Baz>(t: &T) -> &dyn Baz {
    t
}


fn main() {
}
