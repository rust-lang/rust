// run-pass

// Test that non-method associated functions can be specialized

#![feature(specialization)]

trait Foo {
    fn mk() -> Self;
}

impl<T: Default> Foo for T {
    default fn mk() -> T {
        T::default()
    }
}

impl Foo for Vec<u8> {
    fn mk() -> Vec<u8> {
        vec![0]
    }
}

fn main() {
    let v1: Vec<i32> = Foo::mk();
    let v2: Vec<u8> = Foo::mk();

    assert!(v1.len() == 0);
    assert!(v2.len() == 1);
}
