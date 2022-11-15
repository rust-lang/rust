// run-pass

pub trait Identity {
    type Identity;
}

impl<T> Identity for T {
    type Identity = Self;
}

pub fn foo<const X: <i32 as Identity>::Identity>() {
    assert!(X == 12);
}

fn main() {
    foo::<12>();
}
