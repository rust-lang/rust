//@ check-pass

#![deny(dead_code)]

#[derive(Default)]
struct Test {

}

fn main() {
    if let Some::<Test>(test) = magic::<Test>() { }
}

fn magic<T: Default>() -> Option<T> {
    Some(T::default())
}
