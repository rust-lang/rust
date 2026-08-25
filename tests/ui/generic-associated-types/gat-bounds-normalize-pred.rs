//@ check-pass

trait Foo {
    type Assoc<T>: PartialEq<Self::Assoc<i32>>;
}

impl Foo for () {
    type Assoc<T> = Wrapper<T>;
}

struct Wrapper<T>(T);

impl<T> PartialEq<Wrapper<i32>> for Wrapper<T> {
    fn eq(&self, _other: &Wrapper<i32>) -> bool { true }
}

fn main() {}
