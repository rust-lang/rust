#![feature(type_alias_impl_trait)]

// edition: 2021
// check-pass

type MyIter = impl Iterator<Item = i32>;

struct Foo {
    iter: MyIter,
}

// ensure that `self` is seen as `Foo` and thus
// counts as a defining use for `MyIter` due to the
// struct field.
impl Foo {
    fn set_iter(&mut self) {
        self.iter = [1, 2, 3].into_iter();
    }
}

fn main() {}
