#![feature(type_alias_impl_trait)]

// edition: 2021
// check-pass

type MyIter = impl Iterator<Item = i32>;

struct Foo {
    iter: MyIter,
}

impl Foo {
    #[defines(MyIter)]
    fn set_iter(&mut self) {
        self.iter = [1, 2, 3].into_iter();
    }
}

fn main() {}
