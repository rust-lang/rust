trait Foo {
    fn foo(self);
}

impl<'a> Foo for &'a [isize] {
    fn foo(self) {}
}

pub fn main() {
    let items = vec![ 3, 5, 1, 2, 4 ];
    items.foo();
}
