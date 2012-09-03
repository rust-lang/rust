trait Foo {
    fn foo(self);
}

impl &[int]: Foo {
    fn foo(self) {}
}

fn main() {
    let items = ~[ 3, 5, 1, 2, 4 ];
    items.foo();
}

