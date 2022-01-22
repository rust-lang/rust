use std::pin::Pin;

struct Foo;

impl Foo {
    fn foo(self: Pin<&mut Self>) {}
}

fn main() {
    let mut foo = Foo;
    let foo = Pin::new(&mut foo);
    foo.foo();
    //~^ HELP consider calling `.as_mut()` to borrow the type's contents
    foo.foo();
    //~^ ERROR use of moved value

    let mut x = 1;
    let mut x = Pin::new(&mut x);
    x.get_mut();
    //~^ HELP consider calling `.as_mut()` to borrow the type's contents
    x.get_mut();
    //~^ ERROR use of moved value
}
