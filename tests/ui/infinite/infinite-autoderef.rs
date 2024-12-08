use std::ops::Deref;

struct Foo;

impl Deref for Foo {
    type Target = Foo;

    fn deref(&self) -> &Foo {
        self
    }
}

pub fn main() {
    let mut x;
    loop {
        x = Box::new(x);
        //~^ ERROR overflow assigning `Box<_>` to `_`
        x.foo;
        x.bar();
    }

    Foo.foo;
    Foo.bar();
}
