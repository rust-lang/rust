#![feature(arbitrary_self_types)]

#[derive(Debug)]
struct Foo;
#[derive(Debug)]
struct Bar<'a>(&'a Foo);

impl std::ops::Deref for Bar<'_> {
    type Target = Foo;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Foo {
    fn a(self: &Box<Foo>, f: &Foo) -> &Foo { f } //~ ERROR E0106

    fn b(self: &Box<Foo>, f: &Foo) -> &Box<Foo> { self } //~ ERROR E0106

    fn c(this: &Box<Foo>, f: &Foo) -> &Foo { f } //~ ERROR E0106
}

impl<'a> Bar<'a> {
    fn d(self: Self, f: &Foo, g: &Foo) -> (Bar<'a>, &Foo) { (self, f) } //~ ERROR E0106
}

fn main() {}
