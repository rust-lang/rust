// compile-fail

use std::pin::Pin;

struct Foo;

impl Foo {
    fn a(self: Pin<&Foo>, f: &Foo) -> &Foo { f } //~ ERROR E0623

    fn c(self: Pin<&Self>, f: &Foo, g: &Foo) -> (Pin<&Foo>, &Foo) { (self, f) } //~ ERROR E0623
}

fn main() {}
