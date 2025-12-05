use std::pin::Pin;

struct Foo;

impl Foo {
    fn a(self: Pin<&Foo>, f: &Foo) -> &Foo {
        f //~ ERROR lifetime may not live long enough
    }

    // For this suggestion to be right, we'd need to also suggest `self: Pin<&'a Self>`, which we
    // don't, but we provide a follow up suggestion to do so, so I condider that good at least for
    // now.
    fn c(self: Pin<&Self>, f: &Foo, g: &Foo) -> (Pin<&Foo>, &Foo) {
        (self, f) //~ ERROR lifetime may not live long enough
    }
}

type Alias<T> = Pin<T>;
impl Foo {
    fn bar<'a>(self: Alias<&Self>, arg: &'a ()) -> &() {
        arg //~ ERROR lifetime may not live long enough
    }
}

fn main() {}
