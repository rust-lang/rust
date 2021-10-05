// Regression test for issue #73159
// Tests thar we don't suggest replacing 'a with 'static'

#![feature(nll)]

struct Foo<'a>(&'a [u8]);

impl<'a> Foo<'a> {
    fn make_it(&self) -> impl Iterator<Item = u8> { //~ ERROR lifetime may not live
        self.0.iter().copied()
    }
}

fn main() {}
