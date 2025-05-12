#![deny(redundant_lifetimes)]

fn a<'a, 'b>(x: &'a &'b &'a ()) {} //~ ERROR unnecessary lifetime parameter `'b`

fn b<'a: 'b, 'b: 'a>() {} //~ ERROR unnecessary lifetime parameter `'b`

struct Foo<T: 'static>(T);
fn c<'a>(_: Foo<&'a ()>) {} //~ ERROR unnecessary lifetime parameter `'a`

struct Bar<'a>(&'a ());
impl<'a> Bar<'a> {
    fn d<'b: 'a>(&'b self) {} //~ ERROR unnecessary lifetime parameter `'b`
}

fn ok(x: &'static &()) {}

trait Tr<'a> {}
impl<'a: 'static> Tr<'a> for () {} //~ ERROR unnecessary lifetime parameter `'a`

fn main() {}
