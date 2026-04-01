#![allow(warnings)]

struct Foo<'a, 'b, T: 'a + 'b> { x: &'a T, y: &'b T }

fn gimme(_: impl for<'a, 'b, 'c> FnOnce(&'a Foo<'a, 'b, u32>,
                                        &'a Foo<'a, 'c, u32>) -> &'a Foo<'a, 'b, u32>) { }

fn main() {
    gimme(|x, y| y)
    //~^ ERROR lifetime may not live long enough
}
