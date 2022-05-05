// Test the simplest of outlives suggestions.

#![feature(nll)]

fn foo1<'a, 'b>(x: &'a usize) -> &'b usize {
    x //~ERROR lifetime may not live long enough
}

fn foo2<'a>(x: &'a usize) -> &'static usize {
    x //~ERROR lifetime may not live long enough
}

fn foo3<'a, 'b>(x: &'a usize, y: &'b usize) -> (&'b usize, &'a usize) {
    (x, y) //~ERROR lifetime may not live long enough
           //~^ERROR lifetime may not live long enough
}

fn foo4<'a, 'b, 'c>(x: &'a usize) -> (&'b usize, &'c usize) {
    // FIXME: ideally, we suggest 'a: 'b + 'c, but as of today (may 04, 2019), the null error
    // reporting stops after the first error in a MIR def so as not to produce too many errors, so
    // currently we only report 'a: 'b. The user would then re-run and get another error.
    (x, x) //~ERROR lifetime may not live long enough
}

struct Foo<'a> {
    x: &'a usize,
}

impl Foo<'static> {
    pub fn foo<'a>(x: &'a usize) -> Self {
        Foo { x } //~ERROR lifetime may not live long enough
    }
}

struct Bar<'a> {
    x: &'a usize,
}

impl<'a> Bar<'a> {
    pub fn get<'b>(&self) -> &'b usize {
        self.x //~ERROR lifetime may not live long enough
    }
}

// source: https://stackoverflow.com/questions/41417057/why-do-i-get-a-lifetime-error-when-i-use-a-mutable-reference-in-a-struct-instead
struct Baz<'a> {
    x: &'a mut i32,
}

impl<'a> Baz<'a> {
    fn get<'b>(&'b self) -> &'a i32 {
        self.x //~ERROR lifetime may not live long enough
    }
}

// source: https://stackoverflow.com/questions/41204134/rust-lifetime-error
struct Bar2<'a> {
    bar: &'a str,
}
impl<'a> Bar2<'a> {
    fn new(foo: &'a Foo2<'a>) -> Bar2<'a> {
        Bar2 { bar: foo.raw }
    }
}

pub struct Foo2<'a> {
    raw: &'a str,
    cell: std::cell::Cell<&'a str>,
}
impl<'a> Foo2<'a> {
    // should not produce outlives suggestions to name 'self
    fn get_bar(&self) -> Bar2 {
        Bar2::new(&self) //~ERROR lifetime may not live long enough
    }
}

fn main() {}
