#![allow(many_single_char_names, blacklisted_name)]

#[derive(Copy, Clone)]
struct Foo(u32);

#[derive(Copy, Clone)]
struct Bar([u8; 24]);

type Baz = u32;

fn good(a: &mut u32, b: u32, c: &Bar) {
}

fn good_return_implicit_lt_ref(foo: &Foo) -> &u32 {
    &foo.0
}

#[allow(needless_lifetimes)]
fn good_return_explicit_lt_ref<'a>(foo: &'a Foo) -> &'a u32 {
    &foo.0
}

fn bad(x: &u32, y: &Foo, z: &Baz) {
}

impl Foo {
    fn good(self, a: &mut u32, b: u32, c: &Bar) {
    }

    fn good2(&mut self) {
    }

    fn bad(&self, x: &u32, y: &Foo, z: &Baz) {
    }

    fn bad2(x: &u32, y: &Foo, z: &Baz) {
    }
}

impl AsRef<u32> for Foo {
    fn as_ref(&self) -> &u32 {
        &self.0
    }
}

impl Bar {
    fn good(&self, a: &mut u32, b: u32, c: &Bar) {
    }

    fn bad2(x: &u32, y: &Foo, z: &Baz) {
    }
}

fn main() {
    let (mut foo, bar) = (Foo(0), Bar([0; 24]));
    let (mut a, b, c, x, y, z) = (0, 0, Bar([0; 24]), 0, Foo(0), 0);
    good(&mut a, b, &c);
    good_return_implicit_lt_ref(&y);
    good_return_explicit_lt_ref(&y);
    bad(&x, &y, &z);
    foo.good(&mut a, b, &c);
    foo.good2();
    foo.bad(&x, &y, &z);
    Foo::bad2(&x, &y, &z);
    bar.good(&mut a, b, &c);
    Bar::bad2(&x, &y, &z);
    foo.as_ref();
}
