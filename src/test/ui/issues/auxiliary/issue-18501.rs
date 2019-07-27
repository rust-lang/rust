#![crate_type = "rlib"]
struct Foo;

trait Tr {
    fn tr(&self);
}

impl Tr for Foo {
    fn tr(&self) {}
}

fn take_method<T>(f: fn(&T), t: &T) {}

#[inline]
pub fn pass_method() {
    take_method(Tr::tr, &Foo);
}
