// compile-flags: -Cmetadata=aux

use std::ops::Deref;

pub struct Foo;

impl Deref for Foo {
    type Target = String;
    fn deref(&self) -> &String { loop {} }
}

pub struct Bar;
pub struct Baz;

impl Baz {
    pub fn baz(&self) {}
    pub fn static_baz() {}
}

impl Deref for Bar {
    type Target = Baz;
    fn deref(&self) -> &Baz { loop {} }
}
