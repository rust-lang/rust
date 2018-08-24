struct Foo(u8);

impl Foo {
    fn bar(&self) -> bool { self.0 > 5 }
    fn bar() {} //~ ERROR E0201
}

trait Baz {
    type Quux;
    fn baz(&self) -> bool;
}

impl Baz for Foo {
    type Quux = u32;

    fn baz(&self) -> bool { true }
    fn baz(&self) -> bool { self.0 > 5 } //~ ERROR E0201
    type Quux = u32; //~ ERROR E0201
}

fn main() {
}
