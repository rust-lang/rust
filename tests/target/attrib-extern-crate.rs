// Attributes on extern crate.

extern crate Foo;
#[Attr1]
extern crate Bar;
#[Attr2]
#[Attr2]
extern crate Baz;

fn foo() {
    extern crate Foo;
    #[Attr1]
    extern crate Bar;
    #[Attr2]
    #[Attr2]
    extern crate Baz;
}
