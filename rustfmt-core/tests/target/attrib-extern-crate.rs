// Attributes on extern crate.

#[Attr1]
extern crate Bar;
#[Attr2]
#[Attr2]
extern crate Baz;
extern crate Foo;

fn foo() {
    #[Attr1]
    extern crate Bar;
    #[Attr2]
    #[Attr2]
    extern crate Baz;
    extern crate Foo;
}
