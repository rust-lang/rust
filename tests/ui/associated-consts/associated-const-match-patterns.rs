// run-pass
// aux-build:empty-struct.rs


extern crate empty_struct;
use empty_struct::XEmpty2 as XFoo;

struct Foo;

#[derive(PartialEq, Eq)]
enum Bar {
    Var1,
    Var2,
}

// Use inherent and trait impls to test UFCS syntax.
impl Foo {
    const MYBAR: Bar = Bar::Var2;
}

trait HasBar {
    const THEBAR: Bar;
}

impl HasBar for Foo {
    const THEBAR: Bar = Bar::Var1;
}

impl HasBar for XFoo {
    const THEBAR: Bar = Bar::Var1;
}

fn main() {
    // Inherent impl
    assert!(match Bar::Var2 {
        Foo::MYBAR => true,
        _ => false,
    });
    assert!(match Bar::Var2 {
        <Foo>::MYBAR => true,
        _ => false,
    });
    // Trait impl
    assert!(match Bar::Var1 {
        Foo::THEBAR => true,
        _ => false,
    });
    assert!(match Bar::Var1 {
        <Foo>::THEBAR => true,
        _ => false,
    });
    assert!(match Bar::Var1 {
        <Foo as HasBar>::THEBAR => true,
        _ => false,
    });
    assert!(match Bar::Var1 {
        XFoo::THEBAR => true,
        _ => false,
    });
    assert!(match Bar::Var1 {
        <XFoo>::THEBAR => true,
        _ => false,
    });
    assert!(match Bar::Var1 {
        <XFoo as HasBar>::THEBAR => true,
        _ => false,
    });
}
