#![allow(unused, bare_trait_objects)]
#[repr(align(256))]
struct A {
    v: u8,
}

trait Foo {
    fn foo(&self);
}

impl Foo for A {
    fn foo(&self) {
        assert_eq!(self as *const A as usize % 256, 0);
    }
}

fn foo(x: dyn Foo) { //~ ERROR [E0277]
    x.foo()
}

fn bar(x: Foo) { //~ ERROR [E0277]
    x.foo()
}

fn qux(_: [()]) {} //~ ERROR [E0277]

fn main() {
    let x: Box<dyn Foo> = Box::new(A { v: 22 });
    foo(*x); //~ ERROR [E0277]
}
