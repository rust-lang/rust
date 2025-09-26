// This test ensures that patterns also get a link generated.

//@ compile-flags: -Zunstable-options --generate-link-to-definition

#![crate_name = "foo"]

//@ has 'src/foo/jump-to-def-assoc-items.rs.html'

pub trait Trait {
    type T;
}
pub trait Another {
    type T;
    const X: u32;
}

pub struct Foo;

impl Foo {
    pub fn new() -> Self { Foo }
}

pub struct C;

impl C {
    pub fn wat() {}
}

pub struct Bar;
impl Trait for Bar {
    type T = Foo;
}
impl Another for Bar {
    type T = C;
    const X: u32 = 12;
}

pub fn bar() {
    //@ has - '//a[@href="#20"]' 'new'
    <Bar as Trait>::T::new();
    //@ has - '//a[@href="#26"]' 'wat'
    <Bar as Another>::T::wat();

    match 12u32 {
        //@ has - '//a[@href="#14"]' 'X'
        <Bar as Another>::X => {}
        _ => {}
    }
}

pub struct Far {
        //@ has - '//a[@href="#10"]' 'T'
    x: <Bar as Trait>::T,
}
