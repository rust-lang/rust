#![crate_type = "lib"]

pub struct Bar;
pub trait Foo {
    type X;
    fn foo(x: u32) -> Self::X;
}

#[doc(alias = "foo")] //~ ERROR
extern "C" {}

#[doc(alias = "bar")] //~ ERROR
impl Bar {
    #[doc(alias = "const")]
    const A: u32 = 0;
}

#[doc(alias = "foobar")] //~ ERROR
impl Foo for Bar {
    #[doc(alias = "assoc")] //~ ERROR
    type X = i32;
    fn foo(#[doc(alias = "qux")] _x: u32) -> Self::X {
        //~^ ERROR
        #[doc(alias = "stmt")] //~ ERROR
        let x = 0;
        #[doc(alias = "expr")] //~ ERROR
        match x {
            #[doc(alias = "arm")] //~ ERROR
            _ => 0
        }
    }
}
