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
        //~| WARN `#[doc]` attribute cannot be used on function params
        //~| WARN: this was previously accepted by the compiler
        #[doc(alias = "stmt")]
        //~^ ERROR
        //~| WARN `#[doc]` attribute cannot be used on statements
        //~| WARN: this was previously accepted by the compiler
        let x = 0;
        #[doc(alias = "expr")]
        //~^ ERROR
        //~| WARN `#[doc]` attribute cannot be used on expressions
        //~| WARN: this was previously accepted by the compiler
        match x {
            #[doc(alias = "arm")]
            //~^ ERROR
            //~| WARN `#[doc]` attribute cannot be used on match arms
            //~| WARN: this was previously accepted by the compiler
            _ => 0
        }
    }
}
