#![forbid(unsafe_code)]

trait Foo {
    unsafe fn dangerous();
    //~^ ERROR declaration of an `unsafe` method [unsafe_obligation_define]
}

struct ImplOk;
impl Foo for ImplOk {
    #[forbid(unsafe_op_in_unsafe_fn)]
    unsafe fn dangerous() {}
    //~^ ERROR implementation of an `unsafe` method [unsafe_obligation_define]
}

struct ImplBad;
impl Foo for ImplBad {
    unsafe fn dangerous() {}
    //~^ ERROR implementation of an `unsafe` method [unsafe_obligation_discharge]
}

fn main() {}
