//@ edition: 2018..
#![feature(mut_restriction)]

pub mod inner {
    #[derive(Default)]
    pub struct Wrapper(pub mut(self) u8);

    pub enum EnumTup {
        Tup(mut(self) u8),
        Foo(u8),
        Bar(u8, mut(crate) u8),
    }

    pub fn construct_inner() {
        let _ = Wrapper;
        let _ = EnumTup::Tup;
        let _ = EnumTup::Foo;
        let _ = EnumTup::Bar;
    }
}

fn param_is_fn(_: fn(u8) -> inner::Wrapper) {}
fn param_impl_fn(_: impl Fn(u8) -> inner::Wrapper) {}

fn main() {
    let _ = inner::Wrapper;
    //~^ ERROR `Wrapper` cannot be constructed using a `struct` expression outside `crate::inner`

    param_is_fn(inner::Wrapper);
    //~^ ERROR `Wrapper` cannot be constructed using a `struct` expression outside `crate::inner`

    param_impl_fn(inner::Wrapper);
    //~^ ERROR `Wrapper` cannot be constructed using a `struct` expression outside `crate::inner`

    let _ = inner::EnumTup::Tup;
    //~^ ERROR `Tup` cannot be constructed using a `variant` expression outside `crate::inner`

    let _ = inner::EnumTup::Foo;
    let _ = inner::EnumTup::Bar;
}
