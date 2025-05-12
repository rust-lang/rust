#![feature(type_alias_impl_trait)]

type FooArg<'a> = &'a dyn ToString;

type FooItem = Box<dyn Fn(FooArg) -> FooRet>;

#[repr(C)]
struct Bar(u8);

impl Iterator for Bar {
    type Item = FooItem;

    fn next(&mut self) -> Option<Self::Item> {
        Some(Box::new(quux))
    }
}

pub type FooRet = impl std::fmt::Debug;
#[define_opaque(FooRet)]
pub fn quux(st: FooArg) -> FooRet {
    Some(st.to_string())
}
pub type Foo = impl Iterator<Item = FooItem>;
#[define_opaque(Foo)]
pub fn ham() -> Foo {
    //~^ ERROR: item does not constrain `FooRet::{opaque#0}`
    Bar(1)
}
#[define_opaque(Foo)]
pub fn oof() -> impl std::fmt::Debug {
    //~^ ERROR: item does not constrain `FooRet::{opaque#0}`
    //~| ERROR: item does not constrain `Foo::{opaque#0}`
    let mut bar = ham();
    let func = bar.next().unwrap();
    return func(&"oof");
    //~^ ERROR: opaque type's hidden type cannot be another opaque type
}

fn main() {
    let _ = oof();
}
