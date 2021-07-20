// compile-flags: --test

// Issue #55228

#![feature(termination_trait_lib)]

use std::process::Termination;

#[test]
fn unit_1() -> Result<(), &'static str> { // this is OK
    Ok(())
}

#[test]
fn unit_2<'a>() -> Result<(), &'a str> { // also OK
    Ok(())
}

#[test]
fn unit_3<T: Termination>() -> Result<(), T> { // nope (how would this get monomorphized?)
    //~^ ERROR return value of functions used as tests must either be `()` or implement
    Ok(())
}


struct Quux;

trait Bar {
    type Baz;

    fn rah(&self) -> Self::Baz;
}

impl Bar for Quux {
    type Baz = String;

    fn rah(&self) -> Self::Baz {
        "rah".to_owned()
    }
}

#[test]
fn unit_4<T: Bar<Baz = String>>() -> <T as Bar>::Baz { // also nope
    //~^ ERROR return value of functions used as tests must either be `()` or implement
    let q = Quux{};
    <Quux as Bar>::rah(&q)
}
