// Make sure the generated suggestion suggest editing the user code instead of
// the macro implementation (which might come from an external crate).
// issue: <https://github.com/rust-lang/rust/issues/139049>

//@ run-rustfix

#![allow(dead_code)]

// You could assume that this comes from an extern crate (it doesn't
// because an aux crate would be overkill for this test).
macro_rules! perform { ($e:expr) => { D(&$e).end() } }
//~^ ERROR does not live long enough
//~| ERROR does not live long enough

fn main() {
    { let l = (); perform!(l) };
    //~^ SUGGESTION ;

    let _x = { let l = (); perform!(l) };
    //~^ SUGGESTION let x
}

struct D<T>(T);
impl<T> Drop for D<T> { fn drop(&mut self) {} }
impl<T> D<T> { fn end(&self) -> String { String::new() } }
