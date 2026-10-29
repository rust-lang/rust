#![feature(supertrait_item_shadowing)]
#![deny(resolving_to_items_shadowing_supertrait_items)]
#![allow(dead_code)]

trait Base {
    fn method(&self) {}
    fn function() {}
    const VALUE: u8 = 0;
}
trait Sub: Base {
    fn method(&self) {}
    fn function() {}
    const VALUE: u8 = 1;
}
impl Base for () {}
impl Sub for () {}

fn uses<T: Sub>(x: &T) {
    x.method();
    //~^ ERROR trait item `method` from `Sub` shadows identically named item
    T::method(x);
    //~^ ERROR trait item `method` from `Sub` shadows identically named item
    T::function();
    //~^ ERROR trait item `function` from `Sub` shadows identically named item
    let _ = T::function;
    //~^ ERROR trait item `function` from `Sub` shadows identically named item
    let _ = T::VALUE;
    //~^ ERROR trait item `VALUE` from `Sub` shadows identically named item

    <T as Sub>::method(x);
    <T as Sub>::function();
    let _ = <T as Sub>::VALUE;
}

fn pattern(x: u8) {
    match x {
        <()>::VALUE => {}
        //~^ ERROR trait item `VALUE` from `Sub` shadows identically named item
        _ => {}
    }
}

#[allow(resolving_to_items_shadowing_supertrait_items)]
fn allowed<T: Sub>(x: &T) {
    x.method();
    T::method(x);
    let _ = T::VALUE;
}

// Error reporting may probe and confirm a hypothetical method call.
// That must not emit the shadowing lint.
fn diagnostic_only<T: Sub>(x: &T) {
    method(x);
    //~^ ERROR cannot find function `method` in this scope
}

// Distinct instantiations of Parent must not produce repeated diagnostic notes.
trait Parent<T> {
    fn repeated(&self) {}
}
trait Winner: Parent<u8> + Parent<u16> {
    fn repeated(&self) {}
}
fn repeated<T: Winner>(x: &T) {
    x.repeated();
    //~^ ERROR trait item `repeated` from `Winner` shadows identically named item
    T::repeated(x);
    //~^ ERROR trait item `repeated` from `Winner` shadows identically named item
}

fn main() {}
