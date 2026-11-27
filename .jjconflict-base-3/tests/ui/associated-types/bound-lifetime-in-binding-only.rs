//@ revisions: angle paren ok elision
//@[ok] check-pass

#![allow(dead_code)]
#![feature(unboxed_closures)]

trait Foo {
    type Item;
}

#[cfg(angle)]
fn angle<T: for<'a> Foo<Item=&'a i32>>() {
    //[angle]~^ ERROR binding for associated type `Item` references lifetime `'a`
}

#[cfg(angle)]
fn angle1<T>() where T: for<'a> Foo<Item=&'a i32> {
    //[angle]~^ ERROR binding for associated type `Item` references lifetime `'a`
}

#[cfg(angle)]
fn angle2<T>() where for<'a> T: Foo<Item=&'a i32> {
    //[angle]~^ ERROR binding for associated type `Item` references lifetime `'a`
}

#[cfg(angle)]
fn angle3(_: &dyn for<'a> Foo<Item=&'a i32>) {
    //[angle]~^ ERROR binding for associated type `Item` references lifetime `'a`
}

#[cfg(paren)]
fn paren<T: for<'a> Fn() -> &'a i32>() {
    //[paren]~^ ERROR binding for associated type `Output` references lifetime `'a`
}

#[cfg(paren)]
fn paren1<T>() where T: for<'a> Fn() -> &'a i32 {
    //[paren]~^ ERROR binding for associated type `Output` references lifetime `'a`
}

#[cfg(paren)]
fn paren2<T>() where for<'a> T: Fn() -> &'a i32 {
    //[paren]~^ ERROR binding for associated type `Output` references lifetime `'a`
}

#[cfg(paren)]
fn paren3(_: &dyn for<'a> Fn() -> &'a i32) {
    //[paren]~^ ERROR binding for associated type `Output` references lifetime `'a`
}

#[cfg(elision)]
fn elision<T: Fn() -> &i32>() {
    //[elision]~^ ERROR E0106
}

struct Parameterized<'a> { x: &'a str }

#[cfg(ok)]
fn ok1<T: for<'a> Fn(&Parameterized<'a>) -> &'a i32>() {
}

#[cfg(ok)]
fn ok2<T: for<'a,'b> Fn<(&'b Parameterized<'a>,), Output=&'a i32>>() {
}

#[cfg(ok)]
fn ok3<T>() where for<'a> Parameterized<'a>: Foo<Item=&'a i32> {
}

fn main() { }
