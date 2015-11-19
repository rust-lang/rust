#![feature(plugin)]
#![plugin(clippy)]

#![allow(unused)]
#![deny(clippy, clippy_pedantic)]

use std::ops::Mul;

struct T;

impl T {
    fn add(self, other: T) -> T { self } //~ERROR defining a method called `add`
    fn drop(&mut self) { } //~ERROR defining a method called `drop`

    fn sub(&self, other: T) -> &T { self } // no error, self is a ref
    fn div(self) -> T { self } // no error, different #arguments
    fn rem(self, other: T) { } // no error, wrong return type

    fn into_u32(self) -> u32 { 0 } // fine
    fn into_u16(&self) -> u16 { 0 } //~ERROR methods called `into_*` usually take self by value

    fn to_something(self) -> u32 { 0 } //~ERROR methods called `to_*` usually take self by reference
}

#[derive(Clone,Copy)]
struct U;

impl U {
    fn to_something(self) -> u32 { 0 } // ok because U is Copy
}

impl Mul<T> for T {
    type Output = T;
    fn mul(self, other: T) -> T { self } // no error, obviously
}

fn main() {
    use std::io;

    let opt = Some(0);
    let _ = opt.unwrap();  //~ERROR used unwrap() on an Option

    let res: Result<i32, ()> = Ok(0);
    let _ = res.unwrap();  //~ERROR used unwrap() on a Result

    let _ = "str".to_string();  //~ERROR `"str".to_owned()` is faster

    let v = &"str";
    let string = v.to_string();  //~ERROR `(*v).to_owned()` is faster
    let _again = string.to_string();  //~ERROR `String.to_string()` is a no-op

    res.ok().expect("disaster!"); //~ERROR called `ok().expect()`
    // the following should not warn, since `expect` isn't implemented unless
    // the error type implements `Debug`
    let res2: Result<i32, MyError> = Ok(0);
    res2.ok().expect("oh noes!");
    // we're currently don't warn if the error type has a type parameter
    // (but it would be nice if we did)
    let res3: Result<u32, MyErrorWithParam<u8>>= Ok(0);
    res3.ok().expect("whoof");
    let res4: Result<u32, io::Error> = Ok(0);
    res4.ok().expect("argh"); //~ERROR called `ok().expect()`
    let res5: io::Result<u32> = Ok(0);
    res5.ok().expect("oops"); //~ERROR called `ok().expect()`
}

struct MyError(()); // doesn't implement Debug

#[derive(Debug)]
struct MyErrorWithParam<T> {
    x: T
}
