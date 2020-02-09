// edition:2018
#![crate_type = "lib"]
use std::future::Future;
use std::pin::Pin;

// This tests the parser for "x as Y[z]". It errors, but we want to give useful
// errors and parse such that further code gives useful errors.
pub fn index_after_as_cast() {
    vec![1, 2, 3] as Vec<i32>[0];
    //~^ ERROR: casts followed by index operators are not supported
}

pub fn index_after_cast_to_index() {
    (&[0]) as &[i32][0];
    //~^ ERROR: casts followed by index operators are not supported
}

// this tests that the precedence for `!x as Y.Z` is still what we expect
pub fn precedence() {
    let x: i32 = &vec![1, 2, 3] as &Vec<i32>[0];
    //~^ ERROR: casts followed by index operators are not supported
}

pub fn complex() {
    let _ = format!(
        "{}",
        if true { 33 } else { 44 } as i32.max(0)
        //~^ ERROR: casts followed by method call expressions are not supported
    );
}

pub fn in_condition() {
    if 5u64 as i32.max(0) == 0 {
        //~^ ERROR: casts followed by method call expressions are not supported
    }
}

pub fn inside_block() {
    let _ = if true {
        5u64 as u32.max(0) == 0
        //~^ ERROR: casts followed by method call expressions are not supported
    } else { false };
}

static bar: &[i32] = &(&[1,2,3] as &[i32][0..1]);
//~^ ERROR: casts followed by index operators are not supported

pub async fn cast_then_await() {
    Box::pin(noop()) as Pin<Box<dyn Future<Output = ()>>>.await;
    //~^ ERROR: casts followed by awaits are not supported
}

pub async fn noop() {}

#[derive(Default)]
pub struct Foo {
    pub bar: u32,
}

pub fn struct_field() {
    Foo::default() as Foo.bar;
    //~^ ERROR: casts followed by field access expressions are not supported
}
