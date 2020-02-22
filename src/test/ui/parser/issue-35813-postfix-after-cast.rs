// edition:2018
#![crate_type = "lib"]
#![feature(type_ascription)]
use std::future::Future;
use std::pin::Pin;

// This tests the parser for "x as Y[z]". It errors, but we want to give useful
// errors and parse such that further code gives useful errors.
pub fn index_after_as_cast() {
    vec![1, 2, 3] as Vec<i32>[0];
    //~^ ERROR: casts cannot be followed by indexing
    vec![1, 2, 3]: Vec<i32>[0];
    //~^ ERROR: casts cannot be followed by indexing
}

pub fn index_after_cast_to_index() {
    (&[0]) as &[i32][0];
    //~^ ERROR: casts cannot be followed by indexing
    (&[0i32]): &[i32; 1][0];
    //~^ ERROR: casts cannot be followed by indexing
}

pub fn cast_after_cast() {
    if 5u64 as i32 as u16 == 0u16 {

    }
    if 5u64: u64: u64 == 0u64 {

    }
    let _ = 5u64: u64: u64 as u8 as i8 == 9i8;
    let _ = 0i32: i32: i32;
    let _ = 0 as i32: i32;
    let _ = 0i32: i32 as i32;
    let _ = 0 as i32 as i32;
    let _ = 0i32: i32: i32 as u32 as i32;
}

pub fn cast_cast_method_call() {
    let _ = 0i32: i32: i32.count_ones();
    //~^ ERROR: casts cannot be followed by a method call
    let _ = 0 as i32: i32.count_ones();
    //~^ ERROR: casts cannot be followed by a method call
    let _ = 0i32: i32 as i32.count_ones();
    //~^ ERROR: casts cannot be followed by a method call
    let _ = 0 as i32 as i32.count_ones();
    //~^ ERROR: casts cannot be followed by a method call
    let _ = 0i32: i32: i32 as u32 as i32.count_ones();
    //~^ ERROR: casts cannot be followed by a method call
    let _ = 0i32: i32.count_ones(): u32;
    //~^ ERROR: casts cannot be followed by a method call
    let _ = 0 as i32.count_ones(): u32;
    //~^ ERROR: casts cannot be followed by a method call
    let _ = 0i32: i32.count_ones() as u32;
    //~^ ERROR: casts cannot be followed by a method call
    let _ = 0 as i32.count_ones() as u32;
    //~^ ERROR: casts cannot be followed by a method call
    let _ = 0i32: i32: i32.count_ones() as u32 as i32;
    //~^ ERROR: casts cannot be followed by a method call
}

pub fn multiline_error() {
    let _ = 0
        as i32
        .count_ones();
    //~^^^ ERROR: casts cannot be followed by a method call
}

// this tests that the precedence for `!x as Y.Z` is still what we expect
pub fn precedence() {
    let x: i32 = &vec![1, 2, 3] as &Vec<i32>[0];
    //~^ ERROR: casts cannot be followed by indexing
}

pub fn method_calls() {
    0 as i32.max(0);
    //~^ ERROR: casts cannot be followed by a method call
    0: i32.max(0);
    //~^ ERROR: casts cannot be followed by a method call
}

pub fn complex() {
    let _ = format!(
        "{} and {}",
        if true { 33 } else { 44 } as i32.max(0),
        //~^ ERROR: casts cannot be followed by a method call
        if true { 33 } else { 44 }: i32.max(0)
        //~^ ERROR: casts cannot be followed by a method call
    );
}

pub fn in_condition() {
    if 5u64 as i32.max(0) == 0 {
        //~^ ERROR: casts cannot be followed by a method call
    }
    if 5u64: u64.max(0) == 0 {
        //~^ ERROR: casts cannot be followed by a method call
    }
}

pub fn inside_block() {
    let _ = if true {
        5u64 as u32.max(0) == 0
        //~^ ERROR: casts cannot be followed by a method call
    } else { false };
    let _ = if true {
        5u64: u64.max(0) == 0
        //~^ ERROR: casts cannot be followed by a method call
    } else { false };
}

static bar: &[i32] = &(&[1,2,3] as &[i32][0..1]);
//~^ ERROR: casts cannot be followed by indexing

static bar2: &[i32] = &(&[1i32,2,3]: &[i32; 3][0..1]);
//~^ ERROR: casts cannot be followed by indexing


pub fn cast_then_try() -> Result<u64,u64> {
    Err(0u64) as Result<u64,u64>?;
    //~^ ERROR: casts cannot be followed by ?
    Err(0u64): Result<u64,u64>?;
    //~^ ERROR: casts cannot be followed by ?
    Ok(1)
}


pub fn cast_then_call() {
    type F = fn(u8);
    // type ascription won't actually do [unique drop fn type] -> fn(u8) casts.
    let drop_ptr = drop as fn(u8);
    drop as F();
    //~^ ERROR: parenthesized type parameters may only be used with a `Fn` trait [E0214]
    drop_ptr: F();
    //~^ ERROR: parenthesized type parameters may only be used with a `Fn` trait [E0214]
}

pub fn cast_to_fn_should_work() {
    let drop_ptr = drop as fn(u8);
    drop as fn(u8);
    drop_ptr: fn(u8);
}

pub fn parens_after_cast_error() {
    let drop_ptr = drop as fn(u8);
    drop as fn(u8)(0);
    //~^ ERROR: casts cannot be followed by a function call
    drop_ptr: fn(u8)(0);
    //~^ ERROR: casts cannot be followed by a function call
}

pub async fn cast_then_await() {
    Box::pin(noop()) as Pin<Box<dyn Future<Output = ()>>>.await;
    //~^ ERROR: casts cannot be followed by `.await`

    Box::pin(noop()): Pin<Box<_>>.await;
    //~^ ERROR: casts cannot be followed by `.await`
}

pub async fn noop() {}

#[derive(Default)]
pub struct Foo {
    pub bar: u32,
}

pub fn struct_field() {
    Foo::default() as Foo.bar;
    //~^ ERROR: cannot be followed by a field access
    Foo::default(): Foo.bar;
    //~^ ERROR: cannot be followed by a field access
}
