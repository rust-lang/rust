// run-rustfix

#![warn(clippy::needless_question_mark)]
#![allow(
    clippy::needless_return,
    clippy::unnecessary_unwrap,
    clippy::upper_case_acronyms,
    dead_code,
    unused_must_use
)]
#![feature(custom_inner_attributes)]

struct TO {
    magic: Option<usize>,
}

struct TR {
    magic: Result<usize, bool>,
}

fn simple_option_bad1(to: TO) -> Option<usize> {
    // return as a statement
    return Some(to.magic?);
}

// formatting will add a semi-colon, which would make
// this identical to the test case above
#[rustfmt::skip]
fn simple_option_bad2(to: TO) -> Option<usize> {
    // return as an expression
    return Some(to.magic?)
}

fn simple_option_bad3(to: TO) -> Option<usize> {
    // block value "return"
    Some(to.magic?)
}

fn simple_option_bad4(to: Option<TO>) -> Option<usize> {
    // single line closure
    to.and_then(|t| Some(t.magic?))
}

// formatting this will remove the block brackets, making
// this test identical to the one above
#[rustfmt::skip]
fn simple_option_bad5(to: Option<TO>) -> Option<usize> {
    // closure with body
    to.and_then(|t| {
        Some(t.magic?)
    })
}

fn simple_result_bad1(tr: TR) -> Result<usize, bool> {
    return Ok(tr.magic?);
}

// formatting will add a semi-colon, which would make
// this identical to the test case above
#[rustfmt::skip]
fn simple_result_bad2(tr: TR) -> Result<usize, bool> {
    return Ok(tr.magic?)
}

fn simple_result_bad3(tr: TR) -> Result<usize, bool> {
    Ok(tr.magic?)
}

fn simple_result_bad4(tr: Result<TR, bool>) -> Result<usize, bool> {
    tr.and_then(|t| Ok(t.magic?))
}

// formatting this will remove the block brackets, making
// this test identical to the one above
#[rustfmt::skip]
fn simple_result_bad5(tr: Result<TR, bool>) -> Result<usize, bool> {
    tr.and_then(|t| {
        Ok(t.magic?)
    })
}

fn also_bad(tr: Result<TR, bool>) -> Result<usize, bool> {
    if tr.is_ok() {
        let t = tr.unwrap();
        return Ok(t.magic?);
    }
    Err(false)
}

fn false_positive_test<U, T>(x: Result<(), U>) -> Result<(), T>
where
    T: From<U>,
{
    Ok(x?)
}

// not quite needless
fn deref_ref(s: Option<&String>) -> Option<&str> {
    Some(s?)
}

fn main() {}

// #6921 if a macro wraps an expr in Some(  ) and the ? is in the macro use,
// the suggestion fails to apply; do not lint
macro_rules! some_in_macro {
    ($expr:expr) => {
        || -> _ { Some($expr) }()
    };
}

pub fn test1() {
    let x = Some(3);
    let _x = some_in_macro!(x?);
}

// this one is ok because both the ? and the Some are both inside the macro def
macro_rules! some_and_qmark_in_macro {
    ($expr:expr) => {
        || -> Option<_> { Some(Some($expr)?) }()
    };
}

pub fn test2() {
    let x = Some(3);
    let _x = some_and_qmark_in_macro!(x?);
}

async fn async_option_bad(to: TO) -> Option<usize> {
    let _ = Some(3);
    Some(to.magic?)
}

async fn async_deref_ref(s: Option<&String>) -> Option<&str> {
    Some(s?)
}

async fn async_result_bad(s: TR) -> Result<usize, bool> {
    Ok(s.magic?)
}
