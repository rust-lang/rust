//@run-rustfix

#![feature(lint_reasons)]
#![feature(yeet_expr)]
#![allow(unused)]
#![allow(
    clippy::if_same_then_else,
    clippy::single_match,
    clippy::needless_bool,
    clippy::equatable_if_let,
    clippy::needless_else
)]
#![warn(clippy::needless_return)]

use std::cell::RefCell;

macro_rules! the_answer {
    () => {
        42
    };
}

fn test_end_of_fn() -> bool {
    if true {
        // no error!
        return true;
    }
    return true;
}

fn test_no_semicolon() -> bool {
    return true;
}

#[rustfmt::skip]
fn test_multiple_semicolon() -> bool {
    return true;;;
}

#[rustfmt::skip]
fn test_multiple_semicolon_with_spaces() -> bool {
    return true;; ; ;
}

fn test_if_block() -> bool {
    if true {
        return true;
    } else {
        return false;
    }
}

fn test_match(x: bool) -> bool {
    match x {
        true => return false,
        false => {
            return true;
        },
    }
}

fn test_closure() {
    let _ = || {
        return true;
    };
    let _ = || return true;
}

fn test_macro_call() -> i32 {
    return the_answer!();
}

fn test_void_fun() {
    return;
}

fn test_void_if_fun(b: bool) {
    if b {
        return;
    } else {
        return;
    }
}

fn test_void_match(x: u32) {
    match x {
        0 => (),
        _ => return,
    }
}

fn test_nested_match(x: u32) {
    match x {
        0 => (),
        1 => {
            let _ = 42;
            return;
        },
        _ => return,
    }
}

fn temporary_outlives_local() -> String {
    let x = RefCell::<String>::default();
    return x.borrow().clone();
}

fn borrows_but_not_last(value: bool) -> String {
    if value {
        let x = RefCell::<String>::default();
        let _a = x.borrow().clone();
        return String::from("test");
    } else {
        return String::new();
    }
}

macro_rules! needed_return {
    ($e:expr) => {
        if $e > 3 {
            return;
        }
    };
}

fn test_return_in_macro() {
    // This will return and the macro below won't be executed. Removing the `return` from the macro
    // will change semantics.
    needed_return!(10);
    needed_return!(0);
}

mod issue6501 {
    #[allow(clippy::unnecessary_lazy_evaluations)]
    fn foo(bar: Result<(), ()>) {
        bar.unwrap_or_else(|_| return)
    }

    fn test_closure() {
        let _ = || {
            return;
        };
        let _ = || return;
    }

    struct Foo;
    #[allow(clippy::unnecessary_lazy_evaluations)]
    fn bar(res: Result<Foo, u8>) -> Foo {
        res.unwrap_or_else(|_| return Foo)
    }
}

async fn async_test_end_of_fn() -> bool {
    if true {
        // no error!
        return true;
    }
    return true;
}

async fn async_test_no_semicolon() -> bool {
    return true;
}

async fn async_test_if_block() -> bool {
    if true {
        return true;
    } else {
        return false;
    }
}

async fn async_test_match(x: bool) -> bool {
    match x {
        true => return false,
        false => {
            return true;
        },
    }
}

async fn async_test_closure() {
    let _ = || {
        return true;
    };
    let _ = || return true;
}

async fn async_test_macro_call() -> i32 {
    return the_answer!();
}

async fn async_test_void_fun() {
    return;
}

async fn async_test_void_if_fun(b: bool) {
    if b {
        return;
    } else {
        return;
    }
}

async fn async_test_void_match(x: u32) {
    match x {
        0 => (),
        _ => return,
    }
}

async fn async_temporary_outlives_local() -> String {
    let x = RefCell::<String>::default();
    return x.borrow().clone();
}

async fn async_borrows_but_not_last(value: bool) -> String {
    if value {
        let x = RefCell::<String>::default();
        let _a = x.borrow().clone();
        return String::from("test");
    } else {
        return String::new();
    }
}

async fn async_test_return_in_macro() {
    needed_return!(10);
    needed_return!(0);
}

fn let_else() {
    let Some(1) = Some(1) else { return };
}

fn needless_return_macro() -> String {
    let _ = "foo";
    let _ = "bar";
    return format!("Hello {}", "world!");
}

fn issue_9361() -> i32 {
    let n = 1;
    #[allow(clippy::arithmetic_side_effects)]
    return n + n;
}

fn issue8336(x: i32) -> bool {
    if x > 0 {
        println!("something");
        return true;
    } else {
        return false;
    };
}

fn issue8156(x: u8) -> u64 {
    match x {
        80 => {
            return 10;
        },
        _ => {
            return 100;
        },
    };
}

// Ideally the compiler should throw `unused_braces` in this case
fn issue9192() -> i32 {
    {
        return 0;
    };
}

fn issue9503(x: usize) -> isize {
    unsafe {
        if x > 12 {
            return *(x as *const isize);
        } else {
            return !*(x as *const isize);
        };
    };
}

mod issue9416 {
    pub fn with_newline() {
        let _ = 42;

        return;
    }

    #[rustfmt::skip]
    pub fn oneline() {
        let _ = 42; return;
    }
}

fn issue9947() -> Result<(), String> {
    do yeet "hello";
}

// without anyhow, but triggers the same bug I believe
#[expect(clippy::useless_format)]
fn issue10051() -> Result<String, String> {
    if true {
        return Ok(format!("ok!"));
    } else {
        return Err(format!("err!"));
    }
}

mod issue10049 {
    fn single() -> u32 {
        return if true { 1 } else { 2 };
    }

    fn multiple(b1: bool, b2: bool, b3: bool) -> u32 {
        return if b1 { 0 } else { 1 } | if b2 { 2 } else { 3 } | if b3 { 4 } else { 5 };
    }
}

fn test_match_as_stmt() {
    let x = 9;
    match x {
        1 => 2,
        2 => return,
        _ => 0,
    };
}

fn main() {}
